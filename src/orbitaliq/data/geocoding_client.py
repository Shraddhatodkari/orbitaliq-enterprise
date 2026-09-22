"""Real street-address geocoding via the US Census Bureau's Geocoder —
free, keyless, US-government-authoritative.

Used only to turn a company's genuine SEC-EDGAR-registered business
address (see ``sec_edgar_client.py::SecEdgarClient.fetch_business_address``)
into verified latitude/longitude for the "resolve by company name" workflow
(``data/company_resolver.py``). This client never guesses, interpolates, or
falls back to an approximate coordinate: an address that doesn't match
anything (non-US, unrecognized, or the service being unreachable) correctly
returns a typed ``INSUFFICIENT_DATA`` result, never a fabricated lat/lon.

API docs: https://www.census.gov/programs-surveys/geography/technical-documentation/complete-technical-documentation/census-geocoder.html
No account, no API key — the one thing SEC EDGAR requires (a descriptive
User-Agent) doesn't even apply here.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from orbitaliq.config import Settings, get_settings
from orbitaliq.core.data_quality_state import classify_state
from orbitaliq.core.logging_config import logger

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


@dataclass
class GeocodeResult:
    """A single geocoding attempt's outcome. ``latitude``/``longitude`` are
    ``None`` exactly when ``status == "INSUFFICIENT_DATA"`` — this type has
    no third state and never carries a guessed coordinate.
    """

    latitude: float | None
    longitude: float | None
    matched_address: str | None
    status: str  # "AVAILABLE" | "INSUFFICIENT_DATA"
    source: str
    note: str | None = None
    # See core/data_quality_state.py's 5-state contract — always derived
    # from `source` via classify_state() at construction time.
    quality_state: str = "LIVE"

    def as_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "matched_address": self.matched_address,
            "status": self.status,
            "source": self.source,
            "note": self.note,
            "quality_state": self.quality_state,
        }


def _insufficient(reason: str) -> GeocodeResult:
    source = "insufficient_data"
    return GeocodeResult(
        latitude=None, longitude=None, matched_address=None, status="INSUFFICIENT_DATA",
        source=source, note=reason, quality_state=classify_state(source).value,
    )


def _error(reason: str, *, error_type: str) -> GeocodeResult:
    logger.warning(f"Census geocoder ERROR: {reason}")
    source = f"error:census_geocoder:{error_type}"
    return GeocodeResult(
        latitude=None, longitude=None, matched_address=None, status="INSUFFICIENT_DATA",
        source=source, note=reason, quality_state=classify_state(source).value,
    )


class CensusGeocodingClient:
    """Free, keyless, US-government-authoritative one-line-address
    geocoder client. Every method returns a typed ``GeocodeResult`` —
    never raises for an ordinary "no match"/"unreachable" outcome, and
    never returns a coordinate it isn't reporting as ``AVAILABLE``.
    """

    def __init__(self, settings: Settings | None = None, http_client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=self._settings.live_data_timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def _get(self, url: str, params: dict) -> httpx.Response:
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response

    def geocode_address(self, one_line_address: str) -> GeocodeResult:
        """Geocode a single free-text US street address (e.g.
        ``"1 Tesla Road, Austin, TX 78725"``) into verified lat/lon via the
        Census Bureau's ``onelineaddress`` endpoint.

        Returns the FIRST entry in the Census Bureau's own ranked
        ``addressMatches`` list — this client never re-ranks or second-
        guesses which match is "right", and never invents a result when the
        address doesn't match anything (e.g. a non-US address, or a
        genuinely malformed one). A network/parse failure is reported as
        ``ERROR`` (see ``core/data_quality_state.py``), distinct from the
        clean "no match" ``INSUFFICIENT_DATA`` case.
        """
        if not one_line_address or not one_line_address.strip():
            return _insufficient("no address to geocode")

        url = f"{self._settings.census_geocoder_base_url}/locations/onelineaddress"
        params = {"address": one_line_address, "benchmark": "Public_AR_Current", "format": "json"}
        try:
            response = self._get(url, params)
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            return _error(
                f"Census geocoder request failed for '{one_line_address}': {exc}", error_type=type(exc).__name__
            )

        matches = (payload.get("result") or {}).get("addressMatches") or []
        if not matches:
            return _insufficient(f"Census geocoder found no verified match for '{one_line_address}'")

        best = matches[0]
        coords = best.get("coordinates") or {}
        lat, lon = coords.get("y"), coords.get("x")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            return _insufficient(f"Census geocoder match for '{one_line_address}' carried no usable coordinates")

        source = "census_geocoder_live"
        return GeocodeResult(
            latitude=float(lat), longitude=float(lon), matched_address=best.get("matchedAddress"),
            status="AVAILABLE", source=source, quality_state=classify_state(source).value,
        )
