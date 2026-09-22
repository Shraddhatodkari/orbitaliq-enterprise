"""Real satellite imagery via NASA GIBS (Global Imagery Browse Services).

GIBS is NASA's free, keyless, public tile service for genuine satellite
imagery — VIIRS (Suomi NPP / NOAA-20) and MODIS (Terra/Aqua) true-color
Earth observation, refreshed daily, served as standard Web-Mercator
("slippy map") tiles. Docs: https://nasa-gibs.github.io/gibs-api-docs/

This is the same family of satellite platform (VIIRS) used in NVIDIA's
"Disaster Risk Monitoring Using Satellite Imagery" course content, and
requires no account, no API key, and no signup — a GET request against a
documented public URL returns a real JPEG satellite image tile for the
requested location and date.

**Honesty note on resolution.** GIBS' native VIIRS/MODIS true-color
product is roughly 250m-1km per pixel depending on layer. This client
additionally resizes every fetched tile down to a fixed 64x64px (the
CNN's input shape), which coarsens the *effective* resolution further —
see ``approximate_resolution_m_per_pixel()``, which every satellite API
response reports explicitly (typically ~600m/pixel to ~1.2km/pixel
depending on latitude) rather than asserting a single fixed number. That
is real, freely-available imagery, but at that resolution a tile can only
show large-scale land-cover change — a new factory building, a cleared
lot, a large parking/laydown-yard expansion — the same scale of signal
real alternative-data analysts tracked at Tesla's Gigafactory Nevada site
via satellite through 2018-2020. It cannot resolve individual vehicles,
small structures, or building-level detail; firms like Orbital Insight or
SpaceKnow that do car-counting-level analysis use commercial sub-meter
imagery (Planet, Maxar), which is not free. This module is honest about
that ceiling throughout (see ``nvidia/vision_model.py``), and reports the
actual computed resolution rather than a marketing figure.

This module fetches a **bi-temporal pair** — the most recent available
tile and one from ``lookback_days`` earlier — so downstream vision code
does genuine before/after change detection rather than classifying a
single snapshot. Every fetch independently degrades to a typed
``status="INSUFFICIENT_DATA"`` result on failure (outage, rate limit, no
cloud-free pass for that date) — never a fabricated substitute image —
and the returned provenance always discloses whether each half of the
pair is genuinely live NASA imagery or unavailable. The deterministic
synthetic-tile generator in ``data/synthetic_data.py`` exists solely for
``ORBITALIQ_LIVE_DATA_MODE=false`` (offline demo mode) and test fixtures;
this module never calls it on a live-mode failure.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
import numpy as np
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from orbitaliq.config import Settings, get_settings
from orbitaliq.core.data_quality_state import classify_state
from orbitaliq.core.logging_config import logger
from orbitaliq.data.synthetic_data import TILE_SIZE

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)

DEFAULT_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
DEFAULT_TILE_MATRIX_SET = "GoogleMapsCompatible_Level9"
DEFAULT_ZOOM = 9  # native max resolution of the GoogleMapsCompatible_Level9 matrix set.
# After this client's resize to TILE_SIZE (64px, for the CNN's fixed input shape), the
# effective resolution is ~600m/pixel at 60N/S down to ~1.2km/pixel at the equator — see
# approximate_resolution_m_per_pixel(), which every satellite API response reports
# explicitly rather than asserting a single fixed number. A lower zoom (this module used
# zoom 6 in an earlier revision) covers a much wider area (~150km/tile) than a single
# facility needs and would report an even coarser, less honest effective resolution.
# GIBS near-real-time imagery typically publishes with ~1-2 day latency;
# requesting "yesterday" can occasionally 404 before that day's mosaic is
# ready, so default a couple of days back for reliability.
DEFAULT_LATENCY_DAYS = 3
# Default bi-temporal baseline for change detection: about six months,
# long enough for large-scale construction/land-clearing to show up in
# 250m-1km imagery without requiring a long archival history.
DEFAULT_LOOKBACK_DAYS = 180


@dataclass
class ImageryResult:
    """One (current or prior) tile fetch. ``tile`` is always a real array
    shape -- the CNN and PNG renderers need *some* array -- but when a
    live-mode fetch fails, ``tile`` is a blank placeholder (all zeros),
    never the deterministic synthetic-imagery generator (that generator
    is reserved for offline demo mode -- see ``status``/``source`` below,
    which are what every consumer must actually check before treating a
    tile, or anything derived from it, as real).
    """

    tile: np.ndarray  # float32, shape (3, TILE_SIZE, TILE_SIZE), values in [0, 1]
    live: bool
    source: str
    summary: str
    # Structured observation metadata (populated by fetch_satellite_tile;
    # defaulted so existing callers/tests that construct ImageryResult with
    # only the four original fields keep working unchanged).
    observation_date: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    layer: str | None = None
    zoom: int | None = None
    resolution_m_per_pixel: float | None = None
    # "AVAILABLE" (genuine NASA GIBS imagery) | "INSUFFICIENT_DATA" (live
    # fetch failed) | "OFFLINE_DEMO" (deliberate, disclosed offline mode).
    status: str = "AVAILABLE"
    # The canonical 5-state classification of `source` (see
    # core/data_quality_state.py) -- always derived from `source` via
    # classify_state() at construction time. Additive: `status` above is
    # unchanged and remains what every existing caller checks.
    quality_state: str = "LIVE"

    def provenance(self) -> dict:
        return {
            "live": self.live, "source": self.source, "summary": self.summary, "status": self.status,
            "quality_state": self.quality_state,
        }


@dataclass
class ChangeTilePairResult:
    """A bi-temporal pair of tiles for the same site, ready for change detection."""

    current: ImageryResult
    prior: ImageryResult
    lookback_days: int

    @property
    def live(self) -> bool:
        return self.current.live and self.prior.live

    @property
    def source(self) -> str:
        if self.live:
            return "nasa_gibs_live"
        if self.current.live or self.prior.live:
            return "nasa_gibs_partial_live"
        return "insufficient_data"

    def summary(self) -> str:
        return f"current: {self.current.summary} | prior (-{self.lookback_days}d): {self.prior.summary}"


def _lonlat_to_tile_xy(latitude: float, longitude: float, zoom: int) -> tuple[int, int]:
    """Standard Web-Mercator slippy-map tile indices for a lat/lon at a zoom level."""
    lat_rad = math.radians(max(min(latitude, 85.05112878), -85.05112878))
    n = 2.0**zoom
    x = int((longitude + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(x, int(n) - 1))
    y = max(0, min(y, int(n) - 1))
    return x, y


def approximate_resolution_m_per_pixel(latitude: float, zoom: int = DEFAULT_ZOOM, tile_px: int = TILE_SIZE) -> float:
    """Approximate ground resolution of one output-tile pixel, via the
    standard Web-Mercator tile-resolution formula (256px reference tiles).

    Stated as an approximation on purpose: GIBS' native VIIRS/MODIS
    true-color product resolution is nominally ~375m-1km/pixel depending
    on the layer, and the *effective* resolution of what this client
    returns also depends on the requested zoom level and the resize to
    ``TILE_SIZE``. This function reports the latter (the geometry actually
    used), not a marketing spec for the sensor.
    """
    earth_circumference_m = 40075016.686
    native_256px_tile_resolution = earth_circumference_m * math.cos(math.radians(latitude)) / (2**zoom) / 256
    return native_256px_tile_resolution * (256 / tile_px)


def tile_to_png_data_uri(tile: np.ndarray) -> str:
    """Render a ``[3, H, W]`` float32 ``[0, 1]`` tile array to a base64
    PNG data URI, for the dashboard's Before / After / Change view.
    """
    arr = np.clip(tile, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    image = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def diff_heatmap_png_data_uri(current: np.ndarray, prior: np.ndarray) -> str:
    """Render a per-pixel change-magnitude heatmap (red intensity = amount
    of change between the two tiles at that location) as a base64 PNG.
    """
    diff = np.abs(current.astype(np.float32) - prior.astype(np.float32)).mean(axis=0)  # (H, W)
    peak = max(float(diff.max()), 1e-6)
    diff_norm = np.clip(diff / peak, 0.0, 1.0)
    heat = np.zeros((diff_norm.shape[0], diff_norm.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (diff_norm * 255).astype(np.uint8)
    heat[..., 2] = ((1.0 - diff_norm) * 40).astype(np.uint8)
    image = Image.fromarray(heat, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_satellite_visual_package(pair: ChangeTilePairResult, *, source_name: str = "NASA GIBS (VIIRS/MODIS)") -> dict:
    """The full Satellite Change Detection View payload: both raw images,
    a diff heatmap, and every provenance/resolution field the dashboard's
    evidence drawer needs — built once here so the API layer never has to
    duplicate image-rendering or resolution-math logic.
    """
    current, prior = pair.current, pair.prior
    return {
        "current_observation_date": current.observation_date,
        "prior_observation_date": prior.observation_date,
        "lookback_days": pair.lookback_days,
        "satellite_product": current.layer or prior.layer,
        "source_name": source_name,
        "source_url": "https://gibs.earthdata.nasa.gov/wmts/",
        "coordinates": {"latitude": current.latitude, "longitude": current.longitude},
        "resolution_m_per_pixel": current.resolution_m_per_pixel,
        "current_image_available": current.live,
        "prior_image_available": prior.live,
        "current_provenance": current.provenance(),
        "prior_provenance": prior.provenance(),
        "change_score_note": (
            "Observed satellite change reflects large-scale land-cover change only "
            "(resolution ceiling ~250m-1km/pixel) — not confirmation of any specific business event."
        ),
        "current_image_png": tile_to_png_data_uri(current.tile),
        "prior_image_png": tile_to_png_data_uri(prior.tile),
        "diff_image_png": diff_heatmap_png_data_uri(current.tile, prior.tile),
    }


class NasaGibsImageryProvider:
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
    def _get(self, url: str) -> httpx.Response:
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def _build_url(self, x: int, y: int, zoom: int, tile_date: str, layer: str, tile_matrix_set: str) -> str:
        return (
            f"{self._settings.nasa_gibs_base_url}/epsg3857/best/{layer}/default/"
            f"{tile_date}/{tile_matrix_set}/{zoom}/{y}/{x}.jpg"
        )

    def fetch_satellite_tile(
        self,
        latitude: float,
        longitude: float,
        *,
        tile_date: str | None = None,
        zoom: int = DEFAULT_ZOOM,
        layer: str = DEFAULT_LAYER,
        tile_matrix_set: str = DEFAULT_TILE_MATRIX_SET,
        seed_salt: str = "tile",
    ) -> ImageryResult:
        resolved_date = tile_date or (date.today() - timedelta(days=DEFAULT_LATENCY_DAYS)).isoformat()
        x, y = _lonlat_to_tile_xy(latitude, longitude, zoom)
        url = self._build_url(x, y, zoom, resolved_date, layer, tile_matrix_set)

        try:
            response = self._get(url)
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                raise ValueError(f"GIBS returned non-image content-type: {content_type!r}")
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image = image.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, C)
            array = np.transpose(array, (2, 0, 1))  # -> (C, H, W) to match the CNN's expected layout
        except Exception as exc:  # noqa: BLE001 - any network/decode failure degrades gracefully
            logger.warning(f"NASA GIBS live imagery unavailable at ({latitude}, {longitude}) for {resolved_date}: {exc}")
            # A 404 from GIBS for a given date/layer/location is a
            # documented, expected gap (see module docstring: near-real-time
            # imagery can publish with 1-2 day latency, so a requested date
            # can genuinely have no mosaic yet) -- INSUFFICIENT_DATA. Any
            # other failure (transport error, timeout, non-404 HTTP status,
            # bad content-type, image-decode failure) is unexpected --
            # DataQualityState.ERROR (see core/data_quality_state.py).
            is_documented_gap = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404
            if is_documented_gap:
                source = "insufficient_data"
            else:
                source = f"error:nasa_gibs:{type(exc).__name__}"
            return ImageryResult(
                # A blank placeholder, never the deterministic synthetic
                # generator (``generate_satellite_tile`` is reserved for
                # offline demo mode -- see IngestionAgent.run()). Any
                # CNN/change-index output computed over this tile is
                # excluded from the momentum score by the data-sufficiency
                # gate (core/intelligence_engine.py::_is_trusted_source) and
                # must never be presented as a real observation -- that is
                # what status="INSUFFICIENT_DATA" / the typed source above
                # are for.
                tile=np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.float32),
                live=False,
                source=source,
                summary=f"NASA GIBS request failed for {resolved_date}: {exc}",
                observation_date=resolved_date,
                latitude=latitude,
                longitude=longitude,
                layer=layer,
                zoom=zoom,
                resolution_m_per_pixel=approximate_resolution_m_per_pixel(latitude, zoom),
                status="INSUFFICIENT_DATA",
                quality_state=classify_state(source).value,
            )

        live_source = "nasa_gibs_live"
        return ImageryResult(
            tile=array,
            live=True,
            source=live_source,
            summary=f"{layer} tile z{zoom}/{y}/{x} for {resolved_date}",
            observation_date=resolved_date,
            latitude=latitude,
            longitude=longitude,
            layer=layer,
            zoom=zoom,
            resolution_m_per_pixel=approximate_resolution_m_per_pixel(latitude, zoom),
            status="AVAILABLE",
            quality_state=classify_state(live_source).value,
        )

    def fetch_change_pair(
        self,
        latitude: float,
        longitude: float,
        *,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        zoom: int = DEFAULT_ZOOM,
        layer: str = DEFAULT_LAYER,
        tile_matrix_set: str = DEFAULT_TILE_MATRIX_SET,
    ) -> ChangeTilePairResult:
        """Fetch a genuine bi-temporal tile pair for the same site: the most
        recent available imagery ("current") and imagery from
        ``lookback_days`` earlier ("prior"). Each half independently and
        transparently falls back to synthetic data on failure.
        """
        current_date = (date.today() - timedelta(days=DEFAULT_LATENCY_DAYS)).isoformat()
        prior_date = (date.today() - timedelta(days=DEFAULT_LATENCY_DAYS + lookback_days)).isoformat()

        current = self.fetch_satellite_tile(
            latitude, longitude, tile_date=current_date, zoom=zoom, layer=layer,
            tile_matrix_set=tile_matrix_set, seed_salt="current",
        )
        prior = self.fetch_satellite_tile(
            latitude, longitude, tile_date=prior_date, zoom=zoom, layer=layer,
            tile_matrix_set=tile_matrix_set, seed_salt="prior",
        )
        return ChangeTilePairResult(current=current, prior=prior, lookback_days=lookback_days)
