"""Resolve a free-text company name into a verified ticker/CIK and, when
one can be genuinely established, a facility location — the "type a
company name, get an evidence-backed assessment" front door described in
the platform's company-resolution requirement.

Every step here is either a real lookup against an authoritative source or
an honest, typed non-result — never a guess:

- **Ticker / CIK**: ranked search over SEC EDGAR's own public ticker
  directory (``SecEdgarClient.get_ticker_directory``) — the exact same
  data ``resolve_cik`` already uses for the existing single-ticker flow,
  just matched by company name/title too (exact, prefix, substring, then
  typo-tolerant fuzzy matching) rather than only an exact ticker symbol.
  A private, non-SEC-registered, or simply unknown company will not appear
  in this directory at all — that's an honest ``COMPANY_NOT_FOUND``, never
  a fabricated stand-in.
- **Facility location**: this pipeline has no directory of a company's
  individual facilities (plants, factories, stores) — SEC only discloses
  one thing: the company's own registered principal business address (see
  ``SecEdgarClient.fetch_business_address``). That real, SEC-disclosed
  address is geocoded into verified coordinates via the US Census Bureau's
  free, keyless geocoder (``CensusGeocodingClient``) — never estimated,
  interpolated, or defaulted to a guessed location. When either step
  doesn't produce a real result (no disclosed address, a non-US address,
  or the geocoder finding no match), the resolution still succeeds for
  financial intelligence — it is only the facility/satellite portion that
  is honestly reported as unavailable (``RESOLVED_NO_FACILITY``).

This module never fabricates a value to fill any of these gaps; every
field on ``CompanyResolution`` is either real, verified data or ``None``
with a plain-language reason attached.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum

from orbitaliq.config import Settings, get_settings
from orbitaliq.core.logging_config import logger
from orbitaliq.data.geocoding_client import CensusGeocodingClient
from orbitaliq.data.sec_edgar_client import SecEdgarClient

# The only facility this pipeline can ever auto-resolve is a company's own
# SEC-registered business address -- never a specific named facility (e.g.
# "Gigafactory Nevada"), since no directory of a company's individual
# facilities is wired into this pipeline. Analysts can still override this
# with a specific facility name + confirmed coordinates via the existing
# advanced/manual fields on POST /api/v1/assessments.
FACILITY_LABEL_SEC_HQ = "Corporate Headquarters (SEC-registered address)"
NO_FACILITY_LABEL = "No verified facility — financial intelligence only"

_MAX_CANDIDATES = 8
_FUZZY_CUTOFF = 0.72


class ResolutionStatus(str, Enum):
    # A ticker/CIK was found AND a verified facility location was
    # established (SEC-disclosed address, successfully geocoded).
    RESOLVED = "RESOLVED"
    # A ticker/CIK was found, but no verified facility coordinates could be
    # established -- financial intelligence can still run; satellite
    # intelligence must be reported INSUFFICIENT_DATA, never guessed.
    RESOLVED_NO_FACILITY = "RESOLVED_NO_FACILITY"
    # More than one SEC-registered company plausibly matches the query --
    # never silently picks one; the caller must choose.
    AMBIGUOUS = "AMBIGUOUS"
    # No SEC-registered company matches the query at all (unsupported,
    # private, non-SEC-reporting, or simply misspelled/unknown).
    COMPANY_NOT_FOUND = "COMPANY_NOT_FOUND"


@dataclass
class CompanyCandidate:
    """One real, SEC-directory-listed candidate for an ambiguous query."""

    ticker: str
    cik: str
    company_name: str

    def as_dict(self) -> dict:
        return {"ticker": self.ticker, "cik": self.cik, "company_name": self.company_name}


@dataclass
class CompanyResolution:
    status: str  # ResolutionStatus value
    query: str
    ticker: str | None = None
    cik: str | None = None
    company_name: str | None = None
    facility_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    facility_source: str | None = None
    facility_note: str | None = None
    candidates: list[CompanyCandidate] = field(default_factory=list)
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "query": self.query,
            "ticker": self.ticker,
            "cik": self.cik,
            "company_name": self.company_name,
            "facility_name": self.facility_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "facility_source": self.facility_source,
            "facility_note": self.facility_note,
            "candidates": [c.as_dict() for c in self.candidates],
            "reason": self.reason,
        }


def search_companies(sec_client: SecEdgarClient, query: str, limit: int = _MAX_CANDIDATES) -> list[CompanyCandidate]:
    """Rank-search SEC EDGAR's own ticker/title directory for a free-text
    company name. Every result is a real ``(ticker, cik, title)`` row from
    that directory — never a fabricated or partial entry. Ranking, most
    confident first: exact ticker symbol -> exact title -> title starts
    with query -> title contains query -> typo-tolerant fuzzy title match
    (only tried when nothing above matched at all, so it never overrides a
    confident exact/substring match). Returns ``[]`` for a blank query or
    when the SEC directory itself couldn't be fetched.
    """
    if not query or not query.strip():
        return []

    directory = sec_client.get_ticker_directory()
    if not directory:
        return []

    q_ticker = query.strip().upper()
    q_lower = query.strip().lower()

    seen_ciks: set[str] = set()
    ranked: list[tuple[int, str, CompanyCandidate]] = []
    for row in directory.values():
        title = str(row.get("title") or "")
        ticker = str(row.get("ticker") or "").upper()
        cik_raw = row.get("cik_str")
        if not title or not ticker or cik_raw is None:
            continue
        try:
            cik = f"{int(cik_raw):010d}"
        except (TypeError, ValueError):
            continue
        if cik in seen_ciks:
            continue

        title_lower = title.lower()
        rank = None
        if ticker == q_ticker:
            rank = 0
        elif title_lower == q_lower:
            rank = 1
        elif title_lower.startswith(q_lower):
            rank = 2
        elif q_lower in title_lower:
            rank = 3
        if rank is None:
            continue
        seen_ciks.add(cik)
        ranked.append((rank, title, CompanyCandidate(ticker=ticker, cik=cik, company_name=title)))

    if not ranked:
        # Typo-tolerant fallback pass, only when no exact/prefix/substring
        # match exists at all -- never used to override a confident match.
        titles_to_rows = {
            str(row.get("title")): row for row in directory.values() if row.get("title") and row.get("ticker")
        }
        for title in difflib.get_close_matches(query.strip(), list(titles_to_rows.keys()), n=limit, cutoff=_FUZZY_CUTOFF):
            row = titles_to_rows[title]
            try:
                cik = f"{int(row['cik_str']):010d}"
            except (TypeError, ValueError):
                continue
            if cik in seen_ciks:
                continue
            seen_ciks.add(cik)
            ranked.append((4, title, CompanyCandidate(ticker=str(row["ticker"]).upper(), cik=cik, company_name=title)))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [candidate for _, _, candidate in ranked[:limit]]


def _resolve_facility(sec_client: SecEdgarClient, geocoder: CensusGeocodingClient, cik: str) -> tuple[
    str | None, float | None, float | None, str | None, str | None
]:
    """Best-effort, real-data-only facility resolution for one CIK.
    Returns ``(facility_name, latitude, longitude, facility_source,
    facility_note)`` — coordinates are ``None`` (never guessed) unless a
    genuine SEC-disclosed address was successfully geocoded.
    """
    address = sec_client.fetch_business_address(cik)
    if address is None or not address.one_line():
        return (
            NO_FACILITY_LABEL, None, None, None,
            "SEC EDGAR does not disclose a usable business address for this filer — no facility coordinates "
            "can be verified.",
        )

    geocode = geocoder.geocode_address(address.one_line())
    if geocode.status != "AVAILABLE":
        return (
            NO_FACILITY_LABEL, None, None, None,
            f"SEC-registered address ('{address.one_line()}') could not be verified to a coordinate by the US "
            f"Census Bureau geocoder — {geocode.note or 'no match'}.",
        )

    return (
        FACILITY_LABEL_SEC_HQ,
        geocode.latitude,
        geocode.longitude,
        f"sec_edgar_live+{geocode.source}",
        f"Geocoded from SEC EDGAR's registered business address: {address.one_line()}",
    )


def resolve_company(
    query: str,
    *,
    sec_client: SecEdgarClient,
    geocoder: CensusGeocodingClient | None = None,
    settings: Settings | None = None,
) -> CompanyResolution:
    """Resolve a free-text company name into a verified ticker/CIK and, if
    possible, a verified facility location. Never fabricates any field —
    see the module docstring for exactly what "verified" means for each
    part of the result.
    """
    query = (query or "").strip()
    if not query:
        return CompanyResolution(
            status=ResolutionStatus.COMPANY_NOT_FOUND.value, query=query,
            reason="No company name was entered.",
        )

    candidates = search_companies(sec_client, query)

    if not candidates:
        if sec_client.ticker_directory_error:
            return CompanyResolution(
                status=ResolutionStatus.COMPANY_NOT_FOUND.value, query=query,
                reason=f"SEC EDGAR company directory could not be fetched: {sec_client.ticker_directory_error}",
            )
        return CompanyResolution(
            status=ResolutionStatus.COMPANY_NOT_FOUND.value, query=query,
            reason=f"No SEC-registered public company matches '{query}'. It may be private, non-SEC-reporting, "
            "or misspelled — this pipeline only resolves companies that file with the SEC.",
        )

    if len(candidates) > 1:
        # A single unmistakable exact match (ticker or title) is not
        # treated as ambiguous even if a looser fuzzy match also exists for
        # a *different* company — only surface ambiguity when more than one
        # candidate is a genuinely strong match for the query.
        strong = [c for c in candidates if c.ticker == query.strip().upper() or c.company_name.lower() == query.strip().lower()]
        if len(strong) == 1:
            candidates = strong
        else:
            return CompanyResolution(
                status=ResolutionStatus.AMBIGUOUS.value, query=query, candidates=candidates,
                reason=f"'{query}' matches {len(candidates)} SEC-registered companies — select the correct one.",
            )

    chosen = candidates[0]
    logger.info(f"company_resolver: resolved '{query}' -> {chosen.ticker} (CIK{chosen.cik}) '{chosen.company_name}'")

    owns_geocoder = geocoder is None
    geocoder = geocoder or CensusGeocodingClient(settings or get_settings())
    try:
        facility_name, lat, lon, facility_source, facility_note = _resolve_facility(sec_client, geocoder, chosen.cik)
    finally:
        if owns_geocoder:
            geocoder.close()

    status = ResolutionStatus.RESOLVED.value if lat is not None and lon is not None else ResolutionStatus.RESOLVED_NO_FACILITY.value
    return CompanyResolution(
        status=status, query=query, ticker=chosen.ticker, cik=chosen.cik, company_name=chosen.company_name,
        facility_name=facility_name, latitude=lat, longitude=lon,
        facility_source=facility_source, facility_note=facility_note,
    )
