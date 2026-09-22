"""Tests for data/company_resolver.py — the "type a company name, get a
verified ticker/CIK and (when possible) a verified facility location" front
door. Every outcome must be either a real result grounded in SEC EDGAR's
own directory + a genuinely geocoded SEC-registered business address, or an
honest RESOLVED_NO_FACILITY / AMBIGUOUS / COMPANY_NOT_FOUND — never a
fabricated ticker, company, or coordinate.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from orbitaliq.config import Settings
from orbitaliq.data.company_resolver import (
    FACILITY_LABEL_SEC_HQ,
    NO_FACILITY_LABEL,
    ResolutionStatus,
    resolve_company,
    search_companies,
)
from orbitaliq.data.geocoding_client import CensusGeocodingClient
from orbitaliq.data.sec_edgar_client import SecEdgarClient

_TICKER_DIRECTORY = {
    "0": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "3": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
}
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"


def _submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"


@pytest.fixture
def settings():
    return Settings(SEC_EDGAR_USER_AGENT="Test Suite test@example.com", LIVE_DATA_TIMEOUT_SECONDS=2.0)


def _mock_directory():
    respx.get(_TICKERS_URL).mock(return_value=httpx.Response(200, json=_TICKER_DIRECTORY))


def _mock_submissions(cik: str, *, street1="1 Tesla Road", city="Austin", state="TX", zip_code="78725"):
    respx.get(_submissions_url(cik)).mock(
        return_value=httpx.Response(
            200,
            json={
                "addresses": {
                    "business": {
                        "street1": street1, "street2": None, "city": city,
                        "stateOrCountry": state, "zipCode": zip_code,
                    }
                }
            },
        )
    )


def _mock_geocode(lat=39.5380, lon=-119.4425):
    respx.get(_GEOCODER_URL).mock(
        return_value=httpx.Response(
            200,
            json={"result": {"addressMatches": [{"coordinates": {"x": lon, "y": lat}, "matchedAddress": "MATCHED"}]}},
        )
    )


# --- search_companies: ranking ---


@respx.mock
def test_search_companies_exact_ticker_match_ranks_first(settings):
    _mock_directory()
    client = SecEdgarClient(settings=settings)
    results = search_companies(client, "TSLA")
    assert results[0].ticker == "TSLA"
    assert results[0].cik == "0001318605"
    client.close()


@respx.mock
def test_search_companies_exact_title_match(settings):
    _mock_directory()
    client = SecEdgarClient(settings=settings)
    results = search_companies(client, "Tesla, Inc.")
    assert results[0].ticker == "TSLA"
    client.close()


@respx.mock
def test_search_companies_prefix_and_substring_match(settings):
    _mock_directory()
    client = SecEdgarClient(settings=settings)
    results = search_companies(client, "Microsoft")
    assert any(c.ticker == "MSFT" for c in results)
    client.close()


@respx.mock
def test_search_companies_fuzzy_typo_tolerant_fallback(settings):
    _mock_directory()
    client = SecEdgarClient(settings=settings)
    # Misspelled title, no exact/prefix/substring hit at all.
    results = search_companies(client, "Aple Inc")
    assert any(c.ticker == "AAPL" for c in results)
    client.close()


@respx.mock
def test_search_companies_unknown_query_returns_empty(settings):
    _mock_directory()
    client = SecEdgarClient(settings=settings)
    assert search_companies(client, "Totally Fictional Company Name Xyzzy") == []
    client.close()


def test_search_companies_blank_query_returns_empty_without_any_request(settings):
    client = SecEdgarClient(settings=settings)
    assert search_companies(client, "") == []
    assert search_companies(client, "   ") == []
    client.close()


# --- resolve_company: RESOLVED (ticker + verified facility) ---


@respx.mock
def test_resolve_company_fully_resolved_with_verified_facility(settings):
    _mock_directory()
    _mock_submissions("0001318605")
    _mock_geocode()
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)

    resolution = resolve_company("Tesla", sec_client=sec_client, geocoder=geocoder, settings=settings)

    assert resolution.status == ResolutionStatus.RESOLVED.value
    assert resolution.ticker == "TSLA"
    assert resolution.cik == "0001318605"
    assert resolution.company_name == "Tesla, Inc."
    assert resolution.facility_name == FACILITY_LABEL_SEC_HQ
    assert resolution.latitude == pytest.approx(39.5380)
    assert resolution.longitude == pytest.approx(-119.4425)
    assert resolution.facility_source is not None
    assert "sec_edgar_live" in resolution.facility_source
    assert resolution.candidates == []
    sec_client.close()
    geocoder.close()


@respx.mock
def test_resolve_company_by_exact_ticker_symbol(settings):
    _mock_directory()
    _mock_submissions("0001318605")
    _mock_geocode()
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)

    resolution = resolve_company("TSLA", sec_client=sec_client, geocoder=geocoder, settings=settings)
    assert resolution.status == ResolutionStatus.RESOLVED.value
    assert resolution.ticker == "TSLA"
    sec_client.close()
    geocoder.close()


# --- resolve_company: RESOLVED_NO_FACILITY (no disclosed address / no geocode match) ---


@respx.mock
def test_resolve_company_no_business_address_disclosed_is_resolved_no_facility(settings):
    _mock_directory()
    respx.get(_submissions_url("0001318605")).mock(return_value=httpx.Response(200, json={"addresses": {}}))
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)

    resolution = resolve_company("Tesla", sec_client=sec_client, geocoder=geocoder, settings=settings)

    assert resolution.status == ResolutionStatus.RESOLVED_NO_FACILITY.value
    assert resolution.ticker == "TSLA"  # financial intelligence can still run
    assert resolution.facility_name == NO_FACILITY_LABEL
    assert resolution.latitude is None
    assert resolution.longitude is None
    assert resolution.facility_note is not None
    sec_client.close()
    geocoder.close()


@respx.mock
def test_resolve_company_address_disclosed_but_geocoder_finds_no_match(settings):
    _mock_directory()
    _mock_submissions("0001318605")
    respx.get(_GEOCODER_URL).mock(return_value=httpx.Response(200, json={"result": {"addressMatches": []}}))
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)

    resolution = resolve_company("Tesla", sec_client=sec_client, geocoder=geocoder, settings=settings)

    assert resolution.status == ResolutionStatus.RESOLVED_NO_FACILITY.value
    assert resolution.latitude is None
    assert resolution.longitude is None
    # Never a guessed coordinate substituted for the failed geocode.
    assert resolution.facility_name == NO_FACILITY_LABEL
    sec_client.close()
    geocoder.close()


# --- resolve_company: AMBIGUOUS ---


@respx.mock
def test_resolve_company_ambiguous_query_returns_candidates_never_picks_one(settings):
    _mock_directory()
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)

    # "Corp" substring-matches both Microsoft and Nvidia's directory titles.
    resolution = resolve_company("Corp", sec_client=sec_client, geocoder=geocoder, settings=settings)

    assert resolution.status == ResolutionStatus.AMBIGUOUS.value
    assert len(resolution.candidates) >= 2
    assert resolution.ticker is None
    assert resolution.latitude is None
    tickers = {c.ticker for c in resolution.candidates}
    assert {"MSFT", "NVDA"}.issubset(tickers)
    sec_client.close()
    geocoder.close()


# --- resolve_company: COMPANY_NOT_FOUND ---


@respx.mock
def test_resolve_company_unknown_company_is_not_found_never_guessed(settings):
    _mock_directory()
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)

    resolution = resolve_company(
        "Totally Fictional Private Company Xyzzy", sec_client=sec_client, geocoder=geocoder, settings=settings
    )

    assert resolution.status == ResolutionStatus.COMPANY_NOT_FOUND.value
    assert resolution.ticker is None
    assert resolution.candidates == []
    assert resolution.reason is not None
    sec_client.close()
    geocoder.close()


def test_resolve_company_blank_query_is_not_found_without_any_request(settings):
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)
    resolution = resolve_company("   ", sec_client=sec_client, geocoder=geocoder, settings=settings)
    assert resolution.status == ResolutionStatus.COMPANY_NOT_FOUND.value
    sec_client.close()
    geocoder.close()


@respx.mock
def test_resolve_company_directory_fetch_failure_surfaces_as_not_found_with_reason(settings):
    respx.get(_TICKERS_URL).mock(side_effect=httpx.ConnectError("boom"))
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)

    resolution = resolve_company("Tesla", sec_client=sec_client, geocoder=geocoder, settings=settings)
    assert resolution.status == ResolutionStatus.COMPANY_NOT_FOUND.value
    assert "SEC EDGAR company directory" in resolution.reason
    sec_client.close()
    geocoder.close()


# --- resolve_company: owns and closes its own geocoder when none is passed ---


@respx.mock
def test_resolve_company_creates_and_closes_its_own_geocoder_when_none_passed(settings):
    _mock_directory()
    _mock_submissions("0001318605")
    _mock_geocode()
    sec_client = SecEdgarClient(settings=settings)

    resolution = resolve_company("Tesla", sec_client=sec_client, settings=settings)  # no geocoder kwarg

    assert resolution.status == ResolutionStatus.RESOLVED.value
    assert resolution.latitude == pytest.approx(39.5380)
    sec_client.close()


# --- CompanyResolution.as_dict() shape ---


@respx.mock
def test_company_resolution_as_dict_has_every_field(settings):
    _mock_directory()
    _mock_submissions("0001318605")
    _mock_geocode()
    sec_client = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)
    resolution = resolve_company("Tesla", sec_client=sec_client, geocoder=geocoder, settings=settings)
    d = resolution.as_dict()
    assert set(d.keys()) == {
        "status", "query", "ticker", "cik", "company_name", "facility_name",
        "latitude", "longitude", "facility_source", "facility_note", "candidates", "reason",
    }
    sec_client.close()
    geocoder.close()
