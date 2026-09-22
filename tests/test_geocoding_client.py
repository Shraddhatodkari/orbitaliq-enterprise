"""Tests for CensusGeocodingClient — the free, keyless US Census Bureau
geocoder used to turn a company's real SEC-registered business address into
verified latitude/longitude for the "resolve by company name" workflow
(see data/company_resolver.py). Every outcome must be either a genuine,
verified coordinate or a typed INSUFFICIENT_DATA/ERROR result — this client
never guesses, interpolates, or invents a coordinate.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from orbitaliq.config import Settings
from orbitaliq.data.geocoding_client import CensusGeocodingClient

_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"


@pytest.fixture
def settings():
    return Settings(SEC_EDGAR_USER_AGENT="Test Suite test@example.com", LIVE_DATA_TIMEOUT_SECONDS=2.0)


def _match_payload(lat=39.5380, lon=-119.4425, matched="1 TESLA RD, AUSTIN, TX, 78725"):
    return {
        "result": {
            "addressMatches": [
                {"coordinates": {"x": lon, "y": lat}, "matchedAddress": matched},
            ]
        }
    }


@respx.mock
def test_geocode_address_success_returns_verified_coordinates(settings):
    respx.get(_GEOCODER_URL).mock(return_value=httpx.Response(200, json=_match_payload()))
    client = CensusGeocodingClient(settings)
    result = client.geocode_address("1 Tesla Road, Austin, TX 78725")

    assert result.status == "AVAILABLE"
    assert result.latitude == pytest.approx(39.5380)
    assert result.longitude == pytest.approx(-119.4425)
    assert result.matched_address == "1 TESLA RD, AUSTIN, TX, 78725"
    assert result.source == "census_geocoder_live"
    assert result.quality_state == "LIVE"
    client.close()


@respx.mock
def test_geocode_address_takes_first_match_only(settings):
    payload = _match_payload()
    payload["result"]["addressMatches"].append(
        {"coordinates": {"x": -1.0, "y": 1.0}, "matchedAddress": "SOME OTHER MATCH"}
    )
    respx.get(_GEOCODER_URL).mock(return_value=httpx.Response(200, json=payload))
    client = CensusGeocodingClient(settings)
    result = client.geocode_address("1 Tesla Road, Austin, TX 78725")
    assert result.latitude == pytest.approx(39.5380)
    assert result.matched_address == "1 TESLA RD, AUSTIN, TX, 78725"
    client.close()


@respx.mock
def test_geocode_address_no_match_returns_insufficient_data_never_a_guess(settings):
    respx.get(_GEOCODER_URL).mock(return_value=httpx.Response(200, json={"result": {"addressMatches": []}}))
    client = CensusGeocodingClient(settings)
    result = client.geocode_address("123 Nonexistent Way, Nowhere, ZZ 00000")

    assert result.status == "INSUFFICIENT_DATA"
    assert result.latitude is None
    assert result.longitude is None
    assert result.source == "insufficient_data"
    assert result.quality_state == "INSUFFICIENT_DATA"
    assert "no verified match" in result.note.lower()
    client.close()


def test_geocode_address_blank_address_returns_insufficient_data_without_any_request(settings):
    client = CensusGeocodingClient(settings)
    result = client.geocode_address("")
    assert result.status == "INSUFFICIENT_DATA"
    assert result.source == "insufficient_data"
    client.close()

    client2 = CensusGeocodingClient(settings)
    result2 = client2.geocode_address("   ")
    assert result2.status == "INSUFFICIENT_DATA"
    client2.close()


@respx.mock
def test_geocode_address_network_failure_is_typed_error_not_insufficient_data(settings):
    respx.get(_GEOCODER_URL).mock(side_effect=httpx.ConnectError("boom"))
    client = CensusGeocodingClient(settings)
    result = client.geocode_address("1 Tesla Road, Austin, TX 78725")

    assert result.status == "INSUFFICIENT_DATA"  # coarse flag stays the same
    assert result.latitude is None
    assert result.source.startswith("error:census_geocoder:")
    assert result.quality_state == "ERROR"
    client.close()


@respx.mock
def test_geocode_address_missing_coordinates_in_match_is_insufficient_data(settings):
    payload = {
        "result": {
            "addressMatches": [{"coordinates": {}, "matchedAddress": "INCOMPLETE MATCH"}],
        }
    }
    respx.get(_GEOCODER_URL).mock(return_value=httpx.Response(200, json=payload))
    client = CensusGeocodingClient(settings)
    result = client.geocode_address("1 Tesla Road, Austin, TX 78725")
    assert result.status == "INSUFFICIENT_DATA"
    assert result.latitude is None
    assert result.longitude is None
    client.close()


@respx.mock
def test_geocode_address_sends_expected_query_params(settings):
    route = respx.get(_GEOCODER_URL).mock(return_value=httpx.Response(200, json=_match_payload()))
    client = CensusGeocodingClient(settings)
    client.geocode_address("1 Tesla Road, Austin, TX 78725")

    assert route.called
    request = route.calls.last.request
    params = dict(httpx.QueryParams(request.url.query))
    assert params["address"] == "1 Tesla Road, Austin, TX 78725"
    assert params["benchmark"] == "Public_AR_Current"
    assert params["format"] == "json"
    client.close()


def test_as_dict_round_trips_every_field(settings):
    result = CensusGeocodingClient(settings).geocode_address("")
    d = result.as_dict()
    assert set(d.keys()) == {
        "latitude", "longitude", "matched_address", "status", "source", "note", "quality_state",
    }
