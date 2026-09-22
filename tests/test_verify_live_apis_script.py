"""Regression tests for scripts/verify_live_apis.py's own resolve-first
flow -- not the backend package, the standalone verification script
itself. These exist specifically to pin the bug reported and fixed in
this pass: the script used to default ``--ticker``/``--lat``/``--lon`` to
Tesla's own values independently of ``--company``, so an unqualified
``--company Microsoft`` run genuinely printed ``Ticker: TSLA`` and queried
NASA GIBS with Tesla's Gigafactory coordinates. The fix makes the script
resolve the requested company FIRST and drive every downstream check only
from that resolution (or an explicit manual override) -- these tests prove
a Tesla run and a Microsoft run in the same process never leak into each
other, and that AMBIGUOUS/COMPANY_NOT_FOUND never fabricate a ticker or
coordinate either.

The script lives under scripts/, not the ``orbitaliq`` package, so it is
loaded here by file path via importlib rather than a normal import.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest
import respx

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_live_apis.py"

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
_GIBS_URL_REGEX = r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg"

_TICKER_DIRECTORY = {
    "0": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}

# Distinct, individually-identifiable fixtures per company -- if the script
# ever cross-wires these, the wrong ticker/number/coordinate shows up in
# the WRONG run's captured output and the test fails.
_TESLA_REVENUE = [
    {"form": "10-K", "fp": "FY", "fy": 2024, "val": 97_690_000_000},
    {"form": "10-K", "fp": "FY", "fy": 2025, "val": 94_827_000_000},
]
_MSFT_REVENUE = [
    {"form": "10-K", "fp": "FY", "fy": 2025, "val": 245_122_000_000},
    {"form": "10-K", "fp": "FY", "fy": 2026, "val": 281_724_000_000},
]
_TESLA_LAT, _TESLA_LON = 39.5380, -119.4425  # Gigafactory Nevada
_MSFT_LAT, _MSFT_LON = 47.6423, -122.1390  # Redmond campus
_DC_LAT, _DC_LON = 38.8977, -77.0365  # the fixed connectivity-check address


def _facts(revenue_series: list[dict]) -> dict:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": revenue_series}},
            }
        }
    }


def _mock_directory():
    respx.get(_TICKERS_URL).mock(return_value=httpx.Response(200, json=_TICKER_DIRECTORY))


def _mock_submissions(cik: str, one_line_marker: str):
    """``one_line_marker`` (e.g. "TESLA" or "MICROSOFT") is embedded in the
    street address so the geocoder side_effect below can tell which
    company's address is being geocoded, exactly as the real Census API
    receives a plain address string with no company identifier attached.
    """
    respx.get(f"https://data.sec.gov/submissions/CIK{cik}.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "addresses": {
                    "business": {
                        "street1": f"1 {one_line_marker} WAY", "street2": None,
                        "city": "Somewhere", "stateOrCountry": "CA", "zipCode": "00000",
                    }
                }
            },
        )
    )


def _mock_company_facts(cik: str, revenue_series: list[dict]):
    respx.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").mock(
        return_value=httpx.Response(200, json=_facts(revenue_series))
    )


def _geocode_side_effect(request: httpx.Request) -> httpx.Response:
    address = str(request.url.params.get("address", "")).upper()
    if "TESLA" in address:
        lat, lon, label = _TESLA_LAT, _TESLA_LON, "TESLA WAY MATCH"
    elif "MICROSOFT" in address:
        lat, lon, label = _MSFT_LAT, _MSFT_LON, "MICROSOFT WAY MATCH"
    else:
        # the fixed reference address used only by the connectivity check
        lat, lon, label = _DC_LAT, _DC_LON, "PENNSYLVANIA AVE MATCH"
    return httpx.Response(
        200,
        json={"result": {"addressMatches": [{"coordinates": {"x": lon, "y": lat}, "matchedAddress": label}]}},
    )


def _mock_geocoder():
    respx.get(_GEOCODER_URL).mock(side_effect=_geocode_side_effect)


def _mock_gibs():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(40, 70, 130)).save(buf, format="JPEG")
    respx.get(url__regex=_GIBS_URL_REGEX).mock(
        return_value=httpx.Response(200, content=buf.getvalue(), headers={"content-type": "image/jpeg"})
    )


@pytest.fixture
def script_main():
    """Load scripts/verify_live_apis.py's ``main`` fresh for each test
    (module-level respx state must not leak between tests, and the module
    itself has no mutable state, but reloading keeps this test file
    independent of import order / caching quirks)."""
    spec = importlib.util.spec_from_file_location("verify_live_apis_under_test", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_live_apis_under_test"] = module
    spec.loader.exec_module(module)
    yield module.main
    del sys.modules["verify_live_apis_under_test"]


@respx.mock
def test_microsoft_run_never_shows_tesla_ticker_or_coordinates(script_main, monkeypatch, capsys):
    _mock_directory()
    _mock_submissions("0000789019", "MICROSOFT")
    _mock_company_facts("0000789019", _MSFT_REVENUE)
    _mock_geocoder()
    _mock_gibs()
    monkeypatch.setattr(sys, "argv", ["verify_live_apis.py", "--company", "Microsoft"])
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test Suite test@example.com")

    exit_code = script_main()
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "MSFT" in out
    assert "CIK0000789019" in out
    assert f"({_MSFT_LAT}, {_MSFT_LON})" in out
    # The exact regression this test exists to pin: nothing from Tesla's
    # identity or Tesla's coordinates may appear in a Microsoft run.
    assert "TSLA" not in out
    assert "0001318605" not in out
    assert str(_TESLA_LAT) not in out
    assert str(_TESLA_LON) not in out
    assert "Gigafactory" not in out


@respx.mock
def test_tesla_run_never_shows_microsoft_ticker_or_coordinates(script_main, monkeypatch, capsys):
    _mock_directory()
    _mock_submissions("0001318605", "TESLA")
    _mock_company_facts("0001318605", _TESLA_REVENUE)
    _mock_geocoder()
    _mock_gibs()
    monkeypatch.setattr(sys, "argv", ["verify_live_apis.py", "--company", "Tesla"])
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test Suite test@example.com")

    exit_code = script_main()
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "TSLA" in out
    assert "CIK0001318605" in out
    assert f"({_TESLA_LAT}, {_TESLA_LON})" in out
    assert "MSFT" not in out
    assert "0000789019" not in out
    assert str(_MSFT_LAT) not in out
    assert str(_MSFT_LON) not in out


@respx.mock
def test_company_with_no_verified_facility_skips_gibs_without_fabricating_a_coordinate(script_main, monkeypatch, capsys):
    """Mirrors the real, honestly-observed Tesla/Microsoft outcome in this
    project's own live validation: SEC resolves, the disclosed address does
    NOT geocode, so satellite must be skipped -- never a guessed or
    previously-seen coordinate substituted in its place."""
    _mock_directory()
    _mock_submissions("0000789019", "MICROSOFT")
    _mock_company_facts("0000789019", _MSFT_REVENUE)
    respx.get(_GEOCODER_URL).mock(return_value=httpx.Response(200, json={"result": {"addressMatches": []}}))
    gibs_route = respx.get(url__regex=_GIBS_URL_REGEX)
    monkeypatch.setattr(sys, "argv", ["verify_live_apis.py", "--company", "Microsoft"])
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test Suite test@example.com")

    exit_code = script_main()
    out = capsys.readouterr().out

    assert exit_code == 2  # mixed: resolved, no facility -- not a failure
    assert "RESOLVED_NO_FACILITY" in out
    assert "SATELLITE: INSUFFICIENT_DATA" in out
    assert "NASA GIBS was never called" in out
    assert not gibs_route.called  # never guessed a coordinate to satisfy the check
    assert str(_TESLA_LAT) not in out and str(_TESLA_LON) not in out


@respx.mock
def test_ambiguous_company_skips_sec_and_gibs_never_fabricates_identity(script_main, monkeypatch, capsys):
    _mock_directory()
    respx.get(_GEOCODER_URL).mock(side_effect=_geocode_side_effect)
    sec_facts_route = respx.get(url__regex=r"https://data\.sec\.gov/api/xbrl/companyfacts/.*")
    gibs_route = respx.get(url__regex=_GIBS_URL_REGEX)
    # "s" substring-matches both "Tesla, Inc." and "MICROSOFT CORP" -- genuinely ambiguous.
    monkeypatch.setattr(sys, "argv", ["verify_live_apis.py", "--company", "s"])
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test Suite test@example.com")

    exit_code = script_main()
    out = capsys.readouterr().out

    assert exit_code == 2
    assert "AMBIGUOUS" in out
    assert "SEC EDGAR check SKIPPED" in out
    assert not sec_facts_route.called
    assert not gibs_route.called


@respx.mock
def test_nonexistent_company_reports_not_found_never_fabricates(script_main, monkeypatch, capsys):
    _mock_directory()
    respx.get(_GEOCODER_URL).mock(side_effect=_geocode_side_effect)
    monkeypatch.setattr(sys, "argv", ["verify_live_apis.py", "--company", "xyz-company-that-does-not-exist-123"])
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test Suite test@example.com")

    exit_code = script_main()
    out = capsys.readouterr().out

    assert exit_code == 2
    assert "COMPANY_NOT_FOUND" in out
    assert "TSLA" not in out and "MSFT" not in out
