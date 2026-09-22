"""Tests for the enterprise dashboard's presentation-layer endpoints
(routes_intelligence.py): /financials, /historical, /satellite/change-pair,
per-assessment /evidence, /convergence, /data-quality, /compare, and
/watchlist.
"""
from __future__ import annotations

import io

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orbitaliq.agents.orchestrator import AssessmentOrchestrator
from orbitaliq.api.dependencies import get_db_session
from orbitaliq.config import get_settings
from orbitaliq.data.models import Base
from orbitaliq.main import create_app
from orbitaliq.nvidia.nim_client import NvidiaNimClient
from orbitaliq.nvidia.vision_model import VisionInferenceEngine

_TICKER_DIRECTORY = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_api_intelligence.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _override_get_db_session():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    orchestrator = AssessmentOrchestrator(
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    )
    app = create_app(orchestrator=orchestrator)
    app.dependency_overrides[get_db_session] = _override_get_db_session

    with TestClient(app) as test_client:
        yield test_client

    engine.dispose()


@pytest.fixture
def force_live_mode(monkeypatch):
    """Temporarily flips ORBITALIQ_LIVE_DATA_MODE on for one test, bypassing
    the process-wide @lru_cache on get_settings() so the route's own
    `get_settings()` call (not a FastAPI dependency, so it can't be
    overridden via app.dependency_overrides) actually observes it. Restores
    the cache after the test so every other test keeps seeing the
    network-free default from conftest.py.
    """
    monkeypatch.setenv("ORBITALIQ_LIVE_DATA_MODE", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(40, 70, 130)).save(buf, format="JPEG")
    return buf.getvalue()


def _create_assessment(client, ticker="TSLA", **overrides):
    payload = {
        "ticker": ticker, "company_name": f"{ticker} Corp", "facility_name": "Test Facility",
        "latitude": 39.5, "longitude": -119.4, "industry": "general",
    }
    payload.update(overrides)
    resp = client.post("/api/v1/assessments", json=payload)
    assert resp.status_code == 201
    return resp.json()


# --- /financials/{ticker} ---


def test_financials_endpoint_honest_when_live_mode_disabled(client):
    """The test env runs with ORBITALIQ_LIVE_DATA_MODE=false (see
    conftest.py) -- this endpoint must never attempt a live SEC EDGAR
    request there, and must return every metric as explicitly insufficient
    rather than any fabricated value.
    """
    resp = client.get("/api/v1/financials/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["cik"] is None
    assert len(body["metrics"]) == 19
    assert all(m["status"] == "INSUFFICIENT_DATA" for m in body["metrics"].values())
    assert all(m["current_value"] is None for m in body["metrics"].values())


# --- /companies/resolve ---


def test_resolve_company_honest_when_live_mode_disabled(client):
    """The test env runs with ORBITALIQ_LIVE_DATA_MODE=false (see
    conftest.py) -- this endpoint must never attempt a live SEC EDGAR/Census
    Geocoder request there, and must return an honest COMPANY_NOT_FOUND with
    a plain-language reason rather than any fabricated match.
    """
    resp = client.post("/api/v1/companies/resolve", json={"query": "Tesla"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "COMPANY_NOT_FOUND"
    assert body["query"] == "Tesla"
    assert "disabled" in body["reason"].lower()
    assert body["ticker"] is None
    assert body["candidates"] == []


def test_resolve_company_rejects_blank_query(client):
    resp = client.post("/api/v1/companies/resolve", json={"query": "   "})
    assert resp.status_code == 422


@respx.mock
def test_resolve_company_live_mode_returns_verified_ticker_and_facility(client, force_live_mode):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "addresses": {
                    "business": {
                        "street1": "One Apple Park Way", "street2": None, "city": "Cupertino",
                        "stateOrCountry": "CA", "zipCode": "95014",
                    }
                }
            },
        )
    )
    respx.get("https://geocoding.geo.census.gov/geocoder/locations/onelineaddress").mock(
        return_value=httpx.Response(
            200,
            json={
                "result": {
                    "addressMatches": [
                        {"coordinates": {"x": -122.03, "y": 37.33}, "matchedAddress": "ONE APPLE PARK WAY"}
                    ]
                }
            },
        )
    )
    resp = client.post("/api/v1/companies/resolve", json={"query": "Apple"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "RESOLVED"
    assert body["ticker"] == "AAPL"
    assert body["cik"] == "0000320193"
    assert body["latitude"] == pytest.approx(37.33)
    assert body["longitude"] == pytest.approx(-122.03)
    assert body["facility_name"] == "Corporate Headquarters (SEC-registered address)"


@respx.mock
def test_resolve_company_live_mode_unknown_company_is_not_found(client, force_live_mode):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    resp = client.post("/api/v1/companies/resolve", json={"query": "Totally Fictional Xyzzy Corp"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "COMPANY_NOT_FOUND"
    assert body["ticker"] is None


# --- /historical/{ticker} ---


def test_historical_endpoint_honest_when_live_mode_disabled(client):
    resp = client.get("/api/v1/historical/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["fiscal_years"] == []


def test_historical_endpoint_respects_max_years_param_bounds(client):
    resp = client.get("/api/v1/historical/AAPL", params={"max_years": 999})
    assert resp.status_code == 422


# --- Live-mode paths (bypasses conftest's network-free default for these tests only) ---


@respx.mock
def test_financials_endpoint_live_mode_returns_real_metrics(client, force_live_mode):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "units": {
                                "USD": [
                                    {"form": "10-K", "fp": "FY", "fy": 2022, "val": 100_000_000},
                                    {"form": "10-K", "fp": "FY", "fy": 2023, "val": 130_000_000},
                                ]
                            }
                        }
                    }
                }
            },
        )
    )
    resp = client.get("/api/v1/financials/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cik"] == "0000320193"
    assert body["metrics"]["revenue_growth"]["status"] == "AVAILABLE"
    assert body["metrics"]["revenue_growth"]["current_value"] == pytest.approx(130_000_000)


@respx.mock
def test_historical_endpoint_live_mode_returns_real_series(client, force_live_mode):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "units": {
                                "USD": [
                                    {"form": "10-K", "fp": "FY", "fy": 2022, "val": 100_000_000},
                                    {"form": "10-K", "fp": "FY", "fy": 2023, "val": 130_000_000},
                                ]
                            }
                        }
                    }
                }
            },
        )
    )
    resp = client.get("/api/v1/historical/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert {fy["fiscal_year"] for fy in body["fiscal_years"]} == {2022, 2023}


@respx.mock
def test_satellite_change_pair_live_mode_returns_visual_package(client, force_live_mode):
    respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(
        return_value=httpx.Response(200, content=_fake_jpeg_bytes(), headers={"content-type": "image/jpeg"})
    )
    resp = client.get("/api/v1/satellite/change-pair", params={"latitude": 39.5, "longitude": -119.4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["live_data_mode"] is True
    assert body["visual"] is not None
    assert body["visual"]["current_image_available"] is True
    assert body["visual"]["current_image_png"].startswith("data:image/png;base64,")


@respx.mock
def test_satellite_change_pair_strict_mode_suppresses_non_live_fallback_image(client, force_live_mode, monkeypatch):
    monkeypatch.setenv("ORBITALIQ_STRICT_REAL_DATA_MODE", "true")
    get_settings.cache_clear()
    try:
        respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(
            return_value=httpx.Response(404, text="not found")
        )
        resp = client.get("/api/v1/satellite/change-pair", params={"latitude": 39.5, "longitude": -119.4})
        assert resp.status_code == 200
        body = resp.json()
        assert body["visual"] is None
        assert "strict real-data mode" in body["unavailable_reason"].lower()
    finally:
        get_settings.cache_clear()


# --- /satellite/change-pair ---


def test_satellite_change_pair_honest_when_live_mode_disabled(client):
    resp = client.get("/api/v1/satellite/change-pair", params={"latitude": 39.5, "longitude": -119.4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["live_data_mode"] is False
    assert body["visual"] is None
    assert "disabled" in body["unavailable_reason"].lower()


def test_satellite_change_pair_validates_coordinates(client):
    resp = client.get("/api/v1/satellite/change-pair", params={"latitude": 999, "longitude": -119.4})
    assert resp.status_code == 422


# --- Per-assessment analysis endpoints ---


def test_evidence_endpoint_returns_persisted_trail(client):
    assessment = _create_assessment(client)
    resp = client.get(f"/api/v1/assessments/{assessment['id']}/evidence")
    assert resp.status_code == 200
    body = resp.json()
    assert body["assessment_id"] == assessment["id"]
    assert body["evidence"] == assessment["evidence"]
    assert len(body["evidence"]) > 0


def test_convergence_endpoint_returns_persisted_convergence(client):
    assessment = _create_assessment(client)
    resp = client.get(f"/api/v1/assessments/{assessment['id']}/convergence")
    assert resp.status_code == 200
    assert resp.json()["convergence"] == assessment["convergence"]


def test_data_quality_endpoint_returns_persisted_summary(client):
    assessment = _create_assessment(client)
    resp = client.get(f"/api/v1/assessments/{assessment['id']}/data-quality")
    assert resp.status_code == 200
    assert resp.json()["data_quality"] == assessment["data_quality"]


def test_per_assessment_endpoints_404_for_unknown_id(client):
    for suffix in ("evidence", "convergence", "data-quality"):
        resp = client.get(f"/api/v1/assessments/does-not-exist/{suffix}")
        assert resp.status_code == 404


# --- /compare ---


def test_compare_sorts_by_momentum_score_descending(client):
    a = _create_assessment(client, ticker="AAA")
    b = _create_assessment(client, ticker="BBB")
    resp = client.post("/api/v1/compare", json={"assessment_ids": [a["id"], b["id"]]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sorted_by"] == "momentum_score_desc"
    scores = [r["momentum_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["missing_ids"] == []


def test_compare_reports_missing_ids_without_failing(client):
    a = _create_assessment(client, ticker="AAA")
    resp = client.post("/api/v1/compare", json={"assessment_ids": [a["id"], "does-not-exist"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["missing_ids"] == ["does-not-exist"]
    assert len(body["results"]) == 1


def test_compare_requires_at_least_two_ids(client):
    resp = client.post("/api/v1/compare", json={"assessment_ids": ["only-one"]})
    assert resp.status_code == 422


# --- /watchlist ---


def test_watchlist_lists_flagged_entries_after_assessments(client):
    # Repeated offline-demo assessments have a deterministic-per-ticker
    # synthetic score, so at least some will land above STABLE.
    tickers = [f"WL{i}" for i in range(6)]
    for t in tickers:
        _create_assessment(client, ticker=t)

    resp = client.get("/api/v1/watchlist")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == len(body["results"])
    assert all(r["tier"] != "NONE" for r in body["results"])


def test_watchlist_tier_filter_validates_input(client):
    resp = client.get("/api/v1/watchlist", params={"tier": "NOT_A_REAL_TIER"})
    assert resp.status_code == 422


def test_watchlist_none_filter_returns_every_assessed_entry_including_deescalated(client):
    tickers = [f"WLN{i}" for i in range(6)]
    for t in tickers:
        _create_assessment(client, ticker=t)

    all_view = client.get("/api/v1/watchlist", params={"tier": "NONE"}).json()
    default_view = client.get("/api/v1/watchlist").json()
    assert all_view["total"] >= default_view["total"]