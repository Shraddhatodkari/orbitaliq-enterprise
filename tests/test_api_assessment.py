import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orbitaliq.agents.orchestrator import AssessmentOrchestrator
from orbitaliq.api.dependencies import get_db_session
from orbitaliq.data.models import Base
from orbitaliq.main import create_app
from orbitaliq.nvidia.nim_client import NvidiaNimClient
from orbitaliq.nvidia.vision_model import VisionInferenceEngine


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_api.db"
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


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["offline_mode"] is True


def test_create_assessment_success(client):
    resp = client.post(
        "/api/v1/assessments",
        json={
            "ticker": "tsla",
            "company_name": "Tesla, Inc.",
            "facility_name": "Gigafactory Nevada",
            "latitude": 39.5380,
            "longitude": -119.4425,
            "industry": "automotive",
            "market_cap_usd": 800_000_000_000,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["ticker"] == "TSLA"  # normalized to uppercase
    assert body["company_name"] == "Tesla, Inc."
    assert 0.0 <= body["momentum_score"] <= 100.0
    assert body["momentum_level"] in {"STABLE", "EMERGING", "STRONG", "AGGRESSIVE_EXPANSION"}
    assert body["narrative_report"]
    assert len(body["agent_trace"]) == 5
    assert "id" in body and body["id"]

    # Audit-grade identity: every evidence item's assessment_id must be this
    # exact, real, persisted record's own id -- never a placeholder and
    # never mismatched (see api/routes_assessment.py's pre-generated id).
    assert body["evidence"]
    assert all(item["assessment_id"] == body["id"] for item in body["evidence"])
    evidence_ids = [item["evidence_id"] for item in body["evidence"]]
    assert len(evidence_ids) == len(set(evidence_ids))
    assert all(item["correlation_id"] for item in body["evidence"])  # a real per-request id, not blank
    assert all(item["agent"] for item in body["evidence"])


def test_create_assessment_validates_latitude(client):
    resp = client.post(
        "/api/v1/assessments",
        json={"ticker": "BAD", "company_name": "Bad Co", "facility_name": "Bad Site", "latitude": 999, "longitude": 0},
    )
    assert resp.status_code == 422


def test_create_assessment_rejects_blank_ticker(client):
    resp = client.post(
        "/api/v1/assessments",
        json={"ticker": "   ", "company_name": "X", "facility_name": "Y", "latitude": 10, "longitude": 10},
    )
    assert resp.status_code == 422


def test_get_assessment_round_trip(client):
    create_resp = client.post(
        "/api/v1/assessments",
        json={
            "ticker": "AMZN",
            "company_name": "Amazon.com, Inc.",
            "facility_name": "Fulfillment Center Round Trip",
            "latitude": 1.0,
            "longitude": 1.0,
        },
    )
    assessment_id = create_resp.json()["id"]

    get_resp = client.get(f"/api/v1/assessments/{assessment_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["facility_name"] == "Fulfillment Center Round Trip"


def test_get_assessment_not_found(client):
    resp = client.get("/api/v1/assessments/does-not-exist")
    assert resp.status_code == 404


def test_list_assessments_and_min_momentum_score_filter(client):
    client.post(
        "/api/v1/assessments",
        json={"ticker": "AAA", "company_name": "Company A", "facility_name": "Site A", "latitude": 2.0, "longitude": 2.0},
    )
    client.post(
        "/api/v1/assessments",
        json={"ticker": "BBB", "company_name": "Company B", "facility_name": "Site B", "latitude": 3.0, "longitude": 3.0},
    )

    resp = client.get("/api/v1/assessments")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_matching"] >= 2
    assert len(body["results"]) >= 2

    resp_filtered = client.get("/api/v1/assessments", params={"min_momentum_score": 99.9})
    assert resp_filtered.status_code == 200
    assert resp_filtered.json()["results"] == []

    resp_out_of_range = client.get("/api/v1/assessments", params={"min_momentum_score": 200})
    assert resp_out_of_range.status_code == 422


def test_list_assessments_limit_param(client):
    for i in range(3):
        client.post(
            "/api/v1/assessments",
            json={
                "ticker": f"BLK{i}",
                "company_name": f"Bulk Co {i}",
                "facility_name": f"Bulk Site {i}",
                "latitude": 5.0,
                "longitude": 5.0,
            },
        )

    resp = client.get("/api/v1/assessments", params={"limit": 2})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


# --- Structured error handling ------------------------------------------
#
# Regression coverage for the historical "500: Internal Server Error" bug
# report: previously, ONLY orchestrator.run() (the agent pipeline itself)
# was wrapped in a try/except; any failure in the code that ran AFTER a
# successful pipeline run — building the DB record, committing it, or
# building the response model — was completely unhandled and reached the
# caller as a bare, undiagnosable 500 with no body. These three tests each
# force a failure in one of the three now-independently-guarded stages and
# assert it comes back as a structured, readable error instead.


class _ExplodingOrchestrator:
    """A fake whose .run() always raises, to exercise the
    pipeline_execution error path deterministically (no network/CNN
    flakiness involved).
    """

    def run(self, company, *, assessment_id=None, correlation_id=None):
        raise RuntimeError("simulated ingestion agent network failure")

    def close(self) -> None:
        pass


def test_create_assessment_pipeline_failure_returns_structured_502(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'pipeline_fail.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _override_get_db_session():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app = create_app(orchestrator=_ExplodingOrchestrator())
    app.dependency_overrides[get_db_session] = _override_get_db_session

    with TestClient(app) as test_client:
        resp = test_client.post(
            "/api/v1/assessments",
            json={"ticker": "BOOM", "company_name": "Boom Co", "facility_name": "Site", "latitude": 1.0, "longitude": 1.0},
        )
    engine.dispose()

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert detail["stage"] == "pipeline_execution"
    assert "simulated ingestion agent network failure" in detail["message"]
    assert detail["ticker"] == "BOOM"
    assert detail["request_id"]


def test_create_assessment_persistence_failure_returns_structured_500_and_rolls_back(client, monkeypatch):
    from orbitaliq.data.repository import AssessmentRepository

    def _boom_add(self, record):
        raise RuntimeError("simulated sqlite disk I/O error")

    monkeypatch.setattr(AssessmentRepository, "add", _boom_add)

    resp = client.post(
        "/api/v1/assessments",
        json={"ticker": "PFAIL", "company_name": "Persist Fail Co", "facility_name": "Site", "latitude": 1.0, "longitude": 1.0},
    )
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["stage"] == "persistence"
    assert "simulated sqlite disk I/O error" in detail["message"]
    assert detail["source"] == "sqlite_database"
    assert detail["ticker"] == "PFAIL"
    assert detail["request_id"]

    # The failed attempt must not have left a partial row behind.
    list_resp = client.get("/api/v1/assessments")
    assert all(r["ticker"] != "PFAIL" for r in list_resp.json()["results"])


def test_create_assessment_response_serialization_failure_returns_structured_500(client, monkeypatch):
    import orbitaliq.api.routes_assessment as routes_assessment_module

    real_assessment_response = routes_assessment_module.AssessmentResponse

    def _boom_response(**kwargs):
        raise ValueError("simulated response schema mismatch")

    monkeypatch.setattr(routes_assessment_module, "AssessmentResponse", _boom_response)

    resp = client.post(
        "/api/v1/assessments",
        json={"ticker": "RFAIL", "company_name": "Response Fail Co", "facility_name": "Site", "latitude": 1.0, "longitude": 1.0},
    )
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["stage"] == "response_serialization"
    assert "simulated response schema mismatch" in detail["message"]
    assert detail["source"] == "api_response_schema"

    # The pipeline result WAS already committed by this point (only its
    # response representation failed to build) — restore the real schema
    # and confirm a subsequent list call shows it, proving no silent data
    # loss on this specific failure.
    monkeypatch.setattr(routes_assessment_module, "AssessmentResponse", real_assessment_response)
    list_resp = client.get("/api/v1/assessments")
    assert list_resp.status_code == 200
    assert any(r["ticker"] == "RFAIL" for r in list_resp.json()["results"])


def test_get_assessment_response_serialization_failure_returns_structured_500(client, monkeypatch):
    import orbitaliq.api.routes_assessment as routes_assessment_module

    create_resp = client.post(
        "/api/v1/assessments",
        json={"ticker": "GFAIL", "company_name": "Get Fail Co", "facility_name": "Site", "latitude": 1.0, "longitude": 1.0},
    )
    assessment_id = create_resp.json()["id"]

    def _boom_response(**kwargs):
        raise ValueError("simulated get-response schema mismatch")

    monkeypatch.setattr(routes_assessment_module, "AssessmentResponse", _boom_response)

    resp = client.get(f"/api/v1/assessments/{assessment_id}")
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["stage"] == "response_serialization"
    assert "simulated get-response schema mismatch" in detail["message"]
    assert detail["ticker"] == "GFAIL"


def test_list_assessments_response_serialization_failure_returns_structured_500(client, monkeypatch):
    import orbitaliq.api.routes_assessment as routes_assessment_module

    client.post(
        "/api/v1/assessments",
        json={"ticker": "LFAIL", "company_name": "List Fail Co", "facility_name": "Site", "latitude": 1.0, "longitude": 1.0},
    )

    def _boom_response(**kwargs):
        raise ValueError("simulated list-response schema mismatch")

    monkeypatch.setattr(routes_assessment_module, "AssessmentResponse", _boom_response)

    resp = client.get("/api/v1/assessments")
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["stage"] == "response_serialization"
    assert "simulated list-response schema mismatch" in detail["message"]


# --- Regression coverage for the CONFIRMED root cause of the reported
# "500 Internal Server Error" on Executive Dashboard / Company Intelligence
# / Assessment List: a real deployed project database was inspected
# directly (PRAGMA table_info on the live orbitaliq.db) and found to be
# missing 11 columns that AssessmentRecord has grown since that database
# file was first created — Base.metadata.create_all() never alters an
# existing table, so every ORM SELECT against it raised
# sqlite3.OperationalError: no such column: assessments.<name>, which
# reached the browser as a bare 500. These tests build a database on that
# exact legacy shape, run the app's real startup path against it, and
# assert the previously-500ing endpoints now return 200 — not just that a
# unit helper works in isolation (see tests/test_database.py for that). ---

_LEGACY_ASSESSMENTS_TABLE_DDL = """
CREATE TABLE assessments (
    id VARCHAR(36) NOT NULL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    facility_name VARCHAR(255) NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    industry VARCHAR(120) NOT NULL,
    market_cap_usd FLOAT,
    momentum_score FLOAT NOT NULL,
    momentum_level VARCHAR(30) NOT NULL,
    signal_breakdown JSON NOT NULL,
    vision_findings JSON NOT NULL,
    recommended_actions JSON NOT NULL,
    narrative_report VARCHAR NOT NULL,
    agent_trace JSON NOT NULL,
    llm_generated BOOLEAN NOT NULL,
    data_sources JSON NOT NULL,
    created_at DATETIME NOT NULL
)
"""

_LEGACY_TSLA_ROW = {
    "id": "4c94550f-ce25-4693-bc28-497e9de84abe",
    "ticker": "TSLA",
    "company_name": "Tesla, Inc.",
    "facility_name": "Gigafactory Nevada",
    "latitude": 39.538,
    "longitude": -119.4425,
    "industry": "automotive",
    "market_cap_usd": 800000000000.0,
    "momentum_score": 35.91735720018142,
    "momentum_level": "EMERGING",
    "signal_breakdown": '{"revenue_growth": 17.67}',
    "vision_findings": '{"overall_change_score": 0.84}',
    "recommended_actions": "[]",
    "narrative_report": "Tesla, Inc. (TSLA) carries a composite momentum score of 35.9/100.",
    "agent_trace": "[]",
    "llm_generated": 0,
    "data_sources": '{"revenue_growth_signal": "sec_edgar_live"}',
    "created_at": "2026-09-18 15:58:42.514393",
}


@pytest.fixture
def legacy_schema_client(tmp_path):
    """Exactly like the ``client`` fixture above, except the database file
    starts on the pre-enterprise-dashboard schema (as a real deployed
    project's database was found to be) instead of a freshly-created one —
    so the app's own real startup migration is what has to make it work,
    not test setup.
    """
    from orbitaliq.data.database import init_db

    db_path = tmp_path / "legacy_test_api.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.exec_driver_sql(_LEGACY_ASSESSMENTS_TABLE_DDL)
        columns = ", ".join(_LEGACY_TSLA_ROW.keys())
        placeholders = ", ".join(f":{k}" for k in _LEGACY_TSLA_ROW.keys())
        conn.exec_driver_sql(f"INSERT INTO assessments ({columns}) VALUES ({placeholders})", _LEGACY_TSLA_ROW)
    Base.metadata.create_all(engine)  # creates watchlist_entries, which this legacy fixture predates
    init_db(bind_engine=engine)  # the actual repair path — exactly what create_app()'s lifespan runs

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


def test_list_assessments_against_legacy_schema_database_no_longer_500s(legacy_schema_client):
    """This is the Executive Dashboard / Assessment List regression: it
    calls exactly what those Streamlit/dashboard views call.
    """
    resp = legacy_schema_client.get("/api/v1/assessments")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_matching"] == 1
    result = body["results"][0]
    assert result["ticker"] == "TSLA"
    assert result["momentum_score"] == _LEGACY_TSLA_ROW["momentum_score"]
    # The new enterprise-dashboard fields must come back as honest,
    # never-fabricated defaults for a row that predates them.
    assert result["financial_profile"] is None
    assert result["evidence"] == []
    assert result["watchlist_tier"] == "NONE"


def test_get_assessment_against_legacy_schema_database_no_longer_500s(legacy_schema_client):
    """This is the Company Intelligence regression: it calls exactly what
    that view calls.
    """
    resp = legacy_schema_client.get(f"/api/v1/assessments/{_LEGACY_TSLA_ROW['id']}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ticker"] == "TSLA"
    assert body["facility_name"] == "Gigafactory Nevada"
    assert body["data_sources"] == {"revenue_growth_signal": "sec_edgar_live"}


def test_health_endpoint_against_legacy_schema_database(legacy_schema_client):
    resp = legacy_schema_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_new_assessment_still_works_against_a_migrated_legacy_database(legacy_schema_client):
    """A migrated database must accept brand-new assessments too, not just
    read the one legacy row.
    """
    resp = legacy_schema_client.post(
        "/api/v1/assessments",
        json={
            "ticker": "POSTMIG",
            "company_name": "Post Migration Co",
            "facility_name": "New Site",
            "latitude": 2.0,
            "longitude": 2.0,
        },
    )
    assert resp.status_code == 201, resp.text
    list_resp = legacy_schema_client.get("/api/v1/assessments")
    assert list_resp.json()["total_matching"] == 2
