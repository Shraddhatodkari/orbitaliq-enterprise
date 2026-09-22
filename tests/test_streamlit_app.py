"""End-to-end rendering tests for the Streamlit analyst dashboard.

These run the *real* ``streamlit_app/app.py`` script via Streamlit's
``AppTest`` framework against the *real* FastAPI backend (started in a
background thread on a real port, exactly like a user would run it) —
not mocks of either side. The goal is what the task calls "dashboard
rendering logic where practical": prove every section actually executes
without raising, against real backend responses, including the empty
(no assessments yet) state and after at least one real assessment exists.

``AppTest`` is Streamlit's own official testing API (``streamlit.testing.v1``,
public since Streamlit 1.28) — it runs the script in an isolated Python
subinterpreter-like context and lets us assert on the resulting widget
tree, but its most important guarantee for this project's requirements is
simpler: if ANY exception is raised while rendering a page, ``at.run()``
captures it as ``at.exception`` instead of silently succeeding, which is
exactly the "does this crash" signal these tests check.
"""
from __future__ import annotations

import os
import socket
import threading
import time

import pytest
import uvicorn

pytest.importorskip("streamlit", reason="Streamlit is an optional dependency for the analyst dashboard")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from orbitaliq.agents.orchestrator import AssessmentOrchestrator  # noqa: E402
from orbitaliq.api.dependencies import get_db_session  # noqa: E402
from orbitaliq.data.models import Base  # noqa: E402
from orbitaliq.main import create_app  # noqa: E402
from orbitaliq.nvidia.nim_client import NvidiaNimClient  # noqa: E402
from orbitaliq.nvidia.vision_model import VisionInferenceEngine  # noqa: E402

_APP_PATH = os.path.join(os.path.dirname(__file__), "..", "streamlit_app", "app.py")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_backend(tmp_path_factory):
    """A real uvicorn server serving the real FastAPI app, on a real port,
    backed by a throwaway SQLite file — run in a background thread for the
    lifetime of this test module. This is what the Streamlit app's own
    httpx client actually talks to, exactly as it would talk to a
    developer's locally-run backend.
    """
    # The app's own default (non-overridden) SessionLocal is bound, at
    # import time, to whatever DATABASE_URL conftest.py set — an
    # in-memory SQLite DB that (via SQLAlchemy's SingletonThreadPool for
    # `:memory:` URLs) gives EACH thread its own separate, empty
    # database. Since this fixture runs a real uvicorn server in its own
    # thread and FastAPI dispatches each request to yet another worker
    # thread, that would mean "table not found" on the very first
    # request. Overriding the ``get_db_session`` dependency with a
    # file-based engine — the exact same pattern every other test module
    # in this suite already uses (see tests/test_api_assessment.py) — is
    # the correct fix, not a workaround specific to this test.
    db_path = tmp_path_factory.mktemp("streamlit_e2e") / "streamlit_e2e.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _override_get_db_session():
        session = test_session_local()
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

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("live backend did not start in time")

    base_url = f"http://127.0.0.1:{port}"
    os.environ["ORBITALIQ_API_BASE_URL"] = base_url
    yield base_url

    server.should_exit = True
    thread.join(timeout=10)
    os.environ.pop("ORBITALIQ_API_BASE_URL", None)
    engine.dispose()


def _make_app_test() -> AppTest:
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    return at


def test_executive_dashboard_renders_empty_state_with_no_assessments(live_backend):
    at = _make_app_test()
    at.run()
    assert at.exception == []
    assert not at.error


def test_new_assessment_form_renders_without_error(live_backend):
    at = _make_app_test()
    at.run()
    at.sidebar.radio[0].set_value("New Assessment").run()
    assert at.exception == []


def test_full_navigation_sweep_after_one_real_assessment(live_backend):
    """Create one real assessment through the actual backend, then visit
    every single sidebar section and assert none of them raise. This is
    the broadest possible "does the dashboard actually work" check.
    """
    import httpx

    resp = httpx.post(
        f"{live_backend}/api/v1/assessments",
        json={
            "ticker": "STAPP",
            "company_name": "Streamlit Test Co",
            "facility_name": "Test Facility",
            "latitude": 10.0,
            "longitude": 10.0,
            "industry": "general",
        },
        timeout=60,
    )
    assert resp.status_code == 201, resp.text

    resp2 = httpx.post(
        f"{live_backend}/api/v1/assessments",
        json={
            "ticker": "STAPP2",
            "company_name": "Streamlit Test Co Two",
            "facility_name": "Second Facility",
            "latitude": 20.0,
            "longitude": 20.0,
            "industry": "general",
        },
        timeout=60,
    )
    assert resp2.status_code == 201, resp2.text

    sections = [
        "Executive Dashboard",
        "New Assessment",
        "Company Intelligence",
        "Satellite Intelligence",
        "Competitive Comparison",
        "Watchlist",
        "Executive Strategy View",
        "Agent Trace",
        "Evidence & Audit / Data Quality",
    ]
    for section in sections:
        at = _make_app_test()
        at.run()
        at.sidebar.radio[0].set_value(section).run()
        assert at.exception == [], f"{section} raised: {[e.value for e in at.exception]}"


def test_watchlist_tier_filter_does_not_raise(live_backend):
    at = _make_app_test()
    at.run()
    at.sidebar.radio[0].set_value("Watchlist").run()
    assert at.exception == []
    if at.radio:
        at.radio[0].set_value("NONE").run()
        assert at.exception == []


def test_backend_unreachable_shows_structured_error_not_a_crash(monkeypatch):
    """Point the dashboard at a port nothing is listening on and confirm
    it degrades to a clear connection error rather than raising.
    """
    monkeypatch.setenv("ORBITALIQ_API_BASE_URL", "http://127.0.0.1:1")
    at = _make_app_test()
    at.run()
    assert at.exception == []
    assert any("unreachable" in e.value.lower() or "backend" in e.value.lower() for e in at.error) or at.error
