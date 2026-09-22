"""FastAPI application entrypoint.

Run locally with:
    uvicorn orbitaliq.main:app --reload

Or via Docker (see Dockerfile / docker-compose.yml).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from orbitaliq.agents.orchestrator import AssessmentOrchestrator
from orbitaliq.api.routes_assessment import router as assessment_router
from orbitaliq.api.routes_health import router as health_router
from orbitaliq.api.routes_intelligence import router as intelligence_router
from orbitaliq.config import get_settings
from orbitaliq.core.logging_config import configure_logging, logger
from orbitaliq.data.database import init_db
from orbitaliq.data.geocoding_client import CensusGeocodingClient
from orbitaliq.data.imagery_provider import NasaGibsImageryProvider
from orbitaliq.data.sec_edgar_client import SecEdgarClient

# The dashboard is a static, build-free HTML/CSS/JS single-page app (see
# ../dashboard/) — a presentation layer that only calls this same API, never
# duplicates backend logic. Resolved from this file's location (not cwd) so
# `uvicorn orbitaliq.main:app` works the same regardless of the working
# directory it's launched from.
DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def create_app(orchestrator: AssessmentOrchestrator | None = None) -> FastAPI:
    """Application factory. Accepts an injected orchestrator for tests so
    the test suite never needs to load real torch weights or hit a real
    NVIDIA NIM endpoint unless it explicitly chooses to.
    """
    configure_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(f"Starting {settings.app_name} ({settings.orbitaliq_env})")
        init_db()
        app.state.orchestrator = orchestrator or AssessmentOrchestrator()
        # Process-wide, not per-request: the enterprise dashboard's direct
        # /financials, /historical, and /satellite/change-pair lookups
        # share these with each other (and — via the same per-CIK
        # company-facts cache — with every assessment the orchestrator
        # runs), rather than opening a fresh pooled HTTP client per request.
        app.state.sec_client = SecEdgarClient(settings)
        app.state.imagery_provider = NasaGibsImageryProvider(settings)
        # Process-wide, like sec_client/imagery_provider above — used only
        # by POST /api/v1/companies/resolve to geocode a company's real
        # SEC-registered business address (see data/company_resolver.py).
        app.state.geocoding_client = CensusGeocodingClient(settings)
        yield
        app.state.orchestrator.close()
        app.state.sec_client.close()
        app.state.imagery_provider.close()
        app.state.geocoding_client.close()
        logger.info(f"{settings.app_name} shut down cleanly")

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Competitive expansion intelligence platform: benchmarks a company's facility-level "
            "expansion against real evidence and reports a Competitive Expansion Signal, never a "
            "fabricated number. Fuses SEC EDGAR financial-growth signals with bi-temporal satellite "
            "change detection (satellite imagery as alternative data, disclosed as directional only) "
            "through an NVIDIA NIM-hosted LLM agent pipeline to score and explain facility-level "
            "growth signals."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(assessment_router)
    app.include_router(intelligence_router)

    # Mounted last (and at the root) so it never shadows an /api/* or
    # /health route — FastAPI/Starlette matches routers before falling
    # through to a mount. No Docker/Node build step: this is plain static
    # files served directly by the same FastAPI/uvicorn process, so the
    # whole platform is still "clone, pip install, uvicorn" on a CPU-only
    # laptop.
    if DASHBOARD_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
    else:
        logger.warning(f"Dashboard directory not found at {DASHBOARD_DIR} — serving API only, no static UI mounted.")

    return app


app = create_app()
