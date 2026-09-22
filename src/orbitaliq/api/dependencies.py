"""FastAPI dependency providers.

The vision model and NVIDIA NIM HTTP client are expensive/stateful to
construct (loading CNN weights onto a device, opening a pooled HTTP
connection) so both are built once per process and reused across requests
via ``app.state``, rather than per-request — the standard FastAPI pattern
for singleton resources.
"""
from __future__ import annotations

from collections.abc import Generator

from fastapi import Request
from sqlalchemy.orm import Session

from orbitaliq.agents.orchestrator import AssessmentOrchestrator
from orbitaliq.data.database import SessionLocal
from orbitaliq.data.geocoding_client import CensusGeocodingClient
from orbitaliq.data.imagery_provider import NasaGibsImageryProvider
from orbitaliq.data.repository import AssessmentRepository
from orbitaliq.data.sec_edgar_client import SecEdgarClient


def get_orchestrator(request: Request) -> AssessmentOrchestrator:
    return request.app.state.orchestrator


def get_sec_client(request: Request) -> SecEdgarClient:
    """The process-wide SEC EDGAR client (see app.state in main.py's
    lifespan) — shared, not per-request, so the enterprise dashboard's
    direct /financials and /historical lookups benefit from the same
    per-CIK company-facts cache the assessment pipeline uses.
    """
    return request.app.state.sec_client


def get_imagery_provider(request: Request) -> NasaGibsImageryProvider:
    return request.app.state.imagery_provider


def get_geocoding_client(request: Request) -> CensusGeocodingClient:
    """The process-wide Census Bureau geocoding client (see app.state in
    main.py's lifespan) — used only by the company-resolution endpoint
    (``POST /api/v1/companies/resolve``) to turn a company's real
    SEC-registered business address into verified coordinates.
    """
    return request.app.state.geocoding_client


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    except Exception:
        # Without this, a failure partway through a route (after a
        # repository call has already flushed a pending INSERT/UPDATE to
        # the DB-API connection but before an explicit commit) would reach
        # ``session.close()`` with an open transaction still attached.
        # SQLAlchemy's pool does reset a checked-in connection, but rolling
        # back explicitly here — before the session is torn down — is the
        # documented-correct pattern and removes any window where a stuck
        # transaction could make the *next* request's SQLite access see a
        # lock that has no reason to still be held.
        session.rollback()
        raise
    finally:
        session.close()


def get_repository(session: Session) -> AssessmentRepository:
    return AssessmentRepository(session)
