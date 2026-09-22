from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from orbitaliq.agents.orchestrator import AssessmentOrchestrator
from orbitaliq.agents.types import CompanyInput
from orbitaliq.api.dependencies import get_db_session, get_orchestrator
from orbitaliq.api.schemas import AssessmentListResponse, AssessmentResponse, CompanyAssessmentRequest
from orbitaliq.core.logging_config import logger
from orbitaliq.data.models import AssessmentRecord
from orbitaliq.data.repository import AssessmentRepository, WatchlistRepository

router = APIRouter(prefix="/api/v1/assessments", tags=["assessments"])


def _error_detail(*, stage: str, message: str, ticker: str, request_id: str, source: str | None = None) -> dict:
    """A structured, honest error body — never a bare, undiagnosable 500.

    ``stage`` names exactly which part of the request failed (so a caller,
    human or the Streamlit/dashboard UI, can tell "the agent pipeline
    itself failed" apart from "the pipeline succeeded but saving/returning
    the result failed"), ``message`` is the real exception's type and text
    (never swallowed or replaced with a generic phrase), and
    ``request_id`` lets the failure be correlated with the full server-side
    traceback this same request logs via ``logger.exception`` — see
    README > Known limitations for why this exists.
    """
    return {"stage": stage, "message": message, "source": source, "ticker": ticker, "request_id": request_id}


@router.post("", response_model=AssessmentResponse, status_code=201)
def create_assessment(
    payload: CompanyAssessmentRequest,
    orchestrator: AssessmentOrchestrator = Depends(get_orchestrator),
    session: Session = Depends(get_db_session),
) -> AssessmentResponse:
    """Run the full ingestion -> vision -> intelligence-scoring -> report ->
    watchlist agent pipeline for a company facility and persist the result.

    Three independent failure zones are each caught and diagnosed
    separately (pipeline execution, database persistence, response
    serialization) rather than relying on a single catch-all — a failure
    in persistence or serialization used to propagate completely
    unhandled, reaching the caller as a bare "500: Internal Server Error"
    with no indication of what actually broke (see README > Known
    limitations > "Root cause of the historical 500 error").
    """
    request_id = str(uuid.uuid4())
    # Pre-generated here (rather than left to the AssessmentRecord's own
    # default) so the exact same id can be stamped onto every evidence item
    # in this same pipeline run -- an evidence item's assessment_id is only
    # ever this real, later-persisted record's own id, never a value made
    # up after the fact.
    assessment_id = str(uuid.uuid4())
    company = CompanyInput(
        ticker=payload.ticker,
        company_name=payload.company_name,
        facility_name=payload.facility_name,
        latitude=payload.latitude,
        longitude=payload.longitude,
        industry=payload.industry,
        market_cap_usd=payload.market_cap_usd,
    )

    try:
        result = orchestrator.run(company, assessment_id=assessment_id, correlation_id=request_id)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: this is the diagnostic boundary
        logger.exception(f"[{request_id}] Assessment pipeline failed for {payload.ticker}")
        raise HTTPException(
            status_code=502,
            detail=_error_detail(
                stage="pipeline_execution",
                message=f"{type(exc).__name__}: {exc}",
                ticker=payload.ticker,
                source="ingestion/vision/scoring/report/watchlist agents",
                request_id=request_id,
            ),
        ) from exc

    try:
        record = AssessmentRecord(
            id=assessment_id,
            ticker=company.ticker,
            company_name=company.company_name,
            facility_name=company.facility_name,
            latitude=company.latitude,
            longitude=company.longitude,
            industry=company.industry,
            market_cap_usd=company.market_cap_usd,
            momentum_score=result.momentum_score,
            momentum_level=result.momentum_level,
            signal_breakdown=result.signal_breakdown,
            vision_findings=result.vision_findings,
            recommended_actions=result.recommended_actions,
            narrative_report=result.narrative_report,
            agent_trace=[t.as_dict() for t in result.agent_trace],
            llm_generated=result.llm_generated,
            data_sources=result.data_sources,
            watchlist_flagged=result.watchlist_flagged,
            watchlist_channel=result.watchlist_channel,
            watchlist_tier=result.watchlist_tier,
            watchlist_webhook_status=result.watchlist_webhook_status,
            financial_profile=result.financial_profile,
            convergence=result.convergence,
            data_quality=result.data_quality,
            evidence=result.evidence,
            narrative_status=result.narrative_status,
            narrative_grounding=result.narrative_grounding,
            satellite_visual=result.satellite_visual,
            score_breakdown=result.score_breakdown,
        )
        repo = AssessmentRepository(session)
        repo.add(record)
        # Upsert the derived watchlist index for this (ticker, facility)
        # pair — see data/repository.py::WatchlistRepository for why this
        # is a queryable index over the assessment, not a second source of
        # truth.
        WatchlistRepository(session).upsert_from_assessment(record)
        session.commit()
        session.refresh(record)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: this is the diagnostic boundary
        session.rollback()
        logger.exception(f"[{request_id}] Failed to persist assessment result for {payload.ticker}")
        raise HTTPException(
            status_code=500,
            detail=_error_detail(
                stage="persistence",
                message=f"{type(exc).__name__}: {exc}",
                ticker=payload.ticker,
                source="sqlite_database",
                request_id=request_id,
            ),
        ) from exc

    try:
        return AssessmentResponse(**record.to_dict())
    except Exception as exc:  # noqa: BLE001 - deliberately broad: this is the diagnostic boundary
        logger.exception(f"[{request_id}] Failed to serialize assessment response for {payload.ticker}")
        raise HTTPException(
            status_code=500,
            detail=_error_detail(
                stage="response_serialization",
                message=f"{type(exc).__name__}: {exc}",
                ticker=payload.ticker,
                source="api_response_schema",
                request_id=request_id,
            ),
        ) from exc


@router.get("/{assessment_id}", response_model=AssessmentResponse)
def get_assessment(assessment_id: str, session: Session = Depends(get_db_session)) -> AssessmentResponse:
    repo = AssessmentRepository(session)
    record = repo.get(assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="assessment not found")
    try:
        return AssessmentResponse(**record.to_dict())
    except Exception as exc:  # noqa: BLE001 - same diagnostic boundary as create_assessment
        request_id = str(uuid.uuid4())
        logger.exception(f"[{request_id}] Failed to serialize stored assessment {assessment_id}")
        raise HTTPException(
            status_code=500,
            detail=_error_detail(
                stage="response_serialization",
                message=f"{type(exc).__name__}: {exc}",
                ticker=record.ticker,
                source="api_response_schema",
                request_id=request_id,
            ),
        ) from exc


@router.get("", response_model=AssessmentListResponse)
def list_assessments(
    min_momentum_score: float | None = Query(default=None, ge=0, le=100),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db_session),
) -> AssessmentListResponse:
    repo = AssessmentRepository(session)
    records = repo.list(min_momentum_score=min_momentum_score, limit=limit, offset=offset)
    try:
        return AssessmentListResponse(
            total_matching=repo.count(),
            results=[AssessmentResponse(**r.to_dict()) for r in records],
        )
    except Exception as exc:  # noqa: BLE001 - same diagnostic boundary as create_assessment
        request_id = str(uuid.uuid4())
        logger.exception(f"[{request_id}] Failed to serialize assessment list")
        raise HTTPException(
            status_code=500,
            detail=_error_detail(
                stage="response_serialization",
                message=f"{type(exc).__name__}: {exc}",
                ticker="(list)",
                source="api_response_schema",
                request_id=request_id,
            ),
        ) from exc
