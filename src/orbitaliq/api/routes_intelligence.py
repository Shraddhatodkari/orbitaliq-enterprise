"""Enterprise dashboard endpoints — real-data surfaces (financials,
historical trends, satellite change detection) and per-assessment
audit/analysis views (evidence trail, signal convergence, data quality,
multi-company comparison).

This router is a thin presentation layer over the existing agents/core
modules: it never recomputes anything the orchestrator, SEC EDGAR client,
imagery provider, or core/*.py modules already compute deterministically —
see requirement #17 ("dashboard is a separate presentation layer calling
the existing API, not duplicating backend logic").
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from orbitaliq.api.dependencies import get_db_session, get_geocoding_client, get_imagery_provider, get_sec_client
from orbitaliq.api.schemas import (
    CompanyResolveRequest,
    CompanyResolveResponse,
    CompareRequest,
    CompareResponse,
    CompareResultItem,
    ConvergenceResponse,
    DataQualityResponse,
    EvidenceResponse,
    FinancialProfileResponse,
    HistoricalFinancialsResponse,
    SatelliteVisualResponse,
    WatchlistEntryResponse,
    WatchlistListResponse,
)
from orbitaliq.config import get_settings
from orbitaliq.data.company_resolver import resolve_company
from orbitaliq.data.geocoding_client import CensusGeocodingClient
from orbitaliq.data.imagery_provider import NasaGibsImageryProvider, build_satellite_visual_package
from orbitaliq.data.repository import AssessmentRepository, WatchlistRepository
from orbitaliq.data.sec_edgar_client import SecEdgarClient, _all_metrics_insufficient

router = APIRouter(prefix="/api/v1", tags=["intelligence"])

_LIVE_MODE_DISABLED_NOTE = (
    "ORBITALIQ_LIVE_DATA_MODE is disabled for this deployment — no live SEC EDGAR "
    "request was made, so every metric is honestly reported as insufficient rather "
    "than filled in from a synthetic source."
)


@router.get("/financials/{ticker}", response_model=FinancialProfileResponse)
def get_financial_profile(
    ticker: str,
    sec_client: SecEdgarClient = Depends(get_sec_client),
) -> FinancialProfileResponse:
    """Real Financial Intelligence: every metric is either a genuine
    SEC-EDGAR-derived value with full provenance (fiscal period, XBRL
    concept, filing source) or explicitly ``INSUFFICIENT_DATA`` — never a
    synthetic substitute (see ``data/sec_edgar_client.py::FinancialProfile``).
    """
    settings = get_settings()
    if not settings.orbitaliq_live_data_mode:
        profile_metrics = _all_metrics_insufficient(_LIVE_MODE_DISABLED_NOTE)
        return FinancialProfileResponse(
            ticker=ticker.upper(),
            company_name=None,
            cik=None,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            filing_source_url=None,
            metrics={k: v.as_dict() for k, v in profile_metrics.items()},
        )

    profile = sec_client.fetch_financial_profile(ticker)
    return FinancialProfileResponse(**profile.as_dict())


@router.post("/companies/resolve", response_model=CompanyResolveResponse)
def resolve_company_by_name(
    payload: CompanyResolveRequest,
    sec_client: SecEdgarClient = Depends(get_sec_client),
    geocoding_client: CensusGeocodingClient = Depends(get_geocoding_client),
) -> CompanyResolveResponse:
    """The "type a company name" front door: resolves a free-text query
    into a verified ticker/CIK (SEC EDGAR's own directory) and, when one
    can be genuinely established, a verified facility location (the
    company's real SEC-registered business address, geocoded via the US
    Census Bureau) — never a fabricated ticker, company, or coordinate.

    Returns one of four statuses (see ``data/company_resolver.py``):
    ``RESOLVED`` (ticker/CIK + verified coordinates), ``RESOLVED_NO_FACILITY``
    (ticker/CIK found, financial intelligence can run, but no verified
    coordinates exist — satellite intelligence must be reported
    INSUFFICIENT_DATA by the caller, never guessed), ``AMBIGUOUS`` (more
    than one SEC-registered company plausibly matches — the caller must
    choose from ``candidates``), or ``COMPANY_NOT_FOUND`` (no SEC-registered
    company matches at all).

    This endpoint is purely additive and read-only: it never creates or
    persists an assessment. The caller (the Streamlit "New Assessment" UI,
    or any other client) still submits the resolved fields to the existing
    ``POST /api/v1/assessments`` to actually run the pipeline.
    """
    settings = get_settings()
    if not settings.orbitaliq_live_data_mode:
        return CompanyResolveResponse(
            status="COMPANY_NOT_FOUND",
            query=payload.query,
            reason=(
                "ORBITALIQ_LIVE_DATA_MODE is disabled for this deployment — no live SEC EDGAR/Census "
                "Geocoder request was made, so company resolution cannot run."
            ),
        )

    resolution = resolve_company(
        payload.query, sec_client=sec_client, geocoder=geocoding_client, settings=settings,
    )
    return CompanyResolveResponse(**resolution.as_dict())


@router.get("/historical/{ticker}", response_model=HistoricalFinancialsResponse)
def get_historical_financials(
    ticker: str,
    max_years: int = Query(default=6, ge=1, le=15),
    sec_client: SecEdgarClient = Depends(get_sec_client),
) -> HistoricalFinancialsResponse:
    """Historical Analysis view: real, disclosed-only fiscal-year series.
    Missing years or line items are simply absent — never interpolated or
    fabricated.
    """
    settings = get_settings()
    if not settings.orbitaliq_live_data_mode:
        return HistoricalFinancialsResponse(
            ticker=ticker.upper(),
            company_name=None,
            cik=None,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            fiscal_years=[],
        )

    historical = sec_client.fetch_historical_financials(ticker, max_years=max_years)
    return HistoricalFinancialsResponse(**historical.as_dict())


@router.get("/satellite/change-pair", response_model=SatelliteVisualResponse)
def get_satellite_change_pair(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    lookback_days: int = Query(default=180, ge=1, le=3650),
    imagery_provider: NasaGibsImageryProvider = Depends(get_imagery_provider),
) -> SatelliteVisualResponse:
    """Satellite Change Detection View: before / after / diff imagery with
    full provenance and an honestly-computed effective resolution — see
    ``data/imagery_provider.py::build_satellite_visual_package``.

    In strict real-data mode, a fallback (non-live) tile pair is never
    surfaced as a "visual" — the caller gets an explicit unavailable reason
    instead of a synthetic image, per requirement #15 ("never show
    synthetic_fallback in strict enterprise mode").
    """
    settings = get_settings()
    if not settings.orbitaliq_live_data_mode:
        return SatelliteVisualResponse(
            live_data_mode=False,
            unavailable_reason=(
                "ORBITALIQ_LIVE_DATA_MODE is disabled for this deployment — no NASA "
                "GIBS request was made."
            ),
            visual=None,
        )

    pair = imagery_provider.fetch_change_pair(latitude, longitude, lookback_days=lookback_days)
    if settings.orbitaliq_strict_real_data_mode and not pair.live:
        return SatelliteVisualResponse(
            live_data_mode=True,
            unavailable_reason=(
                "Strict real-data mode: NASA GIBS did not return live imagery for this "
                "location/date pair, and synthetic imagery is never substituted or "
                f"displayed in strict mode. Detail: {pair.summary()}"
            ),
            visual=None,
        )
    return SatelliteVisualResponse(
        live_data_mode=True,
        unavailable_reason=None,
        visual=build_satellite_visual_package(pair),
    )


def _get_record_or_404(assessment_id: str, session: Session):
    repo = AssessmentRepository(session)
    record = repo.get(assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="assessment not found")
    return record


@router.get("/assessments/{assessment_id}/evidence", response_model=EvidenceResponse)
def get_assessment_evidence(assessment_id: str, session: Session = Depends(get_db_session)) -> EvidenceResponse:
    """Claim -> Source -> Document -> Retrieved -> Value -> Calculation ->
    Dashboard signal audit trail for this assessment, exactly as computed
    once by the orchestrator and persisted — this endpoint never
    re-derives it.
    """
    record = _get_record_or_404(assessment_id, session)
    return EvidenceResponse(assessment_id=assessment_id, evidence=record.evidence or [])


@router.get("/assessments/{assessment_id}/convergence", response_model=ConvergenceResponse)
def get_assessment_convergence(assessment_id: str, session: Session = Depends(get_db_session)) -> ConvergenceResponse:
    record = _get_record_or_404(assessment_id, session)
    return ConvergenceResponse(assessment_id=assessment_id, convergence=record.convergence)


@router.get("/assessments/{assessment_id}/data-quality", response_model=DataQualityResponse)
def get_assessment_data_quality(assessment_id: str, session: Session = Depends(get_db_session)) -> DataQualityResponse:
    record = _get_record_or_404(assessment_id, session)
    return DataQualityResponse(assessment_id=assessment_id, data_quality=record.data_quality)


@router.post("/compare", response_model=CompareResponse)
def compare_assessments(payload: CompareRequest, session: Session = Depends(get_db_session)) -> CompareResponse:
    """Competitor Comparison: multi-company side-by-side, sorted by
    momentum_score (a disclosed, deterministic sort) — never ranked by any
    subjective or hidden criterion.
    """
    repo = AssessmentRepository(session)
    found = []
    missing_ids: list[str] = []
    for assessment_id in payload.assessment_ids:
        record = repo.get(assessment_id)
        if record is None:
            missing_ids.append(assessment_id)
            continue
        found.append(record)

    found.sort(key=lambda r: r.momentum_score, reverse=True)
    results = [
        CompareResultItem(
            id=r.id,
            ticker=r.ticker,
            company_name=r.company_name,
            facility_name=r.facility_name,
            momentum_score=r.momentum_score,
            momentum_level=r.momentum_level,
            signal_breakdown=r.signal_breakdown,
            data_quality=r.data_quality,
            narrative_status=r.narrative_status,
            created_at=r.created_at,
        )
        for r in found
    ]
    return CompareResponse(results=results, missing_ids=missing_ids)


@router.get("/watchlist", response_model=WatchlistListResponse)
def list_watchlist(
    tier: str | None = Query(
        default=None,
        description=(
            "Filter to one tier: WATCH, REVIEW, HIGH_SIGNAL, or NONE (NONE returns "
            "every company/facility ever assessed, including de-escalated entries). "
            "Omitted: every entry currently on the watchlist in any tier."
        ),
    ),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db_session),
) -> WatchlistListResponse:
    """The current, deterministic-rules-only watchlist (see
    ``agents/watchlist_agent.py``), sorted by momentum_score descending.
    Derived from — and kept in sync with — the assessments that produced
    each entry; never independently computed here.
    """
    if tier is not None and tier not in {"NONE", "WATCH", "REVIEW", "HIGH_SIGNAL"}:
        raise HTTPException(status_code=422, detail="tier must be one of NONE, WATCH, REVIEW, HIGH_SIGNAL")

    repo = WatchlistRepository(session)
    entries = repo.list(tier=tier, limit=limit, offset=offset)
    return WatchlistListResponse(
        total=len(entries),
        results=[WatchlistEntryResponse(**e.to_dict()) for e in entries],
    )
