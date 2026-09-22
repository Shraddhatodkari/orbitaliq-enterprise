"""Repository layer — the only module allowed to issue ORM queries.

Keeping persistence behind a narrow repository interface means the agents
and API layers never import SQLAlchemy directly, which keeps the domain
logic unit-testable without a database.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from orbitaliq.core.logging_config import logger
from orbitaliq.data.models import AssessmentRecord, WatchlistEntry

# Maps a scoring-engine FinancialSignal key to the corresponding key in
# MomentumAssessmentResult.signal_breakdown (see
# core/intelligence_engine.py::compute_momentum_assessment -- the
# `signal_values` dict there uses these shorter names). Only signals that
# were themselves trusted/live at persistence time ever appear in a
# persisted record's `signal_breakdown`, so a hit here is always a
# genuine, previously-observed real value -- never a fabricated one.
_SIGNAL_TO_BREAKDOWN_KEY = {
    "revenue_growth_signal": "revenue_growth",
    "rd_investment_signal": "rd_investment",
    "capex_growth_signal": "capex_growth",
}


def lookup_cached_financial_value(ticker: str, signal_key: str) -> tuple[float, str] | None:
    """Genuine ``CACHED_REAL`` support (see core/data_quality_state.py):
    when a live fetch for ``signal_key`` fails for ``ticker``, look up the
    most recent prior assessment for this ticker that actually persisted a
    trusted value for this exact signal, and return
    ``(value, original_retrieved_at_iso)`` so the caller can disclose both
    the reused value and how stale it is.

    Returns ``None`` on a clean miss (no prior assessment has this signal)
    or on any failure to query the database -- this is a best-effort,
    additive enrichment and must never raise or block an assessment.
    Opens its own short-lived session (rather than requiring a caller to
    thread one through) so it can be injected as a plain, stateless
    callable into ``IngestionAgent`` -- see
    ``agents/ingestion_agent.py::IngestionAgent.__init__``.
    """
    breakdown_key = _SIGNAL_TO_BREAKDOWN_KEY.get(signal_key)
    if breakdown_key is None:
        return None
    try:
        # Imported here, not at module level, to avoid a circular import
        # (data/database.py imports data/models.py, which this module
        # already imports; this keeps the dependency direction simple and
        # avoids paying that import cost for every caller of this module
        # that never actually needs a cache lookup).
        from orbitaliq.data.database import SessionLocal

        with SessionLocal() as session:
            stmt = (
                select(AssessmentRecord)
                # Case-insensitive on both sides: a persisted ticker isn't
                # guaranteed to already be upper-cased (it's stored exactly
                # as the assessment request supplied it), so comparing only
                # `ticker.upper()` against a raw column would silently miss
                # a real cache hit for a lowercase/mixed-case persisted
                # ticker.
                .where(func.upper(AssessmentRecord.ticker) == ticker.upper())
                .order_by(AssessmentRecord.created_at.desc())
                .limit(50)
            )
            for record in session.execute(stmt).scalars():
                value = (record.signal_breakdown or {}).get(breakdown_key)
                if value is not None:
                    retrieved_at = record.created_at.isoformat() if record.created_at else None
                    return value / 100.0, retrieved_at
    except Exception as exc:  # noqa: BLE001 - best-effort cache lookup, never fails the assessment
        logger.warning(f"cached_real lookup failed for {ticker}/{signal_key}: {exc}")
    return None


class AssessmentRepository:
    def __init__(self, session: Session):
        self._session = session

    def add(self, record: AssessmentRecord) -> AssessmentRecord:
        self._session.add(record)
        self._session.flush()
        return record

    def get(self, assessment_id: str) -> AssessmentRecord | None:
        return self._session.get(AssessmentRecord, assessment_id)

    def list(
        self,
        *,
        min_momentum_score: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AssessmentRecord]:
        stmt = select(AssessmentRecord).order_by(AssessmentRecord.created_at.desc())
        if min_momentum_score is not None:
            stmt = stmt.where(AssessmentRecord.momentum_score >= min_momentum_score)
        stmt = stmt.offset(offset).limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def count(self) -> int:
        return self._session.query(AssessmentRecord).count()


class WatchlistRepository:
    """Upserts and reads the derived, queryable watchlist index (see
    ``data/models.py::WatchlistEntry`` for why this is derived rather than
    a second source of truth).
    """

    def __init__(self, session: Session):
        self._session = session

    def upsert_from_assessment(self, record: AssessmentRecord) -> WatchlistEntry:
        stmt = select(WatchlistEntry).where(
            WatchlistEntry.ticker == record.ticker,
            WatchlistEntry.facility_name == record.facility_name,
        )
        entry = self._session.execute(stmt).scalars().first()
        now = datetime.now(timezone.utc)

        if entry is None:
            entry = WatchlistEntry(
                ticker=record.ticker,
                company_name=record.company_name,
                facility_name=record.facility_name,
                first_flagged_at=now if record.watchlist_tier != "NONE" else None,
            )
            self._session.add(entry)

        entry.company_name = record.company_name
        entry.tier = record.watchlist_tier
        entry.channel = record.watchlist_channel
        entry.reason = (
            f"momentum level {record.momentum_level} -> {record.watchlist_tier} tier"
            if record.watchlist_tier != "NONE"
            else "momentum level STABLE — below the WATCH tier's EMERGING threshold"
        )
        entry.webhook_status = record.watchlist_webhook_status
        entry.momentum_score = record.momentum_score
        entry.momentum_level = record.momentum_level
        entry.latest_assessment_id = record.id
        if entry.first_flagged_at is None and record.watchlist_tier != "NONE":
            entry.first_flagged_at = now
        entry.updated_at = now

        self._session.flush()
        return entry

    def list(self, *, tier: str | None = None, limit: int = 100, offset: int = 0) -> list[WatchlistEntry]:
        stmt = select(WatchlistEntry).order_by(WatchlistEntry.momentum_score.desc())
        if tier is None:
            # Default view: only companies actually on the watchlist in
            # some capacity — pass tier="NONE" explicitly to see everyone
            # ever assessed, including de-escalated entries.
            stmt = stmt.where(WatchlistEntry.tier != "NONE")
        elif tier != "NONE":
            stmt = stmt.where(WatchlistEntry.tier == tier)
        # tier == "NONE": no filter at all — every entry, watchlisted or not.
        stmt = stmt.offset(offset).limit(limit)
        return list(self._session.execute(stmt).scalars().all())
