"""SQLAlchemy ORM models for persisted competitive-intelligence assessments."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class AssessmentRecord(Base):
    """A single, immutable competitive-momentum assessment run for one
    company facility.
    """

    __tablename__ = "assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)

    ticker: Mapped[str] = mapped_column(String(20))
    company_name: Mapped[str] = mapped_column(String(255))
    facility_name: Mapped[str] = mapped_column(String(255))
    # Nullable: None exactly for a financial-only assessment with no
    # verified facility coordinates (see data/company_resolver.py). See
    # data/database.py::_migrate_sqlite_relax_lat_lon_not_null for the
    # in-place repair this required for a pre-existing SQLite database
    # whose 'assessments' table was created before this column allowed
    # NULL -- SQLite has no ALTER COLUMN, so that's a table rebuild, not a
    # simple ADD COLUMN like the other additive migrations below.
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    industry: Mapped[str] = mapped_column(String(120), default="general")
    market_cap_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    momentum_score: Mapped[float] = mapped_column(Float)
    momentum_level: Mapped[str] = mapped_column(String(30))

    signal_breakdown: Mapped[dict] = mapped_column(JSON, default=dict)
    vision_findings: Mapped[dict] = mapped_column(JSON, default=dict)
    recommended_actions: Mapped[list] = mapped_column(JSON, default=list)
    narrative_report: Mapped[str] = mapped_column(String)
    agent_trace: Mapped[list] = mapped_column(JSON, default=list)
    llm_generated: Mapped[bool] = mapped_column(default=False)
    # Per-signal provenance: which financial/imagery inputs came from a real
    # live public API (SEC EDGAR / NASA GIBS) vs. the labeled offline
    # fallback — see agents/ingestion_agent.py.
    data_sources: Mapped[dict] = mapped_column(JSON, default=dict)

    watchlist_flagged: Mapped[bool] = mapped_column(default=False)
    watchlist_channel: Mapped[str | None] = mapped_column(String(80), nullable=True)
    # "NONE" | "WATCH" | "REVIEW" | "HIGH_SIGNAL" — see agents/watchlist_agent.py.
    watchlist_tier: Mapped[str] = mapped_column(String(20), default="NONE")
    watchlist_webhook_status: Mapped[str] = mapped_column(String(40), default="NOT_APPLICABLE")

    # --- Enterprise dashboard extensions (all additive/nullable; every
    # existing column above is unchanged) — see
    # agents/types.py::IntelligenceAssessmentResult for what populates each.
    # Real, SEC-EDGAR-only financial metrics (FinancialProfile.as_dict()),
    # or null when unavailable — never a synthetic substitute.
    financial_profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # SignalConvergence.as_dict().
    convergence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # DataQualitySummary.as_dict().
    data_quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Flat Claim->Source->Document->Retrieved->Value->Calculation->Signal
    # audit trail (list of EvidenceItem.as_dict()).
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    # "AI_GENERATED_GROUNDED" | "AI_GENERATED_BLOCKED_REPLACED_WITH_DETERMINISTIC"
    # | "DETERMINISTIC_GROUNDED_TEMPLATE" (formerly "DETERMINISTIC_OFFLINE_
    # TEMPLATE" -- renamed for honesty: this path runs in full live-data
    # mode too, and is "grounded" because it's built directly from already-
    # verified evidence, never generated. See
    # data/database.py::_migrate_narrative_status_labels for the one-time
    # migration that relabels any already-persisted row using the old
    # value.) Always discloses whether narrative_report is genuine LLM
    # output or the deterministic template.
    narrative_status: Mapped[str] = mapped_column(String(60), default="DETERMINISTIC_GROUNDED_TEMPLATE")
    # GroundingResult.as_dict() when the critic actually ran, else null.
    narrative_grounding: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # build_satellite_visual_package() — before/after/diff PNGs + full
    # provenance, or null when no live change-pair was fetched.
    satellite_visual: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # core/score_explainability.py::ScoreBreakdown.as_dict() — the "Why
    # this score?" view, or null for a row persisted before this field
    # existed (see the additive-column migration below).
    score_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "facility_name": self.facility_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "industry": self.industry,
            "market_cap_usd": self.market_cap_usd,
            "momentum_score": self.momentum_score,
            "momentum_level": self.momentum_level,
            "signal_breakdown": self.signal_breakdown,
            "vision_findings": self.vision_findings,
            "recommended_actions": self.recommended_actions,
            "narrative_report": self.narrative_report,
            "agent_trace": self.agent_trace,
            "llm_generated": self.llm_generated,
            "data_sources": self.data_sources,
            "watchlist_flagged": self.watchlist_flagged,
            "watchlist_channel": self.watchlist_channel,
            "watchlist_tier": self.watchlist_tier,
            "watchlist_webhook_status": self.watchlist_webhook_status,
            "financial_profile": self.financial_profile,
            "convergence": self.convergence,
            "data_quality": self.data_quality,
            "evidence": self.evidence,
            "narrative_status": self.narrative_status,
            "narrative_grounding": self.narrative_grounding,
            "satellite_visual": self.satellite_visual,
            "score_breakdown": self.score_breakdown,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class WatchlistEntry(Base):
    """The current watchlist status for one company/facility — one row per
    (ticker, facility_name), upserted every time a new assessment is
    created for that pair (see api/routes_assessment.py). This lets the
    dashboard's Watchlist view list who's currently flagged without
    re-running every historical assessment; the assessment itself remains
    the source of truth (``AssessmentRecord.watchlist_tier`` etc.) — this
    table is a derived, queryable index over it, not a second source of
    truth.
    """

    __tablename__ = "watchlist_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)

    ticker: Mapped[str] = mapped_column(String(20))
    company_name: Mapped[str] = mapped_column(String(255))
    facility_name: Mapped[str] = mapped_column(String(255))

    # "NONE" | "WATCH" | "REVIEW" | "HIGH_SIGNAL"
    tier: Mapped[str] = mapped_column(String(20), default="NONE")
    channel: Mapped[str | None] = mapped_column(String(80), nullable=True)
    reason: Mapped[str] = mapped_column(String(255), default="")
    webhook_status: Mapped[str] = mapped_column(String(40), default="NOT_APPLICABLE")

    momentum_score: Mapped[float] = mapped_column(Float)
    momentum_level: Mapped[str] = mapped_column(String(30))

    latest_assessment_id: Mapped[str] = mapped_column(String(36))
    first_flagged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "facility_name": self.facility_name,
            "tier": self.tier,
            "channel": self.channel,
            "reason": self.reason,
            "webhook_status": self.webhook_status,
            "momentum_score": self.momentum_score,
            "momentum_level": self.momentum_level,
            "latest_assessment_id": self.latest_assessment_id,
            "first_flagged_at": self.first_flagged_at.isoformat() if self.first_flagged_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
