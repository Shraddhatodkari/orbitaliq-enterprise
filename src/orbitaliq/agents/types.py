"""Shared data contracts passed between pipeline agents.

Defining these once and importing everywhere keeps each agent's
input/output contract explicit, which is what makes the pipeline safe to
unit-test agent-by-agent and to reorder/replace agents later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CompanyInput:
    ticker: str
    company_name: str
    facility_name: str
    # None exactly when no verified facility location exists for this
    # assessment (see data/company_resolver.py's RESOLVED_NO_FACILITY
    # status) -- a financial-only assessment. Every existing caller
    # supplies real floats, so this is purely additive; NEVER a guessed
    # coordinate substituted for a missing one.
    latitude: float | None
    longitude: float | None
    industry: str = "general"
    market_cap_usd: float | None = None


@dataclass
class AgentTraceEntry:
    agent: str
    status: str  # "ok" | "error"
    duration_ms: float
    summary: str

    def as_dict(self) -> dict:
        return {
            "agent": self.agent,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "summary": self.summary,
        }


@dataclass
class IntelligenceAssessmentResult:
    company_input: CompanyInput
    momentum_score: float
    momentum_level: str
    signal_breakdown: dict
    vision_findings: dict
    recommended_actions: list[str]
    narrative_report: str
    llm_generated: bool
    data_sources: dict = field(default_factory=dict)
    agent_trace: list[AgentTraceEntry] = field(default_factory=list)
    watchlist_flagged: bool = False
    watchlist_channel: str | None = None
    # "NONE" | "WATCH" | "REVIEW" | "HIGH_SIGNAL" — see agents/watchlist_agent.py.
    watchlist_tier: str = "NONE"
    # "NOT_APPLICABLE" | "STUBBED_NO_WEBHOOK_CONFIGURED" | "DISPATCHED" | "DISPATCH_FAILED"
    watchlist_webhook_status: str = "NOT_APPLICABLE"
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Enterprise dashboard extensions (all additive; every existing
    # field above is unchanged) ---
    # Real, SEC-EDGAR-only financial metrics (FinancialProfile.as_dict()),
    # or None when the enrichment fetch didn't run (offline demo mode) or
    # the ticker/CIK couldn't be resolved. Never a synthetic substitute.
    financial_profile: dict | None = None
    # SignalConvergence.as_dict() — independent Financial/Physical/External
    # evidence classification, never forced into agreement.
    convergence: dict | None = None
    # DataQualitySummary.as_dict() — source availability/freshness/
    # completeness derived from data_sources.
    data_quality: dict | None = None
    # Flat Claim->Source->Document->Retrieved->Value->Calculation->Signal
    # audit trail (list of EvidenceItem.as_dict()).
    evidence: list = field(default_factory=list)
    # "AI_GENERATED_GROUNDED" | "AI_GENERATED_BLOCKED_REPLACED_WITH_DETERMINISTIC"
    # | "DETERMINISTIC_GROUNDED_TEMPLATE" — always states plainly whether
    # narrative_report is genuine LLM output or the deterministic template
    # (built directly from the same verified evidence, never generated —
    # hence "GROUNDED" rather than the older, misleadingly demo-sounding
    # "OFFLINE_TEMPLATE" naming; this template path runs in full live-data
    # mode too, whenever no NIM key is configured or the critic blocks an
    # LLM narrative), per the "never publish an unsupported numerical
    # claim" requirement.
    narrative_status: str = "DETERMINISTIC_GROUNDED_TEMPLATE"
    # GroundingResult.as_dict() when the critic actually ran (i.e. the NIM
    # produced text), else None.
    narrative_grounding: dict | None = None
    # build_satellite_visual_package() — before/after/diff PNGs + full
    # provenance, or None when no live change-pair was fetched.
    satellite_visual: dict | None = None
    # core/score_explainability.py::ScoreBreakdown.as_dict() — the "Why
    # this score?" view: the real financial/satellite/convergence
    # contributions the formula actually used, a data-confidence % (from
    # data_quality), whether convergence found contradictory signals, and
    # a per-signal drill-down (trusted/value/source/reason). Never a
    # second, separate estimate of the score — built entirely from values
    # already computed elsewhere in this same assessment.
    score_breakdown: dict | None = None
