"""Pydantic request/response models for the public HTTP API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class CompanyAssessmentRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20, examples=["TSLA"])
    company_name: str = Field(..., min_length=1, max_length=255, examples=["Tesla, Inc."])
    facility_name: str = Field(..., min_length=1, max_length=255, examples=["Gigafactory Nevada"])
    # None exactly when no verified facility location exists for this
    # assessment (see data/company_resolver.py's RESOLVED_NO_FACILITY
    # status / POST /api/v1/companies/resolve) -- a financial-only
    # assessment. Existing callers that already send real coordinates are
    # completely unaffected; this is purely additive. NEVER a signal to
    # guess a coordinate -- when omitted, satellite intelligence is always
    # reported INSUFFICIENT_DATA, never estimated.
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    industry: str = Field(default="general", max_length=120)
    market_cap_usd: float | None = Field(default=None, ge=0)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker must not be blank")
        return v

    @field_validator("company_name", "facility_name")
    @classmethod
    def _strip_names(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _lat_lon_together(self) -> "CompanyAssessmentRequest":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must both be provided, or both omitted")
        return self


class CompanyResolveRequest(BaseModel):
    """POST /api/v1/companies/resolve — the "type a company name" front
    door. Only a free-text query; every other field is resolved from
    authoritative sources server-side (see data/company_resolver.py).
    """

    query: str = Field(..., min_length=1, max_length=255, examples=["Tesla"])

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class CompanyCandidateSchema(BaseModel):
    ticker: str
    cik: str
    company_name: str


class CompanyResolveResponse(BaseModel):
    # "RESOLVED" | "RESOLVED_NO_FACILITY" | "AMBIGUOUS" | "COMPANY_NOT_FOUND"
    status: str
    query: str
    ticker: str | None = None
    cik: str | None = None
    company_name: str | None = None
    facility_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    facility_source: str | None = None
    facility_note: str | None = None
    candidates: list[CompanyCandidateSchema] = Field(default_factory=list)
    reason: str | None = None


class AgentTraceEntrySchema(BaseModel):
    agent: str
    status: str
    duration_ms: float
    summary: str


class AssessmentResponse(BaseModel):
    id: str
    ticker: str
    company_name: str
    facility_name: str
    # None exactly for a financial-only assessment (no verified facility
    # coordinates) -- see CompanyAssessmentRequest above.
    latitude: float | None
    longitude: float | None
    industry: str
    market_cap_usd: float | None
    momentum_score: float
    momentum_level: str
    signal_breakdown: dict[str, float]
    vision_findings: dict
    recommended_actions: list[str]
    narrative_report: str
    llm_generated: bool
    data_sources: dict[str, str] = Field(default_factory=dict)
    agent_trace: list[AgentTraceEntrySchema] = Field(default_factory=list)
    watchlist_flagged: bool = False
    watchlist_channel: str | None = None
    watchlist_tier: str = "NONE"
    watchlist_webhook_status: str = "NOT_APPLICABLE"
    # --- Enterprise dashboard extensions (all additive) ---
    financial_profile: dict | None = None
    convergence: dict | None = None
    data_quality: dict | None = None
    evidence: list[dict] = Field(default_factory=list)
    narrative_status: str = "DETERMINISTIC_GROUNDED_TEMPLATE"
    narrative_grounding: dict | None = None
    satellite_visual: dict | None = None
    score_breakdown: dict | None = None
    created_at: datetime | None = None


class AssessmentListResponse(BaseModel):
    total_matching: int
    results: list[AssessmentResponse]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    offline_mode: bool
    live_data_mode: bool
    vision_device: str


class FinancialMetricSchema(BaseModel):
    key: str
    label: str
    category: str
    unit: str
    status: str
    current_value: float | None = None
    prior_value: float | None = None
    yoy_change_pct: float | None = None
    fiscal_period: str | None = None
    prior_fiscal_period: str | None = None
    xbrl_concept: str | None = None
    source: str
    note: str | None = None


class FinancialProfileResponse(BaseModel):
    ticker: str
    company_name: str | None
    cik: str | None
    retrieved_at: str
    filing_source_url: str | None
    metrics: dict[str, FinancialMetricSchema]


class HistoricalFinancialsResponse(BaseModel):
    ticker: str
    company_name: str | None
    cik: str | None
    retrieved_at: str
    fiscal_years: list[dict]


class SatelliteChangePairRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    lookback_days: int = Field(default=180, ge=1, le=3650)


class SatelliteVisualResponse(BaseModel):
    live_data_mode: bool
    unavailable_reason: str | None = None
    visual: dict | None = None


class EvidenceResponse(BaseModel):
    assessment_id: str
    evidence: list[dict] = Field(default_factory=list)


class ConvergenceResponse(BaseModel):
    assessment_id: str
    convergence: dict | None = None


class DataQualityResponse(BaseModel):
    assessment_id: str
    data_quality: dict | None = None


class CompareRequest(BaseModel):
    assessment_ids: list[str] = Field(..., min_length=2, max_length=10)


class CompareResultItem(BaseModel):
    id: str
    ticker: str
    company_name: str
    facility_name: str
    momentum_score: float
    momentum_level: str
    signal_breakdown: dict[str, float]
    data_quality: dict | None = None
    narrative_status: str
    created_at: datetime | None = None


class CompareResponse(BaseModel):
    # Sorted by momentum_score descending — a disclosed, deterministic sort,
    # not a subjective ranking or peer-group judgment.
    sorted_by: str = "momentum_score_desc"
    results: list[CompareResultItem]
    missing_ids: list[str] = Field(default_factory=list)


class WatchlistEntryResponse(BaseModel):
    id: str
    ticker: str
    company_name: str
    facility_name: str
    tier: str
    channel: str | None
    reason: str
    webhook_status: str
    momentum_score: float
    momentum_level: str
    latest_assessment_id: str
    first_flagged_at: str | None
    updated_at: str | None


class WatchlistListResponse(BaseModel):
    total: int
    results: list[WatchlistEntryResponse]
