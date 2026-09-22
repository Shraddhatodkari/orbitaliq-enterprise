"""Evidence / audit trail builder.

For every quantitative claim the dashboard surfaces, this module produces
a single, uniform record: Claim -> Source -> Document/dataset -> Retrieved
timestamp -> Value -> Calculation -> Dashboard signal. It does no fetching
or computation of its own — it only re-packages values that were already
retrieved or computed elsewhere (SEC EDGAR, NASA GIBS, the deterministic
scoring engine) into an auditable trail. Operates on plain dicts (the same
``.as_dict()`` shapes the API already returns) rather than importing
dataclasses from the data layer, to keep this module trivially testable
and free of import coupling.

Audit-grade identity fields (additive): every item also carries an
``evidence_id`` (a real, freshly-generated UUID unique to this one claim —
never reused, never derived from content), the ``assessment_id`` of the
persisted ``AssessmentRecord`` this trail belongs to (``None`` until the
caller actually has one — e.g. a bare call to ``build_evidence_trail()``
outside the persistence path — never a fabricated placeholder id), a
``correlation_id`` tying every item back to the single API request/pipeline
run that produced it (again ``None`` when the caller has none to give), and
an ``agent``/``model`` pair naming exactly which internal pipeline
component produced the claim and, when the claim came from an actual ML
model rather than a deterministic computation, which one. ``model`` is
``None`` for every deterministic/rule-based claim (financial metrics,
signal-breakdown terms, the composite score) — it is never filled with a
placeholder just to have a value; only the satellite change-detection item,
which genuinely runs a CNN, carries a real model name.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    claim: str
    source_name: str
    source_url: str | None
    document: str | None
    retrieved_at: str | None
    value: str
    calculation: str
    dashboard_signal: str
    # --- Audit-grade identity (additive) -- see module docstring. ---
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str | None = None
    correlation_id: str | None = None
    agent: str = "unknown"
    model: str | None = None

    def as_dict(self) -> dict:
        return {
            "claim": self.claim,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "document": self.document,
            "retrieved_at": self.retrieved_at,
            "value": self.value,
            "calculation": self.calculation,
            "dashboard_signal": self.dashboard_signal,
            "evidence_id": self.evidence_id,
            "assessment_id": self.assessment_id,
            "correlation_id": self.correlation_id,
            "agent": self.agent,
            "model": self.model,
        }


def build_evidence_trail(
    *,
    company_name: str,
    ticker: str,
    facility_name: str,
    financial_metrics: dict[str, dict],
    filing_source_url: str | None,
    financial_retrieved_at: str | None,
    satellite_meta: dict,
    vision_findings: dict,
    momentum_score: float,
    momentum_level: str,
    signal_breakdown: dict[str, float],
    assessment_id: str | None = None,
    correlation_id: str | None = None,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []

    for key, metric in financial_metrics.items():
        if metric.get("status") != "AVAILABLE":
            continue
        calculation = (
            f"YoY change vs {metric.get('prior_fiscal_period')}: "
            f"{metric.get('yoy_change_pct') * 100:+.2f}%" if metric.get("yoy_change_pct") is not None
            else "Latest value disclosed for this fiscal period (no comparable prior period)"
        )
        items.append(
            EvidenceItem(
                claim=f"{metric.get('label')} for {company_name} ({ticker})",
                source_name="SEC EDGAR XBRL Company Facts",
                source_url=filing_source_url,
                document=f"{metric.get('fiscal_period')} 10-K — XBRL concept: {metric.get('xbrl_concept')}",
                retrieved_at=financial_retrieved_at,
                value=f"{metric.get('current_value')} {metric.get('unit')}",
                calculation=calculation,
                dashboard_signal=f"financial_profile.{key}",
                assessment_id=assessment_id,
                correlation_id=correlation_id,
                agent="ingestion_agent",
                model=None,
            )
        )

    if satellite_meta:
        items.append(
            EvidenceItem(
                claim=f"Bi-temporal satellite change detection over {facility_name}",
                source_name=satellite_meta.get("source_name", "NASA GIBS (VIIRS/MODIS)"),
                source_url=satellite_meta.get("source_url"),
                document=(
                    f"tile pair: current {satellite_meta.get('current_date')} vs "
                    f"prior {satellite_meta.get('prior_date')} "
                    f"({satellite_meta.get('resolution_m_per_pixel')} m/pixel, "
                    f"{satellite_meta.get('latitude')}, {satellite_meta.get('longitude')})"
                ),
                retrieved_at=satellite_meta.get("retrieved_at"),
                value=f"overall_change_score={vision_findings.get('overall_change_score')}",
                calculation="compute_change_indices (band-difference analytic path) blended with FacilityChangeCNN",
                dashboard_signal="vision_findings.overall_change_score",
                assessment_id=assessment_id,
                correlation_id=correlation_id,
                agent="vision_agent",
                # The one evidence item genuinely produced by an ML model
                # rather than a deterministic computation -- see
                # nvidia/vision_model.py::FacilityChangeCNN. device_used
                # (CPU/GPU) is already disclosed on vision_findings itself.
                model="FacilityChangeCNN",
            )
        )

    for key, value in signal_breakdown.items():
        items.append(
            EvidenceItem(
                claim=f"{key.replace('_', ' ').title()} contribution to the Competitive Expansion Signal",
                source_name="Deterministic scoring engine",
                source_url=None,
                document="core/intelligence_engine.py::compute_momentum_assessment",
                retrieved_at=None,
                value=f"{value:.2f}/100",
                calculation="Direct input to the weighted composite — see Score Breakdown for weights",
                dashboard_signal=f"signal_breakdown.{key}",
                assessment_id=assessment_id,
                correlation_id=correlation_id,
                agent="intelligence_scoring_agent",
                model=None,
            )
        )

    items.append(
        EvidenceItem(
            claim=f"Competitive Expansion Signal for {company_name} ({ticker})",
            source_name="Deterministic scoring engine",
            source_url=None,
            document="core/intelligence_engine.py::compute_momentum_assessment",
            retrieved_at=None,
            value=f"{momentum_level} (composite index {momentum_score:.1f}/100)",
            calculation=(
                "0.6 x financial composite + 0.4 x satellite composite, plus an independent-evidence "
                "convergence bonus when both composites clear 50/100 — see Score Breakdown"
            ),
            dashboard_signal="momentum_score",
            assessment_id=assessment_id,
            correlation_id=correlation_id,
            agent="intelligence_scoring_agent",
            model=None,
        )
    )

    return items
