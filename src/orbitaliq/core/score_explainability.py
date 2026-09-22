"""Explainable score breakdown — a "Why this score?" view built entirely
from values the pipeline already computed, never a second, separate
estimate.

This module does no scoring of its own. It packages the *real, already-
computed* terms of ``core/intelligence_engine.py``'s formula (financial
composite, satellite composite, convergence bonus, weights) alongside a
per-signal "Why?" drill-down (which signals were trusted, their real
values, their source, and — when excluded — the documented reason) and the
already-computed signal-convergence classification
(``core/convergence.py``) reused, not reinvented, for the "contradictory
signals" framing an analyst needs before citing this score in a
deliverable.

An analyst who wants to know "why is this score 67.3, not some other
number" gets a line-by-line answer traceable to the exact formula in
``intelligence_engine.py`` — the same "auditable, not a black box" design
goal that module's own docstring states.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from orbitaliq.core.convergence import SignalConvergence
from orbitaliq.core.data_quality import DataQualitySummary
from orbitaliq.core.intelligence_engine import (
    CONVERGENCE_BONUS_MAX,
    CONVERGENCE_THRESHOLD,
    FINANCIAL_WEIGHT,
    SATELLITE_WEIGHT,
    MomentumAssessmentResult,
)
from orbitaliq.data.sec_edgar_client import FinancialSignal
from orbitaliq.nvidia.vision_model import VisionFindings

# (signal name as used in signal_breakdown/excluded_signals, the
# FinancialSignal key it corresponds to -- None for the two satellite
# signals, which come from VisionFindings instead -- and a one-line,
# fixed description of what the signal measures and how it's derived).
# The description text is static and never derived from a live value, so
# it can never itself be a source of fabrication -- only the per-signal
# value/trusted/source/reason fields below carry real, per-assessment data.
_SIGNAL_DEFS = [
    (
        "revenue_growth", "revenue_growth_signal",
        "SEC EDGAR-disclosed YoY revenue growth, normalized to a 0-1 index over a "
        "calibrated (-10%..+30%) range, then scaled to 0-100.",
    ),
    (
        "rd_investment", "rd_investment_signal",
        "SEC EDGAR-disclosed YoY R&D expense growth, normalized to a 0-1 index over a "
        "calibrated (-20%..+40%) range, then scaled to 0-100.",
    ),
    (
        "capex_growth", "capex_growth_signal",
        "SEC EDGAR-disclosed YoY capital-expenditure growth, normalized to a 0-1 index "
        "over a calibrated (-30%..+50%) range, then scaled to 0-100.",
    ),
    (
        "construction_expansion", None,
        "Bi-temporal NASA GIBS satellite change detection (band-difference analytic "
        "path + CNN), 0-100: how much new construction/bare-earth the tile pair shows.",
    ),
    (
        "vegetation_clearing", None,
        "Bi-temporal NASA GIBS satellite change detection (band-difference analytic "
        "path + CNN), 0-100: how much vegetation clearing the tile pair shows.",
    ),
]


@dataclass
class SignalExplanation:
    signal: str
    trusted: bool
    value_0_100: float | None
    source: str | None
    reason: str
    description: str

    def as_dict(self) -> dict:
        return {
            "signal": self.signal,
            "trusted": self.trusted,
            "value_0_100": round(self.value_0_100, 2) if self.value_0_100 is not None else None,
            "source": self.source,
            "reason": self.reason,
            "description": self.description,
        }


@dataclass
class ScoreBreakdown:
    final_score: float
    momentum_level: str
    financial_contribution: float | None  # financial_composite * FINANCIAL_WEIGHT (or the composite itself if 100% reweighted)
    satellite_contribution: float | None  # satellite_composite * SATELLITE_WEIGHT (or the composite itself if 100% reweighted)
    convergence_bonus: float
    weights: dict
    data_confidence_pct: float
    contradictory_signals: bool
    convergence_outcome: str
    convergence_explanation: str
    signal_explanations: list[SignalExplanation] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "final_score": round(self.final_score, 2),
            "momentum_level": self.momentum_level,
            "financial_contribution": (
                round(self.financial_contribution, 2) if self.financial_contribution is not None else None
            ),
            "satellite_contribution": (
                round(self.satellite_contribution, 2) if self.satellite_contribution is not None else None
            ),
            "convergence_bonus": round(self.convergence_bonus, 2),
            "weights": self.weights,
            "data_confidence_pct": self.data_confidence_pct,
            "contradictory_signals": self.contradictory_signals,
            "convergence_outcome": self.convergence_outcome,
            "convergence_explanation": self.convergence_explanation,
            "signal_explanations": [s.as_dict() for s in self.signal_explanations],
        }


def _signal_reason(
    name: str,
    financial_key: str | None,
    momentum_result: MomentumAssessmentResult,
    financial_signals: dict[str, FinancialSignal] | None,
) -> str:
    """A real, non-fabricated reason string for one signal's inclusion or
    exclusion, sourced from the actual FinancialSignal.summary (financial
    signals) already produced upstream, or a fixed, honest explanation for
    the two satellite signals (which don't carry a per-signal summary —
    see VisionFindings).
    """
    trusted = name not in momentum_result.excluded_signals
    if financial_key is not None and financial_signals is not None and financial_key in financial_signals:
        signal = financial_signals[financial_key]
        if signal.summary:
            return signal.summary
    if trusted:
        return "Live/derived value included in the composite."
    return "Excluded from the score — live fetch did not produce a trusted value; never averaged in or treated as zero."


def build_score_breakdown(
    *,
    momentum_result: MomentumAssessmentResult,
    convergence: SignalConvergence,
    data_quality: DataQualitySummary,
    financial_signals: dict[str, FinancialSignal] | None = None,
    vision_findings: VisionFindings | None = None,
) -> ScoreBreakdown:
    """Build the full "Why this score?" breakdown from already-computed
    results — never recomputes the score itself, and never introduces a
    new scoring term. Reuses ``core/convergence.py``'s own
    ``SignalConvergence`` for the "contradictory signals" framing:
    ``contradictory_signals`` is true exactly when convergence's own
    ``outcome`` is ``SIGNAL_CONFLICT`` — the same classification the
    Evidence/Convergence view already reports, not a second, separate
    check with its own thresholds.
    """
    financial_contribution = (
        momentum_result.financial_composite * FINANCIAL_WEIGHT
        if momentum_result.financial_composite is not None and momentum_result.satellite_composite is not None
        else momentum_result.financial_composite  # 100% reweighted -- the composite IS the contribution
    )
    satellite_contribution = (
        momentum_result.satellite_composite * SATELLITE_WEIGHT
        if momentum_result.financial_composite is not None and momentum_result.satellite_composite is not None
        else momentum_result.satellite_composite
    )

    signal_explanations = []
    for name, financial_key, description in _SIGNAL_DEFS:
        trusted = name not in momentum_result.excluded_signals
        source = None
        if financial_key is not None and financial_signals is not None and financial_key in financial_signals:
            source = financial_signals[financial_key].source
        signal_explanations.append(
            SignalExplanation(
                signal=name,
                trusted=trusted,
                value_0_100=momentum_result.signal_breakdown.get(name),
                source=source,
                reason=_signal_reason(name, financial_key, momentum_result, financial_signals),
                description=description,
            )
        )

    return ScoreBreakdown(
        final_score=momentum_result.momentum_score,
        momentum_level=momentum_result.momentum_level.value,
        financial_contribution=financial_contribution,
        satellite_contribution=satellite_contribution,
        convergence_bonus=momentum_result.convergence_bonus,
        weights={
            "financial_weight": FINANCIAL_WEIGHT,
            "satellite_weight": SATELLITE_WEIGHT,
            "convergence_bonus_max": CONVERGENCE_BONUS_MAX,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
        },
        data_confidence_pct=data_quality.completeness_pct,
        contradictory_signals=(convergence.outcome == "SIGNAL_CONFLICT"),
        convergence_outcome=convergence.outcome,
        convergence_explanation=convergence.explanation,
        signal_explanations=signal_explanations,
    )
