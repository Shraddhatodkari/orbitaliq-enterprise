"""Strategic Signal Convergence — a deterministic check for whether the
independently-derived FINANCIAL, PHYSICAL, and EXTERNAL evidence groups
actually tell the same story, rather than assuming they always do.

Each group is classified HIGH / MODERATE / LOW / INSUFFICIENT from its own
already-computed score (never re-derived by an LLM), and the three levels
are combined by an explicit, auditable rule table — never forced into
agreement. Convergence is deliberately conservative: it only reports
SIGNAL_CONVERGENCE when financial and satellite evidence both independently
read HIGH and external evidence doesn't contradict that; anything short of
that is reported honestly as partial, conflicting, or insufficient.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

HIGH_THRESHOLD = 66.0
MODERATE_THRESHOLD = 33.0


class ConvergenceLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class ConvergenceOutcome(str, Enum):
    CONVERGE = "SIGNAL_CONVERGENCE"
    PARTIAL = "PARTIAL_SIGNAL_CONVERGENCE"
    CONFLICT = "SIGNAL_CONFLICT"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass
class SignalConvergence:
    financial_level: str
    physical_level: str
    external_level: str
    outcome: str
    explanation: str

    def as_dict(self) -> dict:
        return {
            "financial_level": self.financial_level,
            "physical_level": self.physical_level,
            "external_level": self.external_level,
            "outcome": self.outcome,
            "explanation": self.explanation,
        }


def classify_level(score_0_100: float | None, *, available: bool) -> str:
    if not available or score_0_100 is None:
        return ConvergenceLevel.INSUFFICIENT.value
    if score_0_100 >= HIGH_THRESHOLD:
        return ConvergenceLevel.HIGH.value
    if score_0_100 >= MODERATE_THRESHOLD:
        return ConvergenceLevel.MODERATE.value
    return ConvergenceLevel.LOW.value


def assess_convergence(
    *,
    financial_score: float | None,
    financial_available: bool,
    physical_score: float | None,
    physical_available: bool,
    external_score: float | None,
    external_available: bool,
) -> SignalConvergence:
    financial_level = classify_level(financial_score, available=financial_available)
    physical_level = classify_level(physical_score, available=physical_available)
    external_level = classify_level(external_score, available=external_available)

    if financial_level == ConvergenceLevel.INSUFFICIENT.value or physical_level == ConvergenceLevel.INSUFFICIENT.value:
        return SignalConvergence(
            financial_level=financial_level, physical_level=physical_level, external_level=external_level,
            outcome=ConvergenceOutcome.INSUFFICIENT.value,
            explanation=(
                "At least one of the two primary evidence groups (financial filings, satellite change "
                "detection) does not have enough live data to support a convergence judgment."
            ),
        )

    if financial_level == physical_level == ConvergenceLevel.HIGH.value:
        if external_level in (ConvergenceLevel.HIGH.value, ConvergenceLevel.MODERATE.value):
            return SignalConvergence(
                financial_level=financial_level, physical_level=physical_level, external_level=external_level,
                outcome=ConvergenceOutcome.CONVERGE.value,
                explanation=(
                    "Financial evidence and satellite change detection independently indicate strong expansion, "
                    "corroborated by external/public disclosure evidence."
                ),
            )
        return SignalConvergence(
            financial_level=financial_level, physical_level=physical_level, external_level=external_level,
            outcome=ConvergenceOutcome.PARTIAL.value,
            explanation=(
                "Financial evidence and satellite change detection independently indicate strong expansion, "
                "but external/public disclosure evidence is limited or unavailable to corroborate it."
            ),
        )

    if {financial_level, physical_level} == {ConvergenceLevel.HIGH.value, ConvergenceLevel.LOW.value}:
        return SignalConvergence(
            financial_level=financial_level, physical_level=physical_level, external_level=external_level,
            outcome=ConvergenceOutcome.CONFLICT.value,
            explanation=(
                "Financial evidence and satellite change detection point in opposing directions and should not "
                "be treated as confirming the same underlying story."
            ),
        )

    return SignalConvergence(
        financial_level=financial_level, physical_level=physical_level, external_level=external_level,
        outcome=ConvergenceOutcome.PARTIAL.value,
        explanation=(
            f"Financial expansion: {financial_level}. Satellite change: {physical_level}. "
            f"Public disclosure evidence: {external_level}. The signals do not fully align, and only a partial "
            "convergence can be reported."
        ),
    )
