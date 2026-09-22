"""Intelligence Scoring Agent — fuses financial signals + vision findings
into the deterministic composite momentum score (see
``core.intelligence_engine``).
"""
from __future__ import annotations

from orbitaliq.core.intelligence_engine import MomentumAssessmentResult, compute_momentum_assessment
from orbitaliq.data.sec_edgar_client import FinancialSignal
from orbitaliq.nvidia.vision_model import VisionFindings


class IntelligenceScoringAgent:
    def run(
        self,
        financial_signals: dict[str, FinancialSignal],
        vision_findings: VisionFindings,
        data_sources: dict[str, str] | None = None,
    ) -> MomentumAssessmentResult:
        return compute_momentum_assessment(
            financial_signals=financial_signals, vision_findings=vision_findings, data_sources=data_sources
        )
