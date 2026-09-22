"""Report Agent — turns the deterministic momentum result into an
executive-ready competitive-intelligence briefing via the NVIDIA
NIM-hosted LLM (or a deterministic offline template; see
``nvidia.nim_client``).
"""
from __future__ import annotations

from orbitaliq.agents.types import CompanyInput
from orbitaliq.core.intelligence_engine import MomentumAssessmentResult
from orbitaliq.nvidia.nim_client import NimGenerationResult, NvidiaNimClient
from orbitaliq.nvidia.vision_model import VisionFindings


class ReportAgent:
    def __init__(self, client: NvidiaNimClient) -> None:
        self._client = client

    def run(
        self,
        company: CompanyInput,
        momentum_result: MomentumAssessmentResult,
        vision_findings: VisionFindings,
    ) -> NimGenerationResult:
        return self._client.generate_intelligence_briefing(
            company_name=company.company_name,
            ticker=company.ticker,
            facility_name=company.facility_name,
            momentum_score=momentum_result.momentum_score,
            momentum_level=momentum_result.momentum_level.value,
            signal_breakdown=momentum_result.signal_breakdown,
            vision_findings=vision_findings.as_dict(),
        )
