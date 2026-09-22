"""Agentic pipeline orchestrator.

Runs the five specialist agents (ingestion -> vision -> intelligence
scoring -> report -> watchlist) in sequence, recording a timed, per-agent
audit trail (``AgentTraceEntry``) so every assessment is explainable
end-to-end — which agent ran, how long it took, and what it concluded.
Each agent is injected as a dependency so the whole pipeline (and each
stage in isolation) is testable with fakes/mocks and no network or GPU
required.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from orbitaliq.agents.ingestion_agent import IngestionAgent
from orbitaliq.agents.intelligence_scoring_agent import IntelligenceScoringAgent
from orbitaliq.agents.report_agent import ReportAgent
from orbitaliq.agents.types import AgentTraceEntry, CompanyInput, IntelligenceAssessmentResult
from orbitaliq.agents.vision_agent import VisionAgent
from orbitaliq.agents.watchlist_agent import WatchlistAgent
from orbitaliq.core.convergence import assess_convergence
from orbitaliq.core.data_quality import build_data_quality_summary
from orbitaliq.core.evidence import build_evidence_trail
from orbitaliq.core.grounding import build_grounded_values, check_narrative_grounding
from orbitaliq.core.logging_config import logger
from orbitaliq.core.score_explainability import build_score_breakdown
from orbitaliq.data.imagery_provider import build_satellite_visual_package
from orbitaliq.nvidia.nim_client import NvidiaNimClient
from orbitaliq.nvidia.vision_model import VisionFindings, VisionInferenceEngine


@dataclass
class _StageTimer:
    agent_name: str
    trace: list[AgentTraceEntry]

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def record(self, summary: str, status: str = "ok") -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.trace.append(AgentTraceEntry(agent=self.agent_name, status=status, duration_ms=elapsed_ms, summary=summary))

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self.trace.append(
                AgentTraceEntry(agent=self.agent_name, status="error", duration_ms=elapsed_ms, summary=str(exc))
            )
        return False  # never suppress the exception


class AssessmentOrchestrator:
    """Coordinates the full ingestion -> vision -> scoring -> report -> watchlist pipeline."""

    def __init__(
        self,
        ingestion_agent: IngestionAgent | None = None,
        vision_agent: VisionAgent | None = None,
        intelligence_scoring_agent: IntelligenceScoringAgent | None = None,
        report_agent: ReportAgent | None = None,
        watchlist_agent: WatchlistAgent | None = None,
        vision_engine: VisionInferenceEngine | None = None,
        nim_client: NvidiaNimClient | None = None,
    ) -> None:
        self._owns_nim_client = nim_client is None
        self._nim_client = nim_client or NvidiaNimClient()
        vision_engine = vision_engine or VisionInferenceEngine()

        self._ingestion_agent = ingestion_agent or IngestionAgent()
        self._vision_agent = vision_agent or VisionAgent(vision_engine)
        self._intelligence_scoring_agent = intelligence_scoring_agent or IntelligenceScoringAgent()
        self._report_agent = report_agent or ReportAgent(self._nim_client)
        self._watchlist_agent = watchlist_agent or WatchlistAgent()

    def close(self) -> None:
        if self._owns_nim_client:
            self._nim_client.close()
        # Safe even for test fakes that don't define close().
        getattr(self._ingestion_agent, "close", lambda: None)()

    def __enter__(self) -> "AssessmentOrchestrator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def run(
        self,
        company: CompanyInput,
        *,
        assessment_id: str | None = None,
        correlation_id: str | None = None,
    ) -> IntelligenceAssessmentResult:
        """Runs the full pipeline. ``assessment_id``/``correlation_id`` are
        optional, caller-supplied identifiers (see api/routes_assessment.py,
        which pre-generates the persisted record's real id and the
        per-request id before calling this) stamped onto every evidence
        item so the audit trail can be traced back to the exact assessment
        record and API request that produced it. Left as ``None`` (never
        fabricated) for callers — direct orchestrator use, scripts, tests —
        that don't have one to give.
        """
        trace: list[AgentTraceEntry] = []
        logger.info(f"Starting competitive-intelligence assessment for '{company.ticker}' / {company.facility_name}")

        with _StageTimer("ingestion_agent", trace) as stage:
            financial_signals, current_tile, prior_tile, data_sources = self._ingestion_agent.run(company)
            live_count = sum(1 for v in data_sources.values() if str(v).endswith("_live"))
            revenue_signal = financial_signals["revenue_growth_signal"]

            # Optional, additive real-data enrichment for the enterprise
            # dashboard (Real Financial Intelligence + Satellite Change
            # Detection View). Called defensively via getattr()/callable()
            # so ingestion-agent test fakes that only implement run() (see
            # tests/test_orchestrator.py) keep working unchanged, and so a
            # failure in this *optional* view never breaks the core,
            # already-tested assessment pipeline.
            financial_profile = None
            change_pair = None
            enrich = getattr(self._ingestion_agent, "fetch_enriched_evidence", None)
            if callable(enrich):
                try:
                    financial_profile, change_pair = enrich(company)
                except Exception as exc:  # noqa: BLE001 - enrichment is best-effort only
                    logger.warning(f"fetch_enriched_evidence failed for {company.ticker}: {exc}")

            revenue_signal_note = (
                f"{revenue_signal.value:.2f}" if revenue_signal.value is not None else revenue_signal.status
            )
            tile_note = f"{current_tile.shape} satellite tile pair" if current_tile is not None else "no satellite tile pair (no verified facility coordinates)"
            stage.record(
                f"fetched financial signals and {tile_note} "
                f"({live_count}/{len(data_sources)} signals live; "
                f"revenue_growth_signal={revenue_signal_note}; "
                f"financial_profile={'available' if financial_profile else 'unavailable'})"
            )

        with _StageTimer("vision_agent", trace) as stage:
            if current_tile is None or prior_tile is None:
                # No verified facility coordinates for this assessment (see
                # data/company_resolver.py) -- vision inference is never
                # attempted, and NEVER fed a guessed/synthetic tile pair.
                # See VisionFindings.status.
                vision_findings = VisionFindings(
                    construction_expansion_signal=None, vegetation_clearing_signal=None,
                    overall_change_score=None, change_energy=None,
                    device_used="not_attempted_no_verified_facility_coordinates", model_finetuned=False,
                    status="INSUFFICIENT_DATA",
                )
                stage.record("skipped -- no verified facility coordinates for this assessment", status="ok")
            else:
                vision_findings = self._vision_agent.run(current_tile, prior_tile)
                stage.record(
                    f"overall_change_score={vision_findings.overall_change_score:.2f} "
                    f"device={vision_findings.device_used}"
                )

        with _StageTimer("intelligence_scoring_agent", trace) as stage:
            momentum_result = self._intelligence_scoring_agent.run(
                financial_signals, vision_findings, data_sources=data_sources
            )
            excluded_note = f"; excluded (live-mode fallback/unavailable)={momentum_result.excluded_signals}" if momentum_result.excluded_signals else ""
            stage.record(
                f"momentum_score={momentum_result.momentum_score:.1f} level={momentum_result.momentum_level.value}"
                f"{excluded_note}"
            )

        with _StageTimer("report_agent", trace) as stage:
            report = self._report_agent.run(company, momentum_result, vision_findings)
            narrative_status, narrative_grounding, report = self._ground_narrative(
                report=report,
                company=company,
                momentum_result=momentum_result,
                vision_findings=vision_findings,
                financial_profile=financial_profile,
            )
            stage.record(f"narrative generated via source={report.source}; narrative_status={narrative_status}")

        with _StageTimer("watchlist_agent", trace) as stage:
            watchlist_decision = self._watchlist_agent.run(
                company.company_name, momentum_result.momentum_level, momentum_result.momentum_score
            )
            stage.record(watchlist_decision.reason)

        logger.info(
            f"Completed assessment for '{company.ticker}': "
            f"momentum_score={momentum_result.momentum_score:.1f} level={momentum_result.momentum_level.value}"
        )

        convergence = self._assess_convergence(momentum_result, data_sources)
        data_quality = build_data_quality_summary(
            data_sources=data_sources,
            strict_mode=self._ingestion_agent_settings_strict_mode(),
            fiscal_period=self._latest_fiscal_period(financial_profile),
            satellite_current_date=change_pair.current.observation_date if change_pair else None,
            satellite_prior_date=change_pair.prior.observation_date if change_pair else None,
            assessed_at=datetime.now(timezone.utc).isoformat(),
        )
        satellite_meta = self._satellite_meta(change_pair)
        evidence = build_evidence_trail(
            company_name=company.company_name,
            ticker=company.ticker,
            facility_name=company.facility_name,
            financial_metrics=(
                {k: v.as_dict() for k, v in financial_profile.metrics.items()} if financial_profile else {}
            ),
            filing_source_url=financial_profile.filing_source_url if financial_profile else None,
            financial_retrieved_at=financial_profile.retrieved_at if financial_profile else None,
            satellite_meta=satellite_meta,
            vision_findings=vision_findings.as_dict(),
            momentum_score=momentum_result.momentum_score,
            momentum_level=momentum_result.momentum_level.value,
            signal_breakdown=momentum_result.signal_breakdown,
            assessment_id=assessment_id,
            correlation_id=correlation_id,
        )
        satellite_visual = build_satellite_visual_package(change_pair) if change_pair is not None else None
        score_breakdown = build_score_breakdown(
            momentum_result=momentum_result,
            convergence=convergence,
            data_quality=data_quality,
            financial_signals=financial_signals,
            vision_findings=vision_findings,
        )

        return IntelligenceAssessmentResult(
            company_input=company,
            momentum_score=momentum_result.momentum_score,
            momentum_level=momentum_result.momentum_level.value,
            signal_breakdown=momentum_result.signal_breakdown,
            vision_findings=vision_findings.as_dict(),
            recommended_actions=momentum_result.recommended_actions,
            narrative_report=report.text,
            llm_generated=report.source == "nvidia_nim",
            data_sources=data_sources,
            agent_trace=trace,
            watchlist_flagged=watchlist_decision.flagged,
            watchlist_channel=watchlist_decision.channel,
            watchlist_tier=watchlist_decision.tier,
            watchlist_webhook_status=watchlist_decision.webhook_status,
            financial_profile=financial_profile.as_dict() if financial_profile else None,
            convergence=convergence.as_dict(),
            data_quality=data_quality.as_dict(),
            evidence=[item.as_dict() for item in evidence],
            narrative_status=narrative_status,
            narrative_grounding=narrative_grounding,
            satellite_visual=satellite_visual,
            score_breakdown=score_breakdown.as_dict(),
        )

    def _ground_narrative(
        self,
        *,
        report,
        company: CompanyInput,
        momentum_result,
        vision_findings,
        financial_profile,
    ):
        """Run the deterministic grounding critic against an LLM-generated
        narrative and fall back to the deterministic template if it's
        blocked.

        A narrative produced by the deterministic template
        (``report.source == "derived_from_verified_evidence"``) is
        definitionally grounded — it is built directly from these same
        numbers, never generated — so it's never run through the critic;
        only genuine NVIDIA NIM output is checked.
        """
        if report.source != "nvidia_nim":
            return "DETERMINISTIC_GROUNDED_TEMPLATE", None, report

        grounded_values = build_grounded_values(
            momentum_result.momentum_score,
            list(momentum_result.signal_breakdown.values()),
            [
                vision_findings.overall_change_score,
                vision_findings.construction_expansion_signal,
                vision_findings.vegetation_clearing_signal,
                vision_findings.change_energy,
            ],
        )
        if financial_profile is not None:
            for metric in financial_profile.metrics.values():
                if metric.status != "AVAILABLE":
                    continue
                for value in (metric.current_value, metric.prior_value):
                    if value is None:
                        continue
                    grounded_values |= build_grounded_values(value)
                    if metric.unit in ("percent", "ratio"):
                        grounded_values |= build_grounded_values(value * 100)
                if metric.yoy_change_pct is not None:
                    grounded_values |= build_grounded_values(metric.yoy_change_pct)
                    grounded_values |= build_grounded_values(metric.yoy_change_pct * 100)

        grounding_result = check_narrative_grounding(report.text, grounded_values)
        if grounding_result.passed:
            return "AI_GENERATED_GROUNDED", grounding_result.as_dict(), report

        logger.warning(
            f"Grounding critic BLOCKED an NVIDIA NIM narrative for {company.ticker} "
            f"(unsupported numbers: {grounding_result.flagged_values}); "
            "falling back to the deterministic offline template."
        )
        fallback_report = self._nim_client.offline_briefing(
            company_name=company.company_name,
            ticker=company.ticker,
            facility_name=company.facility_name,
            momentum_score=momentum_result.momentum_score,
            momentum_level=momentum_result.momentum_level.value,
            signal_breakdown=momentum_result.signal_breakdown,
            vision_findings=vision_findings.as_dict(),
        )
        return "AI_GENERATED_BLOCKED_REPLACED_WITH_DETERMINISTIC", grounding_result.as_dict(), fallback_report

    def _assess_convergence(self, momentum_result, data_sources: dict):
        financial = momentum_result.signal_breakdown
        financial_composite = (
            financial.get("revenue_growth", 0.0) + financial.get("rd_investment", 0.0) + financial.get("capex_growth", 0.0)
        ) / 3
        satellite_composite = (
            financial.get("construction_expansion", 0.0) + financial.get("vegetation_clearing", 0.0)
        ) / 2
        financial_available = all(
            str(data_sources.get(k, "")).endswith("_live")
            for k in ("revenue_growth_signal", "rd_investment_signal", "capex_growth_signal")
        )
        physical_available = all(
            str(data_sources.get(k, "")).endswith("_live")
            for k in ("satellite_imagery_current", "satellite_imagery_prior")
        )
        # No independent "external/public disclosure" evidence source is
        # wired into the deterministic pipeline yet (see the Multi-Source
        # Competitive Intelligence surface) — honestly reported as
        # INSUFFICIENT rather than inferred from the other two buckets.
        return assess_convergence(
            financial_score=financial_composite,
            financial_available=financial_available,
            physical_score=satellite_composite,
            physical_available=physical_available,
            external_score=None,
            external_available=False,
        )

    def _ingestion_agent_settings_strict_mode(self) -> bool:
        settings = getattr(self._ingestion_agent, "_settings", None)
        return bool(getattr(settings, "orbitaliq_strict_real_data_mode", False))

    @staticmethod
    def _latest_fiscal_period(financial_profile) -> str | None:
        """The most recent fiscal period genuinely evidenced ANYWHERE in
        this assessment's financial profile — the max across every
        AVAILABLE metric's own ``fiscal_period``, not just
        ``revenue_growth``'s. Reading a single arbitrary metric here made
        the dashboard's freshness summary show a stale period (e.g.
        FY2017) whenever that one metric's own source tag happened to be
        older than others — even while several other metrics on the same
        assessment genuinely reflected the latest disclosed fiscal year.
        Never infers or fabricates a period: only ``fiscal_period`` values
        already present on an ``AVAILABLE`` metric are considered.
        """
        if financial_profile is None:
            return None
        latest_fy: int | None = None
        latest_label: str | None = None
        for metric in financial_profile.metrics.values():
            if metric.status != "AVAILABLE" or not metric.fiscal_period:
                continue
            try:
                fy = int(metric.fiscal_period.removeprefix("FY"))
            except (TypeError, ValueError):
                continue
            if latest_fy is None or fy > latest_fy:
                latest_fy = fy
                latest_label = metric.fiscal_period
        return latest_label

    @staticmethod
    def _satellite_meta(change_pair) -> dict:
        if change_pair is None:
            return {}
        current = change_pair.current
        return {
            "source_name": "NASA GIBS (VIIRS/MODIS)",
            "source_url": "https://gibs.earthdata.nasa.gov/wmts/",
            "current_date": current.observation_date,
            "prior_date": change_pair.prior.observation_date,
            "resolution_m_per_pixel": current.resolution_m_per_pixel,
            "latitude": current.latitude,
            "longitude": current.longitude,
            "retrieved_at": None,
        }
