import numpy as np
import pytest

from orbitaliq.agents.orchestrator import AssessmentOrchestrator
from orbitaliq.core.intelligence_engine import MomentumLevel
from orbitaliq.data.imagery_provider import ChangeTilePairResult, ImageryResult
from orbitaliq.data.sec_edgar_client import FinancialMetric, FinancialProfile, FinancialSignal
from orbitaliq.nvidia.nim_client import NimGenerationResult, NvidiaNimClient
from orbitaliq.nvidia.vision_model import VisionInferenceEngine


def test_orchestrator_end_to_end_offline_integration(sample_company):
    with AssessmentOrchestrator(
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert 0.0 <= result.momentum_score <= 100.0
    assert result.momentum_level in {lvl.value for lvl in MomentumLevel}
    assert result.llm_generated is False  # offline mode fixture-wide
    assert result.narrative_report
    assert set(result.signal_breakdown.keys()) == {
        "revenue_growth",
        "rd_investment",
        "capex_growth",
        "construction_expansion",
        "vegetation_clearing",
    }
    assert len(result.agent_trace) == 5
    assert [t.agent for t in result.agent_trace] == [
        "ingestion_agent",
        "vision_agent",
        "intelligence_scoring_agent",
        "report_agent",
        "watchlist_agent",
    ]
    assert all(t.status == "ok" for t in result.agent_trace)
    assert all(t.duration_ms >= 0 for t in result.agent_trace)
    assert result.data_sources  # offline mode still discloses provenance
    assert all(v == "offline_demo_mode" for v in result.data_sources.values())

    # --- Enterprise dashboard extensions: honest defaults in offline mode ---
    assert result.financial_profile is None  # no fetch_enriched_evidence on the default IngestionAgent's fake-free run... actually offline mode itself returns (None, None)
    assert result.convergence is not None
    assert result.convergence["outcome"] == "INSUFFICIENT_EVIDENCE"  # nothing is live in offline mode
    assert result.data_quality is not None
    assert result.data_quality["completeness_pct"] == 0.0  # every signal is offline_demo_mode, none "_live"
    assert result.evidence  # signal_breakdown + momentum score always produce evidence items
    assert result.narrative_status == "DETERMINISTIC_GROUNDED_TEMPLATE"  # offline NvidiaNimClient never calls the LLM
    assert result.narrative_grounding is None  # critic only runs against genuine LLM output
    assert result.satellite_visual is None  # no live change-pair was fetched
    assert result.watchlist_tier in {"NONE", "WATCH", "REVIEW", "HIGH_SIGNAL"}
    assert result.watchlist_webhook_status in {"NOT_APPLICABLE", "STUBBED_NO_WEBHOOK_CONFIGURED"}


def test_orchestrator_stamps_assessment_id_and_correlation_id_onto_every_evidence_item(sample_company):
    """api/routes_assessment.py pre-generates a real assessment id and a
    real per-request correlation id and passes both into orchestrator.run()
    -- confirms they reach every evidence item, and that every item also
    gets its own unique evidence_id, never left blank or fabricated.
    """
    with AssessmentOrchestrator(
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company, assessment_id="assessment-abc", correlation_id="request-xyz")

    assert result.evidence
    assert all(item["assessment_id"] == "assessment-abc" for item in result.evidence)
    assert all(item["correlation_id"] == "request-xyz" for item in result.evidence)
    evidence_ids = [item["evidence_id"] for item in result.evidence]
    assert len(evidence_ids) == len(set(evidence_ids))
    assert all(item.get("agent") for item in result.evidence)  # every item names its producing agent


def test_orchestrator_leaves_assessment_id_and_correlation_id_none_when_not_supplied(sample_company):
    with AssessmentOrchestrator(
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert result.evidence
    assert all(item["assessment_id"] is None for item in result.evidence)
    assert all(item["correlation_id"] is None for item in result.evidence)


def test_orchestrator_flags_watchlist_only_above_threshold(sample_company):
    # Force a guaranteed-STABLE momentum result so the watchlist should
    # never fire.
    class _StableIngestionAgent:
        def run(self, company):
            financials = {
                "revenue_growth_signal": FinancialSignal(0.02, True, "sec_edgar_live", "flat"),
                "rd_investment_signal": FinancialSignal(0.02, True, "sec_edgar_live", "flat"),
                "capex_growth_signal": FinancialSignal(0.02, True, "sec_edgar_live", "flat"),
            }
            current_tile = np.full((3, 64, 64), 0.5, dtype=np.float32)
            prior_tile = np.full((3, 64, 64), 0.5, dtype=np.float32)
            data_sources = {"revenue_growth_signal": "offline_demo_mode"}
            return financials, current_tile, prior_tile, data_sources

    with AssessmentOrchestrator(
        ingestion_agent=_StableIngestionAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert result.momentum_level == "STABLE"
    assert result.watchlist_flagged is False
    assert result.watchlist_channel is None
    assert result.watchlist_tier == "NONE"
    assert result.watchlist_webhook_status == "NOT_APPLICABLE"
    # This fake only implements run() (no fetch_enriched_evidence) -- the
    # orchestrator must degrade gracefully rather than raising.
    assert result.financial_profile is None
    assert result.satellite_visual is None


def test_orchestrator_propagates_and_traces_agent_failure(sample_company):
    class _FailingIngestionAgent:
        def run(self, company):
            raise RuntimeError("upstream SEC EDGAR client unavailable")

    with AssessmentOrchestrator(
        ingestion_agent=_FailingIngestionAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        with pytest.raises(RuntimeError, match="upstream SEC EDGAR client unavailable"):
            orch.run(sample_company)


def test_orchestrator_closes_owned_nim_client(sample_company):
    orch = AssessmentOrchestrator(vision_engine=VisionInferenceEngine(device_preference="cpu"))
    orch.run(sample_company)
    orch.close()  # should not raise


# --- Enterprise enrichment wiring: fetch_enriched_evidence + downstream views ---


def _live_signals():
    return {
        "revenue_growth_signal": FinancialSignal(0.9, True, "sec_edgar_live", "rev"),
        "rd_investment_signal": FinancialSignal(0.9, True, "sec_edgar_live", "rd"),
        "capex_growth_signal": FinancialSignal(0.9, True, "sec_edgar_live", "capex"),
    }


def _live_data_sources():
    return {
        "revenue_growth_signal": "sec_edgar_live",
        "rd_investment_signal": "sec_edgar_live",
        "capex_growth_signal": "sec_edgar_live",
        "satellite_imagery_current": "nasa_gibs_live",
        "satellite_imagery_prior": "nasa_gibs_live",
    }


def _change_pair():
    current = ImageryResult(
        tile=np.full((3, 64, 64), 0.9, dtype=np.float32), live=True, source="nasa_gibs_live", summary="ok",
        observation_date="2026-09-15", latitude=39.5, longitude=-119.4, layer="VIIRS", zoom=9,
        resolution_m_per_pixel=900.0,
    )
    prior = ImageryResult(
        tile=np.full((3, 64, 64), 0.1, dtype=np.float32), live=True, source="nasa_gibs_live", summary="ok",
        observation_date="2026-03-15", latitude=39.5, longitude=-119.4, layer="VIIRS", zoom=9,
        resolution_m_per_pixel=900.0,
    )
    return ChangeTilePairResult(current=current, prior=prior, lookback_days=180)


class _EnrichedIngestionAgent:
    """A fake that also implements fetch_enriched_evidence, exercising the
    orchestrator's getattr()/callable() wiring for the full enrichment path.
    """

    def __init__(self):
        self.financial_profile = FinancialProfile(
            ticker="HOT", company_name="Hot Corp", cik="0000320193", retrieved_at="2026-09-18T00:00:00Z",
            filing_source_url="https://data.sec.gov/x",
            metrics={
                "revenue_growth": FinancialMetric(
                    key="revenue_growth", label="Revenue Growth", category="growth", unit="percent",
                    status="AVAILABLE", current_value=0.9, prior_value=0.5, yoy_change_pct=0.4,
                    fiscal_period="FY2024", prior_fiscal_period="FY2023", xbrl_concept="Revenues",
                    source="sec_edgar_live",
                )
            },
        )

    def run(self, company):
        return _live_signals(), np.full((3, 64, 64), 0.9, dtype=np.float32), np.full((3, 64, 64), 0.1, dtype=np.float32), _live_data_sources()

    def fetch_enriched_evidence(self, company):
        return self.financial_profile, _change_pair()


def test_orchestrator_wires_full_enrichment_when_ingestion_agent_supports_it(sample_company):
    with AssessmentOrchestrator(
        ingestion_agent=_EnrichedIngestionAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert result.financial_profile is not None
    assert result.financial_profile["metrics"]["revenue_growth"]["status"] == "AVAILABLE"
    assert result.satellite_visual is not None
    assert result.satellite_visual["current_image_available"] is True
    assert result.convergence["financial_level"] in {"HIGH", "MODERATE", "LOW"}  # no longer INSUFFICIENT: all live
    assert result.convergence["physical_level"] in {"HIGH", "MODERATE", "LOW"}
    assert result.data_quality["completeness_pct"] == 100.0
    assert result.data_quality["source_count"] == 2  # sec_edgar_live + nasa_gibs_live
    # Evidence trail must include the financial metric, the satellite claim,
    # each signal_breakdown component, and the composite score.
    signals = {item["dashboard_signal"] for item in result.evidence}
    assert "financial_profile.revenue_growth" in signals
    assert "vision_findings.overall_change_score" in signals
    assert "momentum_score" in signals


def test_latest_fiscal_period_uses_max_across_all_available_metrics_not_just_revenue():
    """Regression test for the reported dashboard bug: the "Most recent
    fiscal period" freshness field showed FY2017 for an assessment (APLE)
    whose CapEx Growth, Operating Income Growth, FCF Trend, Asset Growth,
    ROA, ROE, Cash, Total Debt, and Debt/Equity metrics were all genuinely
    FY2025 -- because ``_latest_fiscal_period`` read only the
    ``revenue_growth`` metric's own fiscal_period, ignoring every other
    metric in the same profile. It must now reflect the genuinely most
    recent fiscal period evidenced anywhere in the profile.
    """
    profile = FinancialProfile(
        ticker="APLE", company_name="Apple Hospitality REIT, Inc.", cik="0001418121",
        retrieved_at="2026-09-18T00:00:00Z", filing_source_url="https://data.sec.gov/x",
        metrics={
            "revenue_growth": FinancialMetric(
                key="revenue_growth", label="Revenue Growth", category="growth", unit="USD",
                status="AVAILABLE", current_value=95_000_000, fiscal_period="FY2017",
                xbrl_concept="Revenues", source="sec_edgar_live",
            ),
            "operating_income_growth": FinancialMetric(
                key="operating_income_growth", label="Operating Income Growth", category="growth", unit="USD",
                status="AVAILABLE", current_value=25_000_000, fiscal_period="FY2025",
                xbrl_concept="OperatingIncomeLoss", source="sec_edgar_live",
            ),
            "roa": FinancialMetric(
                key="roa", label="Return on Assets", category="profitability", unit="percent",
                status="AVAILABLE", current_value=0.05, fiscal_period="FY2025",
                xbrl_concept="NetIncomeLoss / Assets", source="derived:sec_edgar",
            ),
            # An unavailable metric with no fiscal_period must never win or crash the comparison.
            "rd_growth": FinancialMetric(
                key="rd_growth", label="R&D Growth", category="growth", unit="USD",
                status="INSUFFICIENT_DATA", fiscal_period=None, source="insufficient_data",
            ),
        },
    )
    assert AssessmentOrchestrator._latest_fiscal_period(profile) == "FY2025"
    assert AssessmentOrchestrator._latest_fiscal_period(profile) != "FY2017"


def test_latest_fiscal_period_returns_none_when_profile_missing_or_empty():
    assert AssessmentOrchestrator._latest_fiscal_period(None) is None
    empty_profile = FinancialProfile(
        ticker="X", company_name=None, cik=None, retrieved_at="2026-09-18T00:00:00Z",
        filing_source_url=None, metrics={},
    )
    assert AssessmentOrchestrator._latest_fiscal_period(empty_profile) is None


def test_orchestrator_dashboard_freshness_reflects_latest_metric_not_just_revenue(sample_company):
    """End-to-end: the freshness summary the dashboard actually renders
    (``result.data_quality["freshness"]["most_recent_fiscal_period"]``)
    must reflect the latest AVAILABLE metric across the whole profile.
    """
    class _StaleRevenueIngestionAgent:
        def __init__(self):
            self.financial_profile = FinancialProfile(
                ticker="APLE", company_name="Apple Hospitality REIT, Inc.", cik="0001418121",
                retrieved_at="2026-09-18T00:00:00Z", filing_source_url="https://data.sec.gov/x",
                metrics={
                    "revenue_growth": FinancialMetric(
                        key="revenue_growth", label="Revenue Growth", category="growth", unit="USD",
                        status="AVAILABLE", current_value=95_000_000, fiscal_period="FY2017",
                        xbrl_concept="Revenues", source="sec_edgar_live",
                    ),
                    "total_debt": FinancialMetric(
                        key="total_debt", label="Total Debt", category="balance_sheet", unit="USD",
                        status="AVAILABLE", current_value=1_500_000_000, fiscal_period="FY2025",
                        xbrl_concept="LongTermDebt", source="sec_edgar_live",
                    ),
                },
            )

        def run(self, company):
            return (
                _live_signals(), np.full((3, 64, 64), 0.9, dtype=np.float32),
                np.full((3, 64, 64), 0.1, dtype=np.float32), _live_data_sources(),
            )

        def fetch_enriched_evidence(self, company):
            return self.financial_profile, _change_pair()

    with AssessmentOrchestrator(
        ingestion_agent=_StaleRevenueIngestionAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    freshness = result.data_quality["freshness"]
    assert freshness["most_recent_fiscal_period"] == "FY2025"
    assert freshness["most_recent_fiscal_period"] != "FY2017"


def test_orchestrator_enrichment_failure_degrades_gracefully(sample_company):
    """If fetch_enriched_evidence raises, the core assessment must still
    complete -- this is an optional, best-effort view, never a hard
    dependency of the pipeline.
    """
    class _BrokenEnrichmentAgent:
        def run(self, company):
            return _live_signals(), np.full((3, 64, 64), 0.9, dtype=np.float32), np.full((3, 64, 64), 0.1, dtype=np.float32), _live_data_sources()

        def fetch_enriched_evidence(self, company):
            raise RuntimeError("SEC EDGAR is down")

    with AssessmentOrchestrator(
        ingestion_agent=_BrokenEnrichmentAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)  # must not raise

    assert result.financial_profile is None
    assert result.satellite_visual is None
    assert result.momentum_score >= 0.0  # core pipeline still completed


# --- Live-mode data-sufficiency gate, exercised through the real pipeline
# end to end (see tests/test_intelligence_engine.py for the underlying unit
# coverage) -- proves orchestrator.run() actually threads data_sources
# through to the scoring agent rather than only the isolated function
# being correct. ---


class _AllFallbackIngestionAgent:
    """Simulates a live-mode run where SEC EDGAR AND NASA GIBS both
    genuinely failed for every signal -- everything fell back to the
    labeled synthetic generator. The real end-to-end pipeline must refuse
    to manufacture a score from this rather than silently scoring it.
    """

    def run(self, company):
        financials = {
            "revenue_growth_signal": FinancialSignal(0.9, False, "synthetic_fallback:revenue_growth", "SEC EDGAR down"),
            "rd_investment_signal": FinancialSignal(0.9, False, "synthetic_fallback:rd_investment", "SEC EDGAR down"),
            "capex_growth_signal": FinancialSignal(0.9, False, "synthetic_fallback:capex_growth", "SEC EDGAR down"),
        }
        current_tile = np.full((3, 64, 64), 0.9, dtype=np.float32)
        prior_tile = np.full((3, 64, 64), 0.1, dtype=np.float32)
        data_sources = {
            "revenue_growth_signal": "synthetic_fallback:revenue_growth",
            "rd_investment_signal": "synthetic_fallback:rd_investment",
            "capex_growth_signal": "synthetic_fallback:capex_growth",
            "satellite_imagery_current": "synthetic_fallback:satellite_tile",
            "satellite_imagery_prior": "synthetic_fallback:satellite_tile",
        }
        return financials, current_tile, prior_tile, data_sources


def test_orchestrator_reports_insufficient_data_when_every_live_source_fails(sample_company):
    """The real, non-negotiable requirement: if every underlying live data
    source failed, the end-to-end pipeline must report INSUFFICIENT_DATA
    rather than a number derived from the synthetic fallback -- even
    though the fallback values (0.9) would score ~90/100 if wrongly used.
    """
    with AssessmentOrchestrator(
        ingestion_agent=_AllFallbackIngestionAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert result.momentum_level == "INSUFFICIENT_DATA"
    assert result.momentum_score == 0.0
    assert result.signal_breakdown == {}
    assert result.watchlist_flagged is False
    assert result.watchlist_tier == "NONE"
    assert "insufficient" in result.narrative_report.lower() or "Insufficient" in result.narrative_report
    # Must not silently claim success -- the agent trace should surface it too.
    scoring_trace = next(t for t in result.agent_trace if t.agent == "intelligence_scoring_agent")
    assert scoring_trace.status == "ok"  # this is a legitimate, non-error outcome, not a crash
    assert "INSUFFICIENT_DATA" in scoring_trace.summary


class _PartialFallbackIngestionAgent:
    """Two of three financial signals live; one fell back. Satellite fully
    live. The real end-to-end pipeline must score off only the live
    financial signals, not all three.
    """

    def run(self, company):
        financials = {
            "revenue_growth_signal": FinancialSignal(0.8, True, "sec_edgar_live", "rev"),
            "rd_investment_signal": FinancialSignal(0.8, True, "sec_edgar_live", "rd"),
            "capex_growth_signal": FinancialSignal(0.05, False, "synthetic_fallback:capex_growth", "SEC EDGAR down for capex"),
        }
        current_tile = np.full((3, 64, 64), 0.7, dtype=np.float32)
        prior_tile = np.full((3, 64, 64), 0.1, dtype=np.float32)
        data_sources = {
            "revenue_growth_signal": "sec_edgar_live",
            "rd_investment_signal": "sec_edgar_live",
            "capex_growth_signal": "synthetic_fallback:capex_growth",
            "satellite_imagery_current": "nasa_gibs_live",
            "satellite_imagery_prior": "nasa_gibs_live",
        }
        return financials, current_tile, prior_tile, data_sources


def test_orchestrator_excludes_only_the_failed_signal_not_the_whole_assessment(sample_company):
    with AssessmentOrchestrator(
        ingestion_agent=_PartialFallbackIngestionAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert result.momentum_level != "INSUFFICIENT_DATA"  # enough real data existed to score legitimately
    assert "capex_growth" not in result.signal_breakdown
    assert "revenue_growth" in result.signal_breakdown
    assert result.momentum_score > 0.0


class _TypedInsufficientDataIngestionAgent:
    """Simulates the REAL current shape a live SecEdgarClient/NasaGibsImageryProvider
    failure now returns (value=None, source="insufficient_data") -- not the
    older synthetic_fallback:* fixtures above, which still carry a
    fabricated float. This is the escalated, non-negotiable case: the
    orchestrator (including its own trace-summary string formatting --
    see orchestrator.py's `revenue_growth_signal={...:.2f}` log line) must
    never crash trying to format/compute over a None value, end to end.
    """

    def run(self, company):
        financials = {
            "revenue_growth_signal": FinancialSignal(None, False, "insufficient_data", "SEC EDGAR down", status="INSUFFICIENT_DATA"),
            "rd_investment_signal": FinancialSignal(None, False, "insufficient_data", "SEC EDGAR down", status="INSUFFICIENT_DATA"),
            "capex_growth_signal": FinancialSignal(None, False, "insufficient_data", "SEC EDGAR down", status="INSUFFICIENT_DATA"),
        }
        current_tile = np.full((3, 64, 64), 0.9, dtype=np.float32)
        prior_tile = np.full((3, 64, 64), 0.1, dtype=np.float32)
        data_sources = {
            "revenue_growth_signal": "insufficient_data",
            "rd_investment_signal": "insufficient_data",
            "capex_growth_signal": "insufficient_data",
            "satellite_imagery_current": "insufficient_data",
            "satellite_imagery_prior": "insufficient_data",
        }
        return financials, current_tile, prior_tile, data_sources


def test_orchestrator_handles_typed_none_valued_signals_end_to_end_without_crashing(sample_company):
    with AssessmentOrchestrator(
        ingestion_agent=_TypedInsufficientDataIngestionAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(sample_company)  # must not raise TypeError formatting/multiplying None

    assert result.momentum_level == "INSUFFICIENT_DATA"
    assert result.momentum_score == 0.0
    assert result.signal_breakdown == {}
    assert all(t.status == "ok" for t in result.agent_trace)  # a clean INSUFFICIENT_DATA result, not a crash


# --- Grounding-gated narrative fallback ---


class _FabricatingNimClient:
    """A fake NvidiaNimClient whose 'LLM' cites a number with no basis in
    any evidence the pipeline computed -- the grounding critic must catch
    this and the orchestrator must fall back to the deterministic template.
    """

    def generate_intelligence_briefing(self, **kwargs):
        return NimGenerationResult(
            text="Momentum is strong, and next-quarter bookings are projected at $9,999,999,999.",
            model="fake-model", source="nvidia_nim", usage={},
        )

    def offline_briefing(self, **kwargs):
        return NimGenerationResult(
            text="Deterministic fallback narrative.", model="offline-deterministic-template-v1",
            source="derived_from_verified_evidence", usage={},
        )

    def close(self):
        pass


def test_orchestrator_blocks_ungrounded_llm_narrative_and_falls_back(sample_company):
    with AssessmentOrchestrator(
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=_FabricatingNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert result.narrative_status == "AI_GENERATED_BLOCKED_REPLACED_WITH_DETERMINISTIC"
    assert result.narrative_report == "Deterministic fallback narrative."
    assert result.llm_generated is False  # the final, published report is the deterministic fallback
    assert result.narrative_grounding is not None
    assert result.narrative_grounding["passed"] is False
    assert result.narrative_grounding["flagged_values"]


class _GroundedNimClient:
    """A fake NvidiaNimClient whose narrative cites only the momentum
    score, which is always in the grounded set -- the critic should pass
    this through unmodified.
    """

    def generate_intelligence_briefing(self, *, momentum_score, momentum_level, **kwargs):
        return NimGenerationResult(
            text=f"The composite momentum score is {momentum_score:.1f}/100, rated {momentum_level}.",
            model="fake-model", source="nvidia_nim", usage={},
        )

    def offline_briefing(self, **kwargs):
        raise AssertionError("offline_briefing should not be called for a grounded narrative")

    def close(self):
        pass


class _FinancialProfileCitingNimClient:
    """A fake NvidiaNimClient whose narrative cites a financial-profile
    percent metric by its x100 form -- exercising the orchestrator's
    financial_profile grounding-enrichment branch (raw fraction + percent
    form both added to the grounded set for ratio/percent-unit metrics).
    """

    def generate_intelligence_briefing(self, *, momentum_score, momentum_level, **kwargs):
        return NimGenerationResult(
            text=f"Momentum is {momentum_score:.1f}/100 ({momentum_level}), and revenue grew 40.0% year over year.",
            model="fake-model", source="nvidia_nim", usage={},
        )

    def offline_briefing(self, **kwargs):
        raise AssertionError("offline_briefing should not be called for a grounded narrative")

    def close(self):
        pass


def test_orchestrator_grounds_llm_narrative_against_financial_profile_metrics(sample_company):
    with AssessmentOrchestrator(
        ingestion_agent=_EnrichedIngestionAgent(),  # yoy_change_pct=0.4 for revenue_growth
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=_FinancialProfileCitingNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert result.narrative_status == "AI_GENERATED_GROUNDED"
    assert result.narrative_grounding["passed"] is True


# --- No verified facility coordinates (financial-only assessment) end to
# end -- see data/company_resolver.py's RESOLVED_NO_FACILITY status: the
# real, non-negotiable requirement is that satellite/vision evidence is
# never attempted from a guessed coordinate, while financial intelligence
# still scores normally off the real, available signals. ---


class _NoFacilityIngestionAgent:
    """Simulates IngestionAgent.run()'s real no-coordinates branch (see
    agents/ingestion_agent.py): financial signals are real and live, but
    current_tile/prior_tile are None and the satellite provenance keys carry
    the documented insufficient_data:no_verified_facility_coordinates label.
    """

    def run(self, company):
        financials = _live_signals()
        data_sources = {
            "revenue_growth_signal": "sec_edgar_live",
            "rd_investment_signal": "sec_edgar_live",
            "capex_growth_signal": "sec_edgar_live",
            "satellite_imagery_current": "insufficient_data:no_verified_facility_coordinates",
            "satellite_imagery_prior": "insufficient_data:no_verified_facility_coordinates",
        }
        return financials, None, None, data_sources


def test_orchestrator_no_facility_coordinates_skips_vision_but_scores_financials(sample_company):
    no_facility_company = sample_company.__class__(
        ticker=sample_company.ticker, company_name=sample_company.company_name,
        facility_name="No verified facility", latitude=None, longitude=None,
        industry=sample_company.industry, market_cap_usd=sample_company.market_cap_usd,
    )

    with AssessmentOrchestrator(
        ingestion_agent=_NoFacilityIngestionAgent(),
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=NvidiaNimClient(),
    ) as orch:
        result = orch.run(no_facility_company)  # must not raise

    # Financial signals scored normally -- real data, not fabricated.
    assert "revenue_growth" in result.signal_breakdown
    assert "rd_investment" in result.signal_breakdown
    assert result.momentum_level != "INSUFFICIENT_DATA"
    assert result.momentum_score > 0.0

    # Vision/satellite was never attempted from a guessed coordinate.
    assert "construction_expansion" not in result.signal_breakdown
    assert "vegetation_clearing" not in result.signal_breakdown
    assert result.vision_findings["status"] == "INSUFFICIENT_DATA"
    assert result.vision_findings["overall_change_score"] is None
    assert result.vision_findings["construction_expansion_signal"] is None
    assert result.satellite_visual is None

    vision_trace = next(t for t in result.agent_trace if t.agent == "vision_agent")
    assert vision_trace.status == "ok"  # a legitimate skip, not a crash
    assert "no verified facility coordinates" in vision_trace.summary.lower()

    # Physical/satellite convergence must be honestly INSUFFICIENT, never guessed.
    assert result.convergence["physical_level"] == "INSUFFICIENT"


def test_orchestrator_passes_through_grounded_llm_narrative(sample_company):
    with AssessmentOrchestrator(
        vision_engine=VisionInferenceEngine(device_preference="cpu"),
        nim_client=_GroundedNimClient(),
    ) as orch:
        result = orch.run(sample_company)

    assert result.narrative_status == "AI_GENERATED_GROUNDED"
    assert result.llm_generated is True
    assert result.narrative_grounding["passed"] is True
