import pytest

from orbitaliq.core.convergence import assess_convergence
from orbitaliq.core.data_quality import build_data_quality_summary
from orbitaliq.core.intelligence_engine import compute_momentum_assessment
from orbitaliq.core.score_explainability import build_score_breakdown
from orbitaliq.data.sec_edgar_client import FinancialSignal
from orbitaliq.nvidia.vision_model import VisionFindings


def _financials(**overrides) -> dict[str, FinancialSignal]:
    defaults = dict(
        revenue_growth_signal=FinancialSignal(0.95, True, "sec_edgar_live", "FY2023 vs FY2022: +30.0% YoY"),
        rd_investment_signal=FinancialSignal(0.9, True, "sec_edgar_live", "FY2023 vs FY2022: +35.0% YoY"),
        capex_growth_signal=FinancialSignal(0.9, True, "sec_edgar_live", "FY2023 vs FY2022: +40.0% YoY"),
    )
    defaults.update(overrides)
    return defaults


def _vision(**overrides) -> VisionFindings:
    defaults = dict(
        construction_expansion_signal=0.9, vegetation_clearing_signal=0.85, overall_change_score=0.9,
        change_energy=0.5, device_used="cpu", model_finetuned=False,
    )
    defaults.update(overrides)
    return VisionFindings(**defaults)


_LIVE_SOURCES = {
    "revenue_growth_signal": "sec_edgar_live", "rd_investment_signal": "sec_edgar_live",
    "capex_growth_signal": "sec_edgar_live", "satellite_imagery_current": "nasa_gibs_live",
    "satellite_imagery_prior": "nasa_gibs_live",
}


def test_breakdown_reproduces_the_real_score_via_its_own_contributions():
    financials = _financials()
    vision = _vision()
    momentum = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=_LIVE_SOURCES)
    convergence = assess_convergence(
        financial_score=momentum.financial_composite, financial_available=True,
        physical_score=momentum.satellite_composite, physical_available=True,
        external_score=None, external_available=False,
    )
    data_quality = build_data_quality_summary(data_sources=_LIVE_SOURCES, strict_mode=True)

    breakdown = build_score_breakdown(
        momentum_result=momentum, convergence=convergence, data_quality=data_quality, financial_signals=financials,
    )

    reconstructed = breakdown.financial_contribution + breakdown.satellite_contribution + breakdown.convergence_bonus
    assert reconstructed == pytest.approx(momentum.momentum_score, abs=0.01)
    assert breakdown.final_score == pytest.approx(momentum.momentum_score)
    assert breakdown.momentum_level == momentum.momentum_level.value


def test_breakdown_exposes_real_weights_matching_the_scoring_engine():
    financials = _financials()
    vision = _vision()
    momentum = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=_LIVE_SOURCES)
    convergence = assess_convergence(
        financial_score=momentum.financial_composite, financial_available=True,
        physical_score=momentum.satellite_composite, physical_available=True,
        external_score=None, external_available=False,
    )
    data_quality = build_data_quality_summary(data_sources=_LIVE_SOURCES, strict_mode=True)
    breakdown = build_score_breakdown(momentum_result=momentum, convergence=convergence, data_quality=data_quality)

    assert breakdown.weights["financial_weight"] == 0.6
    assert breakdown.weights["satellite_weight"] == 0.4
    assert breakdown.weights["convergence_bonus_max"] == 8.0


def test_breakdown_contradictory_signals_reuses_convergence_outcome_not_a_new_check():
    """SIGNAL_CONFLICT (financial HIGH, satellite LOW or vice versa) must
    flip contradictory_signals -- and it must come from convergence's own
    outcome, not a separately invented threshold.
    """
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.95, True, "sec_edgar_live", "s"),
        rd_investment_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
        capex_growth_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
    )
    vision = _vision(construction_expansion_signal=0.05, vegetation_clearing_signal=0.05, overall_change_score=0.05)
    momentum = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=_LIVE_SOURCES)
    convergence = assess_convergence(
        financial_score=momentum.financial_composite, financial_available=True,
        physical_score=momentum.satellite_composite, physical_available=True,
        external_score=None, external_available=False,
    )
    data_quality = build_data_quality_summary(data_sources=_LIVE_SOURCES, strict_mode=True)
    breakdown = build_score_breakdown(momentum_result=momentum, convergence=convergence, data_quality=data_quality)

    assert convergence.outcome == "SIGNAL_CONFLICT"
    assert breakdown.contradictory_signals is True
    assert breakdown.convergence_outcome == "SIGNAL_CONFLICT"
    assert breakdown.convergence_explanation == convergence.explanation


def test_breakdown_data_confidence_matches_data_quality_completeness():
    financials = _financials()
    vision = _vision()
    partial_sources = dict(_LIVE_SOURCES)
    partial_sources["capex_growth_signal"] = "insufficient_data"
    financials = dict(financials)
    financials["capex_growth_signal"] = FinancialSignal(None, False, "insufficient_data", "no tag", status="INSUFFICIENT_DATA")
    momentum = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=partial_sources)
    convergence = assess_convergence(
        financial_score=momentum.financial_composite, financial_available=True,
        physical_score=momentum.satellite_composite, physical_available=True,
        external_score=None, external_available=False,
    )
    data_quality = build_data_quality_summary(data_sources=partial_sources, strict_mode=True)
    breakdown = build_score_breakdown(momentum_result=momentum, convergence=convergence, data_quality=data_quality)

    assert breakdown.data_confidence_pct == data_quality.completeness_pct
    assert breakdown.data_confidence_pct < 100.0


def test_signal_explanations_cover_all_five_signals_with_real_reasons():
    financials = _financials(
        capex_growth_signal=FinancialSignal(None, False, "insufficient_data", "no usable XBRL tag for this filer", status="INSUFFICIENT_DATA")
    )
    vision = _vision()
    sources = dict(_LIVE_SOURCES)
    sources["capex_growth_signal"] = "insufficient_data"
    momentum = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=sources)
    convergence = assess_convergence(
        financial_score=momentum.financial_composite, financial_available=True,
        physical_score=momentum.satellite_composite, physical_available=True,
        external_score=None, external_available=False,
    )
    data_quality = build_data_quality_summary(data_sources=sources, strict_mode=True)
    breakdown = build_score_breakdown(
        momentum_result=momentum, convergence=convergence, data_quality=data_quality, financial_signals=financials,
    )

    names = {e.signal for e in breakdown.signal_explanations}
    assert names == {"revenue_growth", "rd_investment", "capex_growth", "construction_expansion", "vegetation_clearing"}

    capex = next(e for e in breakdown.signal_explanations if e.signal == "capex_growth")
    assert capex.trusted is False
    assert capex.value_0_100 is None
    assert "no usable XBRL tag" in capex.reason  # the real FinancialSignal.summary, not a generic placeholder

    revenue = next(e for e in breakdown.signal_explanations if e.signal == "revenue_growth")
    assert revenue.trusted is True
    assert revenue.value_0_100 == pytest.approx(95.0)
    assert revenue.source == "sec_edgar_live"


def test_as_dict_shape_is_json_serializable_friendly():
    financials = _financials()
    vision = _vision()
    momentum = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=_LIVE_SOURCES)
    convergence = assess_convergence(
        financial_score=momentum.financial_composite, financial_available=True,
        physical_score=momentum.satellite_composite, physical_available=True,
        external_score=None, external_available=False,
    )
    data_quality = build_data_quality_summary(data_sources=_LIVE_SOURCES, strict_mode=True)
    breakdown = build_score_breakdown(
        momentum_result=momentum, convergence=convergence, data_quality=data_quality, financial_signals=financials,
    )
    d = breakdown.as_dict()
    for key in [
        "final_score", "momentum_level", "financial_contribution", "satellite_contribution", "convergence_bonus",
        "weights", "data_confidence_pct", "contradictory_signals", "convergence_outcome", "convergence_explanation",
        "signal_explanations",
    ]:
        assert key in d
    assert isinstance(d["signal_explanations"], list)
    assert len(d["signal_explanations"]) == 5


def test_insufficient_data_result_produces_a_valid_breakdown_with_none_contributions():
    financials = _financials(
        revenue_growth_signal=FinancialSignal(None, False, "insufficient_data", "s", status="INSUFFICIENT_DATA"),
        rd_investment_signal=FinancialSignal(None, False, "insufficient_data", "s", status="INSUFFICIENT_DATA"),
        capex_growth_signal=FinancialSignal(None, False, "insufficient_data", "s", status="INSUFFICIENT_DATA"),
    )
    vision = _vision()
    sources = {
        "revenue_growth_signal": "insufficient_data", "rd_investment_signal": "insufficient_data",
        "capex_growth_signal": "insufficient_data",
        "satellite_imagery_current": "insufficient_data", "satellite_imagery_prior": "insufficient_data",
    }
    momentum = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=sources)
    convergence = assess_convergence(
        financial_score=None, financial_available=False, physical_score=None, physical_available=False,
        external_score=None, external_available=False,
    )
    data_quality = build_data_quality_summary(data_sources=sources, strict_mode=True)
    breakdown = build_score_breakdown(
        momentum_result=momentum, convergence=convergence, data_quality=data_quality, financial_signals=financials,
    )

    assert breakdown.momentum_level == "INSUFFICIENT_DATA"
    assert breakdown.financial_contribution is None
    assert breakdown.satellite_contribution is None
    assert breakdown.convergence_bonus == 0.0
    assert all(not e.trusted for e in breakdown.signal_explanations)
