import pytest

from orbitaliq.core.intelligence_engine import (
    FINANCIAL_WEIGHT,
    SATELLITE_WEIGHT,
    MomentumLevel,
    classify_momentum_level,
    compute_momentum_assessment,
)
from orbitaliq.data.sec_edgar_client import FinancialSignal
from orbitaliq.nvidia.vision_model import VisionFindings


def _financials(**overrides) -> dict[str, FinancialSignal]:
    defaults = dict(
        revenue_growth_signal=FinancialSignal(0.1, True, "sec_edgar_live", "s"),
        rd_investment_signal=FinancialSignal(0.1, True, "sec_edgar_live", "s"),
        capex_growth_signal=FinancialSignal(0.1, True, "sec_edgar_live", "s"),
    )
    defaults.update(overrides)
    return defaults


def _vision(**overrides) -> VisionFindings:
    defaults = dict(
        construction_expansion_signal=0.1,
        vegetation_clearing_signal=0.1,
        overall_change_score=0.1,
        change_energy=0.1,
        device_used="cpu",
        model_finetuned=False,
    )
    defaults.update(overrides)
    return VisionFindings(**defaults)


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, MomentumLevel.STABLE),
        (29.9, MomentumLevel.STABLE),
        (30, MomentumLevel.EMERGING),
        (54.9, MomentumLevel.EMERGING),
        (55, MomentumLevel.STRONG),
        (74.9, MomentumLevel.STRONG),
        (75, MomentumLevel.AGGRESSIVE_EXPANSION),
        (100, MomentumLevel.AGGRESSIVE_EXPANSION),
    ],
)
def test_classify_momentum_level_thresholds(score, expected):
    assert classify_momentum_level(score) == expected


def test_low_signals_score_stable():
    result = compute_momentum_assessment(financial_signals=_financials(), vision_findings=_vision())
    assert result.momentum_score < 30
    assert result.momentum_level == MomentumLevel.STABLE


def test_strong_convergent_signals_score_aggressive_expansion():
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.95, True, "sec_edgar_live", "s"),
        rd_investment_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
        capex_growth_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
    )
    vision = _vision(construction_expansion_signal=0.9, vegetation_clearing_signal=0.85, overall_change_score=0.9)
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision)
    assert result.momentum_score >= 75
    assert result.momentum_level == MomentumLevel.AGGRESSIVE_EXPANSION
    assert "construction_expansion" in result.signal_breakdown
    assert result.signal_breakdown["revenue_growth"] == pytest.approx(95.0)


def test_result_exposes_the_real_composite_terms_for_explainability():
    """financial_composite/satellite_composite/convergence_bonus must be
    the exact values the formula actually used -- not recomputed
    elsewhere -- so core/score_explainability.py can build an accurate
    breakdown from them.
    """
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.95, True, "sec_edgar_live", "s"),
        rd_investment_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
        capex_growth_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
    )
    vision = _vision(construction_expansion_signal=0.9, vegetation_clearing_signal=0.85, overall_change_score=0.9)
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision)

    expected_financial = (95.0 + 90.0 + 90.0) / 3
    expected_satellite = (90.0 + 85.0) / 2
    assert result.financial_composite == pytest.approx(expected_financial)
    assert result.satellite_composite == pytest.approx(expected_satellite)
    assert result.convergence_bonus > 0.0  # both composites clear the threshold
    # The formula's own weights, reapplied, reproduce the final score exactly.
    reconstructed = min(
        100.0, FINANCIAL_WEIGHT * result.financial_composite + SATELLITE_WEIGHT * result.satellite_composite
        + result.convergence_bonus,
    )
    assert result.momentum_score == pytest.approx(reconstructed)


def test_insufficient_data_result_has_none_composites_and_zero_bonus():
    financials = _financials(
        revenue_growth_signal=FinancialSignal(None, False, "insufficient_data", "s", status="INSUFFICIENT_DATA"),
        rd_investment_signal=FinancialSignal(None, False, "insufficient_data", "s", status="INSUFFICIENT_DATA"),
        capex_growth_signal=FinancialSignal(None, False, "insufficient_data", "s", status="INSUFFICIENT_DATA"),
    )
    data_sources = {
        "revenue_growth_signal": "insufficient_data", "rd_investment_signal": "insufficient_data",
        "capex_growth_signal": "insufficient_data",
        "satellite_imagery_current": "insufficient_data", "satellite_imagery_prior": "insufficient_data",
    }
    vision = _vision(construction_expansion_signal=0.5, vegetation_clearing_signal=0.5, overall_change_score=0.5)
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert result.momentum_level == MomentumLevel.INSUFFICIENT_DATA
    assert result.financial_composite is None
    assert result.satellite_composite is None
    assert result.convergence_bonus == 0.0


def test_satellite_excluded_result_has_none_satellite_composite():
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
        rd_investment_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
        capex_growth_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"),
    )
    data_sources = {
        "revenue_growth_signal": "sec_edgar_live", "rd_investment_signal": "sec_edgar_live",
        "capex_growth_signal": "sec_edgar_live",
        "satellite_imagery_current": "insufficient_data", "satellite_imagery_prior": "insufficient_data",
    }
    vision = _vision(construction_expansion_signal=0.9, vegetation_clearing_signal=0.9, overall_change_score=0.9)
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert result.satellite_composite is None
    assert result.financial_composite == pytest.approx(90.0)
    assert result.convergence_bonus == 0.0  # can't corroborate with a fully-excluded side
    assert result.momentum_score == pytest.approx(90.0)  # 100% reweighted to financial


def test_score_bounded_0_100():
    financials = _financials(
        revenue_growth_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
        rd_investment_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
        capex_growth_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
    )
    vision = _vision(construction_expansion_signal=1.0, vegetation_clearing_signal=1.0, overall_change_score=1.0)
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision)
    assert 0.0 <= result.momentum_score <= 100.0


def test_convergence_bonus_requires_both_composites_above_threshold():
    # Strong financials alone (no satellite corroboration) should NOT get
    # the convergence bonus -- only the base weighted blend.
    financials = _financials(
        revenue_growth_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
        rd_investment_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
        capex_growth_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
    )
    weak_satellite = compute_momentum_assessment(financial_signals=financials, vision_findings=_vision())
    strong_satellite = compute_momentum_assessment(
        financial_signals=financials,
        vision_findings=_vision(construction_expansion_signal=0.9, vegetation_clearing_signal=0.9),
    )
    assert strong_satellite.momentum_score > weak_satellite.momentum_score


def test_recommends_action_for_dominant_signal():
    financials = _financials(capex_growth_signal=FinancialSignal(0.9, True, "sec_edgar_live", "s"))
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=_vision())
    assert any("capex" in a.lower() or "hiring" in a.lower() for a in result.recommended_actions)


def test_aggressive_expansion_escalates_to_engagement_lead():
    financials = _financials(
        revenue_growth_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
        rd_investment_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
        capex_growth_signal=FinancialSignal(1.0, True, "sec_edgar_live", "s"),
    )
    vision = _vision(construction_expansion_signal=1.0, vegetation_clearing_signal=1.0, overall_change_score=1.0)
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision)
    assert result.momentum_level == MomentumLevel.AGGRESSIVE_EXPANSION
    assert any("engagement lead" in a.lower() for a in result.recommended_actions)


def test_stable_result_has_reassurance_action():
    result = compute_momentum_assessment(financial_signals=_financials(), vision_findings=_vision())
    assert len(result.recommended_actions) >= 1


# --- Live-mode data-sufficiency gate (the NON-NEGOTIABLE "never manufacture
# a score from fabricated data" requirement). ``synthetic_fallback:*`` is
# the label the real ``SecEdgarClient``/``NasaGibsImageryProvider`` use for
# a live-mode fetch that genuinely failed and fell back to a synthetic
# placeholder value (see data/sec_edgar_client.py, data/imagery_provider.py)
# -- a signal marked with that source must never be blended into the score
# as if it were real. ---

_LIVE = "sec_edgar_live"
_FALLBACK = "synthetic_fallback:revenue_growth"
_GIBS_LIVE = "nasa_gibs_live"
_GIBS_FALLBACK = "synthetic_fallback:satellite_tile"


def test_all_signals_synthetic_fallback_returns_insufficient_data_not_a_score():
    """Every live source failed -- there is nothing legitimate to score.
    Must report that plainly, never a number derived from fallback data.
    """
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.9, False, _FALLBACK, "fallback"),
        rd_investment_signal=FinancialSignal(0.9, False, _FALLBACK, "fallback"),
        capex_growth_signal=FinancialSignal(0.9, False, _FALLBACK, "fallback"),
    )
    vision = _vision(construction_expansion_signal=0.9, vegetation_clearing_signal=0.9)
    data_sources = {
        "revenue_growth_signal": _FALLBACK,
        "rd_investment_signal": _FALLBACK,
        "capex_growth_signal": _FALLBACK,
        "satellite_imagery_current": _GIBS_FALLBACK,
        "satellite_imagery_prior": _GIBS_FALLBACK,
    }
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert result.momentum_level == MomentumLevel.INSUFFICIENT_DATA
    assert result.signal_breakdown == {}
    assert set(result.excluded_signals) == {
        "revenue_growth",
        "rd_investment",
        "capex_growth",
        "construction_expansion",
        "vegetation_clearing",
    }
    # Even though the underlying (unused) financial values were 0.9 --
    # i.e. would have scored ~90/100 if naively blended in -- the reported
    # score must not reflect that fabricated input in any way.
    assert result.momentum_score == 0.0


def test_one_fallback_financial_signal_is_excluded_and_others_renormalized():
    """Two of three financial signals are genuinely live; the third fell
    back. The score must come from only the two live ones, not from all
    three as if the fallback were real.
    """
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.8, True, _LIVE, "s"),
        rd_investment_signal=FinancialSignal(0.8, True, _LIVE, "s"),
        capex_growth_signal=FinancialSignal(0.1, False, _FALLBACK, "fallback"),
    )
    vision = _vision(construction_expansion_signal=0.1, vegetation_clearing_signal=0.1)
    data_sources = {
        "revenue_growth_signal": _LIVE,
        "rd_investment_signal": _LIVE,
        "capex_growth_signal": _FALLBACK,
        "satellite_imagery_current": _GIBS_LIVE,
        "satellite_imagery_prior": _GIBS_LIVE,
    }
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert result.excluded_signals == ["capex_growth"]
    assert "capex_growth" not in result.signal_breakdown
    assert result.signal_breakdown["revenue_growth"] == pytest.approx(80.0)
    # financial_composite should be the average of ONLY the two live
    # signals (80, 80) = 80, not (80+80+10)/3 = 56.67 -- proves the
    # fallback capex value never entered the arithmetic at all.
    expected_financial_composite = 80.0
    expected_satellite_composite = 10.0
    expected_base = 0.6 * expected_financial_composite + 0.4 * expected_satellite_composite
    assert result.momentum_score == pytest.approx(expected_base, abs=0.5)


def test_fully_live_data_sources_reproduces_the_legacy_no_data_sources_score_exactly():
    """When every signal is live, passing data_sources must change nothing
    about the computed score -- proves the sufficiency gate is a pure
    exclusion mechanism, not a different formula.
    """
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.3, True, _LIVE, "s"),
        rd_investment_signal=FinancialSignal(0.5, True, _LIVE, "s"),
        capex_growth_signal=FinancialSignal(0.2, True, _LIVE, "s"),
    )
    vision = _vision(construction_expansion_signal=0.4, vegetation_clearing_signal=0.35)
    data_sources = {
        "revenue_growth_signal": _LIVE,
        "rd_investment_signal": _LIVE,
        "capex_growth_signal": _LIVE,
        "satellite_imagery_current": _GIBS_LIVE,
        "satellite_imagery_prior": _GIBS_LIVE,
    }
    without = compute_momentum_assessment(financial_signals=financials, vision_findings=vision)
    with_sources = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert with_sources.momentum_score == pytest.approx(without.momentum_score)
    assert with_sources.signal_breakdown == without.signal_breakdown
    assert with_sources.excluded_signals == []


def test_offline_demo_mode_source_is_trusted_not_treated_as_a_live_mode_failure():
    """``offline_demo_mode`` is a deliberate, disclosed non-live testing
    mode -- not a live-mode fetch that failed -- so it must be included in
    the score exactly like a live signal, not excluded.
    """
    financials = _financials()  # all sec_edgar_live per the fixture default
    vision = _vision()
    data_sources = {k: "offline_demo_mode" for k in ("revenue_growth_signal", "rd_investment_signal", "capex_growth_signal")}
    data_sources["satellite_imagery_current"] = "offline_demo_mode"
    data_sources["satellite_imagery_prior"] = "offline_demo_mode"
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert result.excluded_signals == []
    assert result.momentum_level != MomentumLevel.INSUFFICIENT_DATA


def test_unavailable_source_is_excluded_like_a_fallback():
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.0, False, "insufficient_data", "no data"),
    )
    vision = _vision()
    data_sources = {
        "revenue_growth_signal": "NOT_AVAILABLE",
        "rd_investment_signal": _LIVE,
        "capex_growth_signal": _LIVE,
        "satellite_imagery_current": _GIBS_LIVE,
        "satellite_imagery_prior": _GIBS_LIVE,
    }
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert "revenue_growth" in result.excluded_signals
    assert "revenue_growth" not in result.signal_breakdown


def test_none_valued_signal_is_never_multiplied_or_treated_as_zero_even_without_data_sources():
    """The escalated requirement: a FinancialSignal with value=None (a
    typed INSUFFICIENT_DATA result from a live-mode SEC EDGAR failure --
    see data/sec_edgar_client.py::_insufficient_financial_signal) must
    never reach the `100 * value` multiply, and must never be silently
    treated as 0. This must hold even when ``data_sources`` isn't passed
    at all (where _is_trusted_source(None) alone would otherwise trust
    every signal) -- the None-value guard is a second, independent line of
    defense, not just the data_sources exclusion.
    """
    financials = _financials(
        revenue_growth_signal=FinancialSignal(value=None, live=False, source="insufficient_data", summary="no data", status="INSUFFICIENT_DATA"),
    )
    vision = _vision()
    # No data_sources passed at all.
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision)
    assert "revenue_growth" not in result.signal_breakdown
    assert "revenue_growth" in result.excluded_signals
    assert result.momentum_level != MomentumLevel.INSUFFICIENT_DATA  # rd/capex are still live and trusted


def test_all_financial_signals_none_and_satellite_excluded_returns_insufficient_data():
    """Every financial signal is a typed INSUFFICIENT_DATA (value=None)
    result and satellite is also excluded -- there is nothing left to
    legitimately score, so this must return INSUFFICIENT_DATA, not crash
    with a TypeError multiplying None, and not silently score 0 as if
    that were a real STABLE result.
    """
    financials = _financials(
        revenue_growth_signal=FinancialSignal(value=None, live=False, source="insufficient_data", summary="x", status="INSUFFICIENT_DATA"),
        rd_investment_signal=FinancialSignal(value=None, live=False, source="insufficient_data", summary="x", status="INSUFFICIENT_DATA"),
        capex_growth_signal=FinancialSignal(value=None, live=False, source="insufficient_data", summary="x", status="INSUFFICIENT_DATA"),
    )
    vision = _vision()
    data_sources = {
        "revenue_growth_signal": "insufficient_data",
        "rd_investment_signal": "insufficient_data",
        "capex_growth_signal": "insufficient_data",
        "satellite_imagery_current": "insufficient_data",
        "satellite_imagery_prior": "insufficient_data",
    }
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert result.momentum_level == MomentumLevel.INSUFFICIENT_DATA
    assert result.momentum_score == 0.0
    assert result.signal_breakdown == {}


def test_satellite_fully_excluded_reweights_to_100pct_financial():
    financials = _financials(
        revenue_growth_signal=FinancialSignal(0.6, True, _LIVE, "s"),
        rd_investment_signal=FinancialSignal(0.6, True, _LIVE, "s"),
        capex_growth_signal=FinancialSignal(0.6, True, _LIVE, "s"),
    )
    vision = _vision(construction_expansion_signal=0.99, vegetation_clearing_signal=0.99)  # would-be value if wrongly included
    data_sources = {
        "revenue_growth_signal": _LIVE,
        "rd_investment_signal": _LIVE,
        "capex_growth_signal": _LIVE,
        "satellite_imagery_current": _GIBS_FALLBACK,
        "satellite_imagery_prior": _GIBS_FALLBACK,
    }
    result = compute_momentum_assessment(financial_signals=financials, vision_findings=vision, data_sources=data_sources)
    assert set(result.excluded_signals) == {"construction_expansion", "vegetation_clearing"}
    # Base score should be exactly the financial composite (60), not a
    # blend that includes the excluded 99-valued satellite signals.
    assert result.momentum_score == pytest.approx(60.0, abs=0.01)
