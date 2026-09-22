from __future__ import annotations

from orbitaliq.core.convergence import ConvergenceLevel, ConvergenceOutcome, assess_convergence, classify_level


def test_classify_level_thresholds():
    assert classify_level(80.0, available=True) == ConvergenceLevel.HIGH.value
    assert classify_level(66.0, available=True) == ConvergenceLevel.HIGH.value
    assert classify_level(65.9, available=True) == ConvergenceLevel.MODERATE.value
    assert classify_level(33.0, available=True) == ConvergenceLevel.MODERATE.value
    assert classify_level(32.9, available=True) == ConvergenceLevel.LOW.value
    assert classify_level(0.0, available=True) == ConvergenceLevel.LOW.value


def test_classify_level_insufficient_when_unavailable():
    assert classify_level(90.0, available=False) == ConvergenceLevel.INSUFFICIENT.value
    assert classify_level(None, available=True) == ConvergenceLevel.INSUFFICIENT.value


def test_convergence_both_high_with_corroborating_external():
    result = assess_convergence(
        financial_score=80.0, financial_available=True,
        physical_score=75.0, physical_available=True,
        external_score=70.0, external_available=True,
    )
    assert result.outcome == ConvergenceOutcome.CONVERGE.value
    assert result.financial_level == "HIGH"
    assert result.physical_level == "HIGH"


def test_convergence_both_high_but_no_external_corroboration_is_partial():
    result = assess_convergence(
        financial_score=80.0, financial_available=True,
        physical_score=75.0, physical_available=True,
        external_score=None, external_available=False,
    )
    assert result.outcome == ConvergenceOutcome.PARTIAL.value


def test_convergence_conflict_when_high_and_low():
    result = assess_convergence(
        financial_score=90.0, financial_available=True,
        physical_score=10.0, physical_available=True,
        external_score=None, external_available=False,
    )
    assert result.outcome == ConvergenceOutcome.CONFLICT.value


def test_convergence_insufficient_when_a_primary_group_is_unavailable():
    result = assess_convergence(
        financial_score=90.0, financial_available=True,
        physical_score=None, physical_available=False,
        external_score=None, external_available=False,
    )
    assert result.outcome == ConvergenceOutcome.INSUFFICIENT.value


def test_convergence_moderate_moderate_is_partial_not_converge():
    result = assess_convergence(
        financial_score=50.0, financial_available=True,
        physical_score=45.0, physical_available=True,
        external_score=None, external_available=False,
    )
    assert result.outcome == ConvergenceOutcome.PARTIAL.value
    assert result.outcome != ConvergenceOutcome.CONVERGE.value


def test_as_dict_shape():
    result = assess_convergence(
        financial_score=80.0, financial_available=True,
        physical_score=80.0, physical_available=True,
        external_score=80.0, external_available=True,
    )
    d = result.as_dict()
    assert set(d.keys()) == {"financial_level", "physical_level", "external_level", "outcome", "explanation"}
