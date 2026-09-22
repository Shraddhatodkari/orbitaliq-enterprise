"""Tests for the deterministic grounding critic (core/grounding.py).

Covers the two real bugs found and fixed during development (documented in
the module docstring): the "/100" denominator mis-extraction, and the
over-permissive automatic x100/÷100 rescaling that let a fabricated
percentage slip through by accidentally matching a real ratio.
"""
from __future__ import annotations

from orbitaliq.core.grounding import build_grounded_values, check_narrative_grounding


def test_grounded_narrative_passes():
    # Per the caller-responsibility contract, a ratio-unit metric's raw
    # fraction AND its x100 percent form both need to be in the grounded
    # set for a "28.4%" phrasing to match a 0.284 fraction.
    grounded = build_grounded_values(74.3, [61.2, 55.0, 40.1], 0.284, 0.284 * 100)
    text = (
        "The composite momentum score is 74.3/100, driven by a 61.2 contribution from revenue growth. "
        "Revenue grew 28.4% year over year."
    )
    result = check_narrative_grounding(text, grounded)
    assert result.passed is True
    assert result.status == "GROUNDED"
    assert result.flagged_values == []
    assert result.checked_numbers >= 3


def test_fabricated_number_is_blocked():
    grounded = build_grounded_values(74.3, [61.2, 55.0, 40.1])
    text = "The composite momentum score is 74.3/100, and next-quarter revenue is projected at $999,000,000."
    result = check_narrative_grounding(text, grounded)
    assert result.passed is False
    assert result.status == "BLOCKED_UNGROUNDED_NUMBER"
    assert len(result.flagged_values) == 1
    assert float(result.flagged_values[0]) == 999_000_000.0


def test_narrative_with_no_numeric_claims_passes_trivially():
    grounded = build_grounded_values(74.3)
    result = check_narrative_grounding("Momentum appears strong across the board.", grounded)
    assert result.passed is True
    assert result.status == "GROUNDED_NO_NUMERIC_CLAIMS"
    assert result.checked_numbers == 0


def test_score_slash_100_denominator_is_not_mis_extracted_as_a_claim():
    """Regression test: '35.4/100' must not also register a standalone
    claim for the literal '100', which previously caused an otherwise
    fully-grounded narrative to be incorrectly blocked.
    """
    grounded = build_grounded_values(35.4)
    text = "The composite momentum score is 35.4/100."
    result = check_narrative_grounding(text, grounded)
    assert result.passed is True
    assert result.status == "GROUNDED"
    assert result.checked_numbers == 1  # only 35.4, not 100 as well


def test_fabricated_percent_is_not_accidentally_grounded_by_rescaling():
    """Regression test: a fabricated '47.9%' must not be waved through just
    because 47.9/100 = 0.479 happens to be numerically close to a real,
    unrelated ratio-scale grounded value (e.g. a 0.4413 gross margin).
    """
    grounded = build_grounded_values(0.4413)  # a real gross margin, as a raw fraction
    text = "Gross margin expanded to 47.9% this quarter."
    result = check_narrative_grounding(text, grounded, tolerance=0.6)
    assert result.passed is False
    assert result.status == "BLOCKED_UNGROUNDED_NUMBER"


def test_caller_must_add_both_fraction_and_percent_form_for_ratio_metrics():
    """Documents the caller-responsibility contract: passing both the raw
    fraction and its x100 percent form grounds a percent-phrased claim.
    """
    grounded = build_grounded_values(0.4413, 0.4413 * 100)
    text = "Gross margin was 44.13% this quarter."
    result = check_narrative_grounding(text, grounded)
    assert result.passed is True


def test_build_grounded_values_flattens_groups_and_ignores_none():
    grounded = build_grounded_values(1.0, [2.0, None, 3.5], None, 4)
    assert grounded == {1.0, 2.0, 3.5, 4.0}


def test_large_dollar_figure_uses_relative_not_absolute_tolerance():
    grounded = build_grounded_values(383_285_000_000)
    text = "Cash and equivalents stood at $383,285,000,050."  # trivial rounding difference
    result = check_narrative_grounding(text, grounded)
    assert result.passed is True

    text_far_off = "Cash and equivalents stood at $400,000,000,000."  # materially different
    result_far = check_narrative_grounding(text_far_off, grounded)
    assert result_far.passed is False
