"""Regression tests for the "Evidence Completeness" -> "Evidence Coverage"
terminology fix (Issue 2).

Context: the dashboard previously reported "Evidence Completeness: 100%"
on the same assessment view where "External / Public Disclosure:
INSUFFICIENT" and "Public Evidence: N/A — No data source for this
evidence category is currently wired into the pipeline" were also shown.
That 100% figure (``score_breakdown.data_confidence_pct`` /
``data_quality.completeness_pct``) only ever measures the signals this
pipeline currently tracks and wires in (financial + satellite) -- it was
never a claim that the company's overall public-evidence picture is
complete, but the old "Completeness" wording could easily be read that
way by an analyst skimming the dashboard.

These tests are deliberately static/textual (reading the UI source files
rather than driving a live Streamlit/JS runtime) because the goal is
narrowly to lock in the *wording* fix -- the underlying computation in
``core/data_quality.py`` is untouched and already has its own behavioral
test coverage in ``test_data_quality.py`` (see
``test_public_evidence_category_is_honestly_empty_not_fabricated`` and
the ``evidence_coverage_pct`` tests).
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_STREAMLIT_COMPANY_INTELLIGENCE = _REPO_ROOT / "streamlit_app" / "sections" / "company_intelligence.py"
_STREAMLIT_STRATEGY = _REPO_ROOT / "streamlit_app" / "sections" / "strategy.py"
_STREAMLIT_EVIDENCE_QUALITY = _REPO_ROOT / "streamlit_app" / "sections" / "evidence_quality.py"
_DASHBOARD_APP_JS = _REPO_ROOT / "dashboard" / "app.js"
_README = _REPO_ROOT / "README.md"


def _read(path: Path) -> str:
    assert path.exists(), f"expected file not found: {path}"
    return path.read_text(encoding="utf-8")


def test_evidence_completeness_label_no_longer_appears_in_streamlit_company_intelligence():
    text = _read(_STREAMLIT_COMPANY_INTELLIGENCE)
    assert "Evidence Completeness" not in text
    assert "Evidence Coverage" in text


def test_evidence_completeness_label_no_longer_appears_in_dashboard_js():
    text = _read(_DASHBOARD_APP_JS)
    assert "Evidence Completeness" not in text
    # "Evidence Coverage" already existed for evidence_coverage_pct in the
    # Data Quality Center panel; the renamed convergence-tile label must
    # now say the same thing, not the old "Evidence Completeness".
    assert text.count("Evidence Coverage") >= 2


def test_evidence_completeness_label_no_longer_appears_anywhere_in_readme():
    text = _read(_README)
    assert "Evidence Completeness" not in text
    assert "Evidence Coverage" in text


def test_evidence_coverage_caption_clarifies_it_is_not_overall_completeness():
    """The renamed label alone isn't enough -- the accompanying text must
    explicitly rule out the "overall public evidence is complete"
    misreading (the reported ambiguity), in both UI surfaces.
    """
    py_text = _read(_STREAMLIT_COMPANY_INTELLIGENCE)
    assert "not a measure of overall public-evidence completeness" in py_text

    js_text = _read(_DASHBOARD_APP_JS)
    assert "not overall public-evidence completeness" in js_text


def test_external_public_disclosure_wording_is_preserved_unchanged():
    """Issue 2 explicitly requires 'External / Public Disclosure:
    INSUFFICIENT' to stay as-is -- that evidence source genuinely isn't
    wired into the pipeline, so this label must not be renamed or
    softened away.
    """
    text = _read(_STREAMLIT_COMPANY_INTELLIGENCE)
    assert "External / Public Disclosure" in text


def test_data_gaps_and_overall_completeness_wording_scoped_to_tracked_signals():
    """The strategy view's 'No data gaps' / 'Overall completeness' copy
    must no longer read as if the whole intelligence picture (including
    public evidence) has no gaps -- it must be scoped explicitly to the
    financial/satellite signals this pipeline actually tracks.
    """
    text = _read(_STREAMLIT_STRATEGY)
    assert "No data gaps — every signal was live." not in text
    assert "Overall completeness:" not in text
    assert "tracked financial/satellite signals" in text
    assert "external/public disclosure" in text.lower()


def test_overall_rating_caption_scoped_to_tracked_signals_in_evidence_quality_view():
    text = _read(_STREAMLIT_EVIDENCE_QUALITY)
    assert "rates tracked signals only, not the full public-evidence picture" in text


def test_overall_rating_caption_scoped_to_tracked_signals_in_dashboard_js():
    text = _read(_DASHBOARD_APP_JS)
    assert "rates only the tracked signals above, not the full public-evidence picture" in text


def test_deterministic_scoring_module_untouched_by_terminology_fix():
    """Issue 2 explicitly forbids changing the actual deterministic
    scoring -- confirm the data-quality computation module's field names
    (data_confidence_pct's source, completeness_pct, evidence_coverage_pct,
    overall_rating) are all still present and unrenamed; only display
    labels in the UI layers change, never the underlying contract.
    """
    from orbitaliq.core.data_quality import build_data_quality_summary

    summary = build_data_quality_summary(
        data_sources={"revenue_growth_signal": "sec_edgar_live"}, strict_mode=True,
    )
    d = summary.as_dict()
    assert "completeness_pct" in d
    assert "evidence_coverage_pct" in d
    assert "overall_rating" in d
    assert d["completeness_pct"] == 100.0
