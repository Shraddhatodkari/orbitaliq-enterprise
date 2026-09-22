from __future__ import annotations

from orbitaliq.core.data_quality import NOT_AVAILABLE, build_data_quality_summary


def test_all_live_sources_are_fully_complete():
    summary = build_data_quality_summary(
        data_sources={
            "revenue_growth_signal": "sec_edgar_live",
            "rd_investment_signal": "sec_edgar_live",
            "satellite_imagery_current": "nasa_gibs_live",
        },
        strict_mode=False,
    )
    assert summary.completeness_pct == 100.0
    assert summary.live_signals == 3
    assert summary.fallback_signals == 0
    assert summary.unavailable_signals == 0
    assert summary.source_count == 2  # sec_edgar + nasa_gibs, distinct systems
    assert summary.missing_fields == []


def test_fallback_sources_are_classified_and_counted():
    summary = build_data_quality_summary(
        data_sources={
            "revenue_growth_signal": "synthetic_fallback:revenue_growth_signal",
            "capex_growth_signal": "offline_demo_mode",
        },
        strict_mode=False,
    )
    assert summary.live_signals == 0
    assert summary.fallback_signals == 2
    assert summary.completeness_pct == 0.0
    assert set(summary.fallback_fields) == {"revenue_growth_signal", "capex_growth_signal"}
    assert set(summary.missing_fields) == {"revenue_growth_signal", "capex_growth_signal"}


def test_insufficient_data_sources_are_classified_unavailable_not_fallback():
    """A live-mode SEC EDGAR/NASA GIBS failure (source="insufficient_data",
    see data/sec_edgar_client.py::_insufficient_financial_signal and
    data/imagery_provider.py's typed live-failure result) must bucket as
    "unavailable" -- the same bucket as NOT_AVAILABLE -- never as
    "fallback", which is reserved for the disclosed, test/demo-only
    synthetic path (synthetic_fallback:*, offline_demo_mode). Conflating
    the two would make a genuine live production failure indistinguishable
    from a deliberate offline demo run.
    """
    summary = build_data_quality_summary(
        data_sources={
            "revenue_growth_signal": "insufficient_data",
            "satellite_imagery_current": "insufficient_data",
        },
        strict_mode=True,
    )
    assert summary.unavailable_signals == 2
    assert summary.fallback_signals == 0
    assert set(summary.unavailable_fields) == {"revenue_growth_signal", "satellite_imagery_current"}


def test_not_available_sources_are_classified_unavailable_not_fallback():
    summary = build_data_quality_summary(
        data_sources={"revenue_growth_signal": NOT_AVAILABLE},
        strict_mode=True,
    )
    assert summary.unavailable_signals == 1
    assert summary.fallback_signals == 0
    assert summary.unavailable_fields == ["revenue_growth_signal"]
    assert summary.strict_mode is True


def test_mixed_provenance_completeness_math():
    summary = build_data_quality_summary(
        data_sources={
            "a": "sec_edgar_live",
            "b": "sec_edgar_live",
            "c": "synthetic_fallback:c",
            "d": NOT_AVAILABLE,
        },
        strict_mode=False,
    )
    assert summary.completeness_pct == 50.0
    assert summary.live_signals == 2
    assert summary.fallback_signals == 1
    assert summary.unavailable_signals == 1


def test_empty_data_sources_does_not_divide_by_zero():
    summary = build_data_quality_summary(data_sources={}, strict_mode=False)
    assert summary.completeness_pct == 0.0
    assert summary.total_signals == 0


def test_unrecognized_source_label_defaults_to_unavailable_never_silently_trusted():
    """Mirrors core/data_quality_state.py's own conservative default for an
    unrecognized label (see test_data_quality_state.py) -- a source string
    that matches none of the known live/fallback/unavailable prefixes must
    still classify as unavailable, never as a bucket that would count it as
    complete or covered.
    """
    summary = build_data_quality_summary(
        data_sources={"revenue_growth_signal": "some_unrecognized_future_source"}, strict_mode=False,
    )
    assert summary.live_signals == 0
    assert summary.fallback_signals == 0
    assert summary.unavailable_signals == 1
    assert summary.completeness_pct == 0.0


def test_freshness_and_provenance_are_passed_through():
    summary = build_data_quality_summary(
        data_sources={"revenue_growth_signal": "sec_edgar_live"},
        strict_mode=False,
        fiscal_period="FY2024",
        satellite_current_date="2026-09-01",
        satellite_prior_date="2026-03-01",
        assessed_at="2026-09-18T00:00:00Z",
    )
    assert summary.freshness == {
        "most_recent_fiscal_period": "FY2024",
        "satellite_current_observation_date": "2026-09-01",
        "satellite_prior_observation_date": "2026-03-01",
        "assessed_at": "2026-09-18T00:00:00Z",
    }
    assert summary.provenance == {"revenue_growth_signal": "sec_edgar_live"}


def test_as_dict_shape():
    summary = build_data_quality_summary(data_sources={"a": "sec_edgar_live"}, strict_mode=False)
    d = summary.as_dict()
    assert set(d.keys()) == {
        "strict_mode", "total_signals", "live_signals", "fallback_signals", "unavailable_signals",
        "completeness_pct", "source_count", "missing_fields", "fallback_fields", "unavailable_fields",
        "freshness", "provenance",
        # Data Quality Center elevation (additive) -- see core/data_quality.py.
        "categories", "freshness_days", "evidence_coverage_pct", "overall_rating",
    }


# --- Data Quality Center elevation: per-category, freshness-in-days, overall rating ---


_FULL_LIVE_SOURCES = {
    "revenue_growth_signal": "sec_edgar_live",
    "rd_investment_signal": "sec_edgar_live",
    "capex_growth_signal": "sec_edgar_live",
    "satellite_imagery_current": "nasa_gibs_live",
    "satellite_imagery_prior": "nasa_gibs_live",
}


def test_categories_present_for_financial_satellite_and_public_evidence():
    summary = build_data_quality_summary(data_sources=_FULL_LIVE_SOURCES, strict_mode=True)
    assert set(summary.categories.keys()) == {"financial", "satellite", "public_evidence"}


def test_financial_category_is_fully_complete_when_all_three_signals_are_live():
    summary = build_data_quality_summary(data_sources=_FULL_LIVE_SOURCES, strict_mode=True)
    fin = summary.categories["financial"]
    assert fin["total_signals"] == 3
    assert fin["live_signals"] == 3
    assert fin["completeness_pct"] == 100.0


def test_satellite_category_reflects_partial_coverage():
    sources = dict(_FULL_LIVE_SOURCES)
    sources["satellite_imagery_prior"] = "insufficient_data"
    summary = build_data_quality_summary(data_sources=sources, strict_mode=True)
    sat = summary.categories["satellite"]
    assert sat["total_signals"] == 2
    assert sat["live_signals"] == 1
    assert sat["completeness_pct"] == 50.0
    assert "satellite_imagery_prior" in sat["fields"]


def test_public_evidence_category_is_honestly_empty_not_fabricated():
    """No public-disclosure data source is wired in yet (matches
    core/convergence.py's external_level, which is always INSUFFICIENT) --
    this category must report itself as empty with an explanatory note,
    never silently report 100% (0/0) or omit itself.
    """
    summary = build_data_quality_summary(data_sources=_FULL_LIVE_SOURCES, strict_mode=True)
    public = summary.categories["public_evidence"]
    assert public["total_signals"] == 0
    assert public["completeness_pct"] == 0.0
    assert public["note"] is not None


def test_cached_real_and_derived_sources_count_as_live_in_categories():
    sources = dict(_FULL_LIVE_SOURCES)
    sources["capex_growth_signal"] = "cached_real:sec_edgar:2026-01-01T00:00:00+00:00"
    summary = build_data_quality_summary(data_sources=sources, strict_mode=True)
    assert summary.categories["financial"]["live_signals"] == 3
    assert summary.completeness_pct == 100.0


def test_evidence_coverage_counts_fallback_as_covered_but_not_complete():
    sources = dict(_FULL_LIVE_SOURCES)
    sources["capex_growth_signal"] = "synthetic_fallback:capex_growth_signal"  # disclosed, not live
    summary = build_data_quality_summary(data_sources=sources, strict_mode=True)
    assert summary.completeness_pct < 100.0  # one signal isn't live
    assert summary.evidence_coverage_pct == 100.0  # but every signal has SOME grounded number


def test_evidence_coverage_drops_for_a_genuinely_unavailable_signal():
    sources = dict(_FULL_LIVE_SOURCES)
    sources["capex_growth_signal"] = "insufficient_data"  # a real gap, not a disclosed fallback
    summary = build_data_quality_summary(data_sources=sources, strict_mode=True)
    assert summary.evidence_coverage_pct < 100.0


def test_freshness_days_computed_for_satellite_dates():
    from datetime import datetime, timedelta, timezone

    ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
    summary = build_data_quality_summary(
        data_sources=_FULL_LIVE_SOURCES, strict_mode=True, satellite_current_date=ten_days_ago,
    )
    assert summary.freshness_days["satellite_current_days_ago"] in (9, 10, 11)  # tolerate a day of drift


def test_freshness_days_is_none_for_a_non_date_fiscal_period():
    summary = build_data_quality_summary(
        data_sources=_FULL_LIVE_SOURCES, strict_mode=True, fiscal_period="FY2023",
    )
    # fiscal_period isn't tracked in freshness_days at all (it isn't a
    # calendar date) -- confirm no crash and no fabricated number appears.
    assert "most_recent_fiscal_period_days_ago" not in summary.freshness_days


def test_freshness_days_handles_missing_dates_gracefully():
    summary = build_data_quality_summary(data_sources=_FULL_LIVE_SOURCES, strict_mode=True)
    assert summary.freshness_days["satellite_current_days_ago"] is None
    assert summary.freshness_days["satellite_prior_days_ago"] is None
    assert summary.freshness_days["assessed_at_days_ago"] is None


def test_freshness_days_returns_none_for_a_genuinely_unparseable_date_string():
    """A string that isn't a valid date OR a valid datetime (unlike
    "FY2023" above, which is simply never passed to _days_ago at all) must
    still exercise _days_ago's dual-parse fallback -- datetime.fromisoformat
    fails, then date.fromisoformat also fails -- and come back None, never
    raise and never fabricate a number.
    """
    summary = build_data_quality_summary(
        data_sources=_FULL_LIVE_SOURCES, strict_mode=True, satellite_current_date="not-a-real-date",
    )
    assert summary.freshness_days["satellite_current_days_ago"] is None


def test_overall_rating_high_when_fully_live():
    summary = build_data_quality_summary(data_sources=_FULL_LIVE_SOURCES, strict_mode=True)
    assert summary.overall_rating == "HIGH"


def test_overall_rating_insufficient_when_no_signals_at_all():
    summary = build_data_quality_summary(data_sources={}, strict_mode=True)
    assert summary.overall_rating == "INSUFFICIENT"


def test_overall_rating_low_when_mostly_unavailable():
    sources = {k: NOT_AVAILABLE for k in _FULL_LIVE_SOURCES}
    summary = build_data_quality_summary(data_sources=sources, strict_mode=True)
    assert summary.overall_rating == "LOW"
