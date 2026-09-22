"""Tests for the canonical 5-state data-quality vocabulary
(core/data_quality_state.py) -- the single source of truth that
core/data_quality.py's bucketing and core/intelligence_engine.py's trust
gate both delegate to (directly or indirectly).
"""
from __future__ import annotations

from orbitaliq.core.data_quality_state import (
    DataQualityState,
    classify_state,
    is_demo_fallback,
    is_trusted,
)


def test_none_source_classifies_as_live_and_trusted():
    assert classify_state(None) is DataQualityState.LIVE
    assert is_trusted(None) is True


def test_live_labels_classify_as_live():
    assert classify_state("sec_edgar_live") is DataQualityState.LIVE
    assert classify_state("nasa_gibs_live") is DataQualityState.LIVE
    assert is_trusted("sec_edgar_live") is True


def test_cached_real_label_classifies_as_cached_real_and_trusted():
    assert classify_state("cached_real:sec_edgar:2026-01-01T00:00:00+00:00") is DataQualityState.CACHED_REAL
    assert is_trusted("cached_real:sec_edgar:2026-01-01T00:00:00+00:00") is True


def test_derived_label_classifies_as_derived_and_trusted():
    assert classify_state("derived:sec_edgar") is DataQualityState.DERIVED
    assert is_trusted("derived:sec_edgar") is True


def test_error_label_classifies_as_error_and_untrusted():
    assert classify_state("error:sec_edgar:ConnectError") is DataQualityState.ERROR
    assert is_trusted("error:sec_edgar:ConnectError") is False


def test_insufficient_data_labels_classify_as_insufficient_and_untrusted():
    assert classify_state("insufficient_data") is DataQualityState.INSUFFICIENT_DATA
    assert classify_state("insufficient_data:some reason") is DataQualityState.INSUFFICIENT_DATA
    assert classify_state("NOT_AVAILABLE") is DataQualityState.INSUFFICIENT_DATA
    assert is_trusted("NOT_AVAILABLE") is False


def test_unrecognized_label_defaults_to_insufficient_data_never_silently_trusted():
    assert classify_state("some_unknown_label") is DataQualityState.INSUFFICIENT_DATA
    assert is_trusted("some_unknown_label") is False


def test_offline_demo_mode_is_demo_fallback_and_trusted():
    """offline_demo_mode is the deliberate, fully-disclosed whole-pipeline
    stand-in (see agents/ingestion_agent.py::OFFLINE_SOURCE_LABELS) --
    trusted for scoring, distinct from a live-mode failure.
    """
    assert is_demo_fallback("offline_demo_mode") is True
    assert classify_state("offline_demo_mode") is DataQualityState.LIVE
    assert is_trusted("offline_demo_mode") is True


def test_synthetic_fallback_is_demo_fallback_but_not_trusted():
    """synthetic_fallback:* is a disclosed demo/test-fixture label, kept
    recognized for backward compatibility, but -- unlike offline_demo_mode
    -- is NOT trusted for scoring (see core/intelligence_engine.py tests
    exercising the exclusion path with this exact label).
    """
    assert is_demo_fallback("synthetic_fallback:revenue_growth_signal") is True
    assert classify_state("synthetic_fallback:revenue_growth_signal") is DataQualityState.INSUFFICIENT_DATA
    assert is_trusted("synthetic_fallback:revenue_growth_signal") is False


def test_is_demo_fallback_false_for_none_and_production_labels():
    assert is_demo_fallback(None) is False
    assert is_demo_fallback("sec_edgar_live") is False
    assert is_demo_fallback("error:sec_edgar:ConnectError") is False
