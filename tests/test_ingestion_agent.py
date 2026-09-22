"""Tests for IngestionAgent's wiring of the live/offline data sources.

The individual API clients (SEC EDGAR, NASA GIBS) each have their own
dedicated test module; here we verify IngestionAgent correctly assembles
their results into financial signals + a tile pair + a provenance map,
using lightweight fakes so this stays fast and independent of any HTTP
mocking.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbitaliq.agents.ingestion_agent import IngestionAgent
from orbitaliq.agents.types import CompanyInput
from orbitaliq.config import Settings
from orbitaliq.data.imagery_provider import ChangeTilePairResult, ImageryResult
from orbitaliq.data.sec_edgar_client import FinancialSignal


class _FakeSecClient:
    def __init__(self, signals: dict[str, FinancialSignal], profile=None):
        self._signals = signals
        self._profile = profile
        self.closed = False
        self.profile_calls = 0

    def fetch_financial_signals(self, ticker):
        return self._signals

    def fetch_financial_profile(self, ticker):
        self.profile_calls += 1
        if self._profile is None:
            raise RuntimeError("no profile configured for this fake")
        return self._profile

    def close(self):
        self.closed = True


class _FakeImagery:
    def __init__(self, pair: ChangeTilePairResult):
        self._pair = pair
        self.closed = False

    def fetch_change_pair(self, lat, lon):
        return self._pair

    def close(self):
        self.closed = True


def _pair(current_live=True, prior_live=True) -> ChangeTilePairResult:
    current = ImageryResult(
        tile=np.full((3, 64, 64), 0.7, dtype=np.float32),
        live=current_live,
        source="nasa_gibs_live" if current_live else "synthetic_fallback:satellite_tile",
        summary="current tile",
    )
    prior = ImageryResult(
        tile=np.full((3, 64, 64), 0.3, dtype=np.float32),
        live=prior_live,
        source="nasa_gibs_live" if prior_live else "synthetic_fallback:satellite_tile",
        summary="prior tile",
    )
    return ChangeTilePairResult(current=current, prior=prior, lookback_days=180)


def _agent(live: bool, **overrides):
    settings = Settings(ORBITALIQ_LIVE_DATA_MODE=live)
    defaults = dict(
        sec_client=_FakeSecClient(
            {
                "revenue_growth_signal": FinancialSignal(0.6, True, "sec_edgar_live", "rev"),
                "rd_investment_signal": FinancialSignal(0.5, True, "sec_edgar_live", "rd"),
                "capex_growth_signal": FinancialSignal(0.0, False, "synthetic_fallback:capex_growth_signal", "no tag"),
            }
        ),
        imagery_provider=_FakeImagery(_pair()),
    )
    defaults.update(overrides)
    return IngestionAgent(settings=settings, **defaults)


@pytest.fixture
def company():
    return CompanyInput(
        ticker="TSLA", company_name="Tesla, Inc.", facility_name="Gigafactory Nevada", latitude=39.53, longitude=-119.44
    )


def test_offline_mode_never_touches_injected_live_clients(company):
    agent = _agent(live=False)
    financials, current_tile, prior_tile, sources = agent.run(company)

    assert current_tile.shape == (3, 64, 64)
    assert prior_tile.shape == (3, 64, 64)
    assert not np.array_equal(current_tile, prior_tile)
    assert all(v == "offline_demo_mode" for v in sources.values())
    assert isinstance(financials["revenue_growth_signal"].value, float)


def test_live_mode_assembles_financial_signals_and_tile_pair(company):
    agent = _agent(live=True)
    financials, current_tile, prior_tile, sources = agent.run(company)

    assert financials["revenue_growth_signal"].value == 0.6
    assert financials["rd_investment_signal"].value == 0.5
    assert financials["capex_growth_signal"].live is False
    np.testing.assert_array_equal(current_tile, np.full((3, 64, 64), 0.7, dtype=np.float32))
    np.testing.assert_array_equal(prior_tile, np.full((3, 64, 64), 0.3, dtype=np.float32))

    assert sources["revenue_growth_signal"] == "sec_edgar_live"
    assert sources["rd_investment_signal"] == "sec_edgar_live"
    assert sources["capex_growth_signal"] == "synthetic_fallback:capex_growth_signal"
    assert sources["satellite_imagery_current"] == "nasa_gibs_live"
    assert sources["satellite_imagery_prior"] == "nasa_gibs_live"


def test_live_mode_mixed_provenance_is_fully_disclosed(company):
    """Even when some sources are live and others fell back, every single
    signal's provenance is reported — nothing is silently presented as real."""
    agent = _agent(live=True)
    _, _, _, sources = agent.run(company)
    live_flags = {k: v.endswith("_live") for k, v in sources.items()}
    assert live_flags == {
        "revenue_growth_signal": True,
        "rd_investment_signal": True,
        "capex_growth_signal": False,
        "satellite_imagery_current": True,
        "satellite_imagery_prior": True,
    }


def test_close_closes_all_owned_clients(company):
    sec = _FakeSecClient({})
    imagery = _FakeImagery(_pair())
    agent = _agent(live=True, sec_client=sec, imagery_provider=imagery)
    agent.close()
    assert sec.closed and imagery.closed


def test_invalid_coordinates_raise_before_any_fetch(company):
    agent = _agent(live=True)
    bad_company = CompanyInput(
        ticker="BAD", company_name="Bad Co", facility_name="Nowhere", latitude=999.0, longitude=0.0
    )
    with pytest.raises(ValueError):
        agent.run(bad_company)


# --- fetch_enriched_evidence: optional, additive real-data enrichment ---


def test_fetch_enriched_evidence_returns_none_none_in_offline_mode(company):
    agent = _agent(live=False)
    profile, pair = agent.fetch_enriched_evidence(company)
    assert profile is None
    assert pair is None


def test_fetch_enriched_evidence_returns_none_none_when_not_yet_run_even_in_live_mode(company):
    """fetch_enriched_evidence() reuses run()'s already-fetched tile pair
    rather than issuing a second NASA GIBS request -- if run() was never
    called on this instance, there is no pair to reuse yet.
    """
    sec = _FakeSecClient({}, profile="a-profile-sentinel")
    agent = _agent(live=True, sec_client=sec)
    profile, pair = agent.fetch_enriched_evidence(company)
    assert profile == "a-profile-sentinel"
    assert pair is None
    assert sec.profile_calls == 1


def test_fetch_enriched_evidence_reuses_the_tile_pair_from_run(company):
    sec = _FakeSecClient(
        {
            "revenue_growth_signal": FinancialSignal(0.5, True, "sec_edgar_live", "rev"),
            "rd_investment_signal": FinancialSignal(0.5, True, "sec_edgar_live", "rd"),
            "capex_growth_signal": FinancialSignal(0.5, True, "sec_edgar_live", "capex"),
        },
        profile="a-profile-sentinel",
    )
    pair = _pair()
    agent = _agent(live=True, sec_client=sec, imagery_provider=_FakeImagery(pair))
    agent.run(company)  # populates self._last_change_pair

    profile, reused_pair = agent.fetch_enriched_evidence(company)
    assert profile == "a-profile-sentinel"
    assert reused_pair is pair  # exact same object, not a second fetch


def test_fetch_enriched_evidence_never_raises_when_financial_profile_fetch_fails(company):
    sec = _FakeSecClient({}, profile=None)  # configured to raise
    agent = _agent(live=True, sec_client=sec)
    profile, pair = agent.fetch_enriched_evidence(company)
    assert profile is None  # failure degrades to None, never propagates
    assert pair is None


# --- CACHED_REAL fallback: genuine persisted-assessment reuse on live failure ---


def test_cached_real_fallback_replaces_insufficient_data_signal(company):
    """When a signal's live fetch genuinely failed (status ==
    INSUFFICIENT_DATA) and cache_lookup finds a genuine prior value for
    this exact (ticker, signal), the signal is replaced with a typed
    CACHED_REAL result carrying the real, previously-observed value and
    its original retrieval timestamp -- never a fabricated number.
    """
    sec = _FakeSecClient(
        {
            "revenue_growth_signal": FinancialSignal(0.6, True, "sec_edgar_live", "rev"),
            "rd_investment_signal": FinancialSignal(0.5, True, "sec_edgar_live", "rd"),
            "capex_growth_signal": FinancialSignal(
                None, False, "insufficient_data", "no usable tag", status="INSUFFICIENT_DATA"
            ),
        }
    )

    def fake_cache_lookup(ticker: str, signal_key: str):
        assert ticker == "TSLA"
        assert signal_key == "capex_growth_signal"
        return 0.42, "2026-01-01T00:00:00+00:00"

    agent = _agent(live=True, sec_client=sec, cache_lookup=fake_cache_lookup)
    financials, _, _, sources = agent.run(company)

    capex = financials["capex_growth_signal"]
    assert capex.status == "AVAILABLE"  # the coarse flag is now usable again
    assert capex.value == pytest.approx(0.42)  # the genuine, previously-observed value -- never fabricated
    assert capex.quality_state == "CACHED_REAL"
    assert capex.source == "cached_real:sec_edgar:2026-01-01T00:00:00+00:00"
    assert "STALE" in capex.summary
    assert sources["capex_growth_signal"] == capex.source
    # Signals that succeeded live are completely untouched by the fallback.
    assert financials["revenue_growth_signal"].quality_state != "CACHED_REAL"


def test_cached_real_fallback_leaves_signal_untouched_on_cache_miss(company):
    sec = _FakeSecClient(
        {
            "revenue_growth_signal": FinancialSignal(0.6, True, "sec_edgar_live", "rev"),
            "rd_investment_signal": FinancialSignal(0.5, True, "sec_edgar_live", "rd"),
            "capex_growth_signal": FinancialSignal(
                None, False, "insufficient_data", "no usable tag", status="INSUFFICIENT_DATA"
            ),
        }
    )
    agent = _agent(live=True, sec_client=sec, cache_lookup=lambda ticker, key: None)
    financials, _, _, _ = agent.run(company)

    capex = financials["capex_growth_signal"]
    assert capex.status == "INSUFFICIENT_DATA"  # untouched -- honestly still a gap
    assert capex.value is None


def test_cached_real_fallback_never_touches_disclosed_synthetic_fallback_signal(company):
    """synthetic_fallback:* signals keep status=='AVAILABLE' by
    construction (see data/sec_edgar_client.py), so the cache-fallback
    pass must never even consider replacing them -- confirms cache_lookup
    is never called for a signal that isn't genuinely INSUFFICIENT_DATA.
    """
    calls: list[tuple[str, str]] = []

    def tracking_cache_lookup(ticker: str, signal_key: str):
        calls.append((ticker, signal_key))
        return 0.99, "2026-01-01T00:00:00+00:00"

    agent = _agent(live=True, cache_lookup=tracking_cache_lookup)  # default fake sec client has a synthetic_fallback signal
    financials, _, _, _ = agent.run(company)

    assert calls == []  # never invoked -- no signal was genuinely INSUFFICIENT_DATA
    assert financials["capex_growth_signal"].value == 0.0  # unchanged from the fake


def test_cache_lookup_none_disables_fallback_entirely(company):
    sec = _FakeSecClient(
        {
            "revenue_growth_signal": FinancialSignal(0.6, True, "sec_edgar_live", "rev"),
            "rd_investment_signal": FinancialSignal(0.5, True, "sec_edgar_live", "rd"),
            "capex_growth_signal": FinancialSignal(
                None, False, "insufficient_data", "no usable tag", status="INSUFFICIENT_DATA"
            ),
        }
    )
    agent = _agent(live=True, sec_client=sec, cache_lookup=None)
    financials, _, _, _ = agent.run(company)
    assert financials["capex_growth_signal"].status == "INSUFFICIENT_DATA"


# --- No verified facility coordinates (financial-only assessment) ---
# See data/company_resolver.py's RESOLVED_NO_FACILITY status: a company
# resolved to a real ticker/CIK, but no verified facility address/coordinate
# could be established. Satellite/vision evidence must never be attempted
# from a guessed location -- the agent must skip imagery entirely and report
# an honest INSUFFICIENT_DATA-classifying provenance label.


@pytest.fixture
def company_no_facility():
    return CompanyInput(
        ticker="TSLA", company_name="Tesla, Inc.", facility_name="No verified facility",
        latitude=None, longitude=None,
    )


def test_offline_mode_no_coordinates_skips_imagery_entirely(company_no_facility):
    agent = _agent(live=False)
    financials, current_tile, prior_tile, sources = agent.run(company_no_facility)

    assert current_tile is None
    assert prior_tile is None
    assert isinstance(financials["revenue_growth_signal"].value, float)  # financial intelligence still ran
    assert sources["satellite_imagery_current"] == "insufficient_data:no_verified_facility_coordinates"
    assert sources["satellite_imagery_prior"] == "insufficient_data:no_verified_facility_coordinates"
    # Financial signal sources are still the offline-demo label, same as the with-coordinates case.
    assert sources["revenue_growth_signal"] == "offline_demo_mode"


def test_live_mode_no_coordinates_skips_nasa_gibs_entirely(company_no_facility):
    class _FailIfCalledImagery:
        def fetch_change_pair(self, lat, lon):
            raise AssertionError("NASA GIBS must never be called when no verified facility coordinates exist")

        def close(self):
            pass

    agent = _agent(live=True, imagery_provider=_FailIfCalledImagery())
    financials, current_tile, prior_tile, sources = agent.run(company_no_facility)

    assert current_tile is None
    assert prior_tile is None
    assert financials["revenue_growth_signal"].value == 0.6  # real SEC EDGAR signals still fetched
    assert financials["rd_investment_signal"].value == 0.5
    assert sources["satellite_imagery_current"] == "insufficient_data:no_verified_facility_coordinates"
    assert sources["satellite_imagery_prior"] == "insufficient_data:no_verified_facility_coordinates"
    assert sources["revenue_growth_signal"] == "sec_edgar_live"


def test_live_mode_no_coordinates_never_raises_lat_lon_validation_error(company_no_facility):
    """validate_lat_lon must only be called when has_coordinates -- a
    financial-only company (lat=lon=None) must never hit the "invalid
    coordinate" ValueError path that a real bad-coordinate company would.
    """
    agent = _agent(live=True)
    agent.run(company_no_facility)  # must not raise


def test_live_mode_no_coordinates_last_change_pair_stays_none(company_no_facility):
    agent = _agent(live=True)
    agent.run(company_no_facility)
    profile, pair = agent.fetch_enriched_evidence(company_no_facility)
    assert pair is None


def test_cache_lookup_exception_never_propagates(company):
    sec = _FakeSecClient(
        {
            "revenue_growth_signal": FinancialSignal(0.6, True, "sec_edgar_live", "rev"),
            "rd_investment_signal": FinancialSignal(0.5, True, "sec_edgar_live", "rd"),
            "capex_growth_signal": FinancialSignal(
                None, False, "insufficient_data", "no usable tag", status="INSUFFICIENT_DATA"
            ),
        }
    )

    def broken_cache_lookup(ticker, key):
        raise RuntimeError("db unavailable")

    agent = _agent(live=True, sec_client=sec, cache_lookup=broken_cache_lookup)
    financials, _, _, _ = agent.run(company)  # must not raise
    assert financials["capex_growth_signal"].status == "INSUFFICIENT_DATA"
