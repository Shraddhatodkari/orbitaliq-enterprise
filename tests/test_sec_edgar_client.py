import httpx
import pytest
import respx

from orbitaliq.config import Settings
from orbitaliq.data.sec_edgar_client import SecEdgarClient, synthetic_financial_signals

_TICKER_DIRECTORY = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."},
}


def _facts(revenue=None, rd=None, capex=None) -> dict:
    gaap = {}
    if revenue:
        gaap["Revenues"] = {"units": {"USD": revenue}}
    if rd:
        gaap["ResearchAndDevelopmentExpense"] = {"units": {"USD": rd}}
    if capex:
        gaap["PaymentsToAcquirePropertyPlantAndEquipment"] = {"units": {"USD": capex}}
    return {"facts": {"us-gaap": gaap}}


def _annual(fy, val):
    return {"form": "10-K", "fp": "FY", "fy": fy, "val": val}


@pytest.fixture
def settings():
    return Settings(SEC_EDGAR_USER_AGENT="Test Suite test@example.com", LIVE_DATA_TIMEOUT_SECONDS=2.0)


@respx.mock
def test_resolve_cik_success(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    client = SecEdgarClient(settings=settings)
    cik, title = client.resolve_cik("aapl")
    assert cik == "0000320193"
    assert title == "Apple Inc."
    client.close()


@respx.mock
def test_resolve_cik_unknown_ticker_returns_none(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    client = SecEdgarClient(settings=settings)
    cik, title = client.resolve_cik("NOPE")
    assert cik is None and title is None
    client.close()


@respx.mock
def test_resolve_cik_caches_directory_across_calls(settings):
    route = respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    client = SecEdgarClient(settings=settings)
    client.resolve_cik("AAPL")
    client.resolve_cik("TSLA")
    assert route.call_count == 1
    client.close()


@respx.mock
def test_fetch_financial_signals_live_growth_computation(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(
            200,
            json=_facts(
                revenue=[_annual(2022, 100_000_000), _annual(2023, 130_000_000)],
                rd=[_annual(2022, 10_000_000), _annual(2023, 9_000_000)],
            ),
        )
    )
    client = SecEdgarClient(settings=settings)
    signals = client.fetch_financial_signals("AAPL")

    assert signals["revenue_growth_signal"].live is True
    assert signals["revenue_growth_signal"].value > 0.5  # +30% YoY is near the top of the (-10%, 30%) range
    assert signals["revenue_growth_signal"].source == "sec_edgar_live"
    assert signals["revenue_growth_signal"].quality_state == "LIVE"
    assert signals["rd_investment_signal"].live is True
    assert signals["rd_investment_signal"].value < 0.5  # R&D shrank YoY

    # capex was never reported for this filer -> a typed INSUFFICIENT_DATA
    # result (never a fabricated value), but the other two metrics still
    # succeeded live (partial-coverage resilience).
    assert signals["capex_growth_signal"].live is False
    assert signals["capex_growth_signal"].value is None
    assert signals["capex_growth_signal"].status == "INSUFFICIENT_DATA"
    assert signals["capex_growth_signal"].source == "insufficient_data"
    client.close()


@respx.mock
def test_fetch_financial_signals_unknown_ticker_falls_back_for_all_metrics(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    client = SecEdgarClient(settings=settings)
    signals = client.fetch_financial_signals("NOPE")
    assert all(not s.live for s in signals.values())
    assert all(s.value is None for s in signals.values())
    assert all(s.status == "INSUFFICIENT_DATA" for s in signals.values())
    assert all(s.source == "insufficient_data" for s in signals.values())
    assert all(not s.source.startswith("synthetic_fallback") for s in signals.values())
    client.close()


@respx.mock
def test_fetch_financial_signals_network_failure_falls_back(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(side_effect=httpx.ConnectError("boom"))
    client = SecEdgarClient(settings=settings)
    signals = client.fetch_financial_signals("AAPL")
    assert all(not s.live for s in signals.values())
    assert all(s.value is None for s in signals.values())
    assert all(s.status == "INSUFFICIENT_DATA" for s in signals.values())
    # An unexpected ConnectError while fetching the ticker directory is an
    # ERROR (unexpected exception), distinct from the clean, documented-gap
    # INSUFFICIENT_DATA case (e.g. a ticker genuinely absent from a
    # successfully-fetched directory) -- see core/data_quality_state.py.
    assert all(s.source == "error:sec_edgar:ConnectError" for s in signals.values())
    assert all(s.quality_state == "ERROR" for s in signals.values())
    client.close()


@respx.mock
def test_fetch_financial_signals_zero_prior_year_value_falls_back(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=_facts(revenue=[_annual(2022, 0), _annual(2023, 500_000)]))
    )
    client = SecEdgarClient(settings=settings)
    signals = client.fetch_financial_signals("AAPL")
    assert signals["revenue_growth_signal"].live is False
    assert signals["revenue_growth_signal"].value is None
    assert signals["revenue_growth_signal"].status == "INSUFFICIENT_DATA"
    assert "zero" in signals["revenue_growth_signal"].summary.lower()
    client.close()


def test_live_mode_failure_never_reaches_the_synthetic_generator():
    """The escalated 'no synthetic production fallback' requirement: every
    live-mode failure path in fetch_financial_signals must return a typed
    INSUFFICIENT_DATA result whose source is never a synthetic_fallback:*
    label -- that label (and the fabricated numeric value behind it) is
    reachable ONLY via synthetic_financial_signals(), the offline-demo-mode
    / test-fixture entry point, never from a live SecEdgarClient call.
    """
    import respx as _respx

    with _respx.mock:
        _respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
        )
        client = SecEdgarClient(settings=Settings(SEC_EDGAR_USER_AGENT="Test test@example.com"))
        signals = client.fetch_financial_signals("NOPE")
        for signal in signals.values():
            assert signal.value is None
            assert signal.status == "INSUFFICIENT_DATA"
            assert not signal.source.startswith("synthetic_fallback")
        client.close()


def test_synthetic_financial_signals_are_deterministic_and_labeled():
    a = synthetic_financial_signals("TSLA")
    b = synthetic_financial_signals("TSLA")
    assert {k: v.as_dict() for k, v in a.items()} == {k: v.as_dict() for k, v in b.items()}
    assert all(not s.live for s in a.values())
    assert set(a.keys()) == {"revenue_growth_signal", "rd_investment_signal", "capex_growth_signal"}


def test_synthetic_financial_signals_differ_by_ticker():
    a = synthetic_financial_signals("TSLA")
    b = synthetic_financial_signals("AAPL")
    assert a["revenue_growth_signal"].value != b["revenue_growth_signal"].value


# --- FinancialProfile / fetch_financial_profile (real-data-only, no synthetic fallback) ---


def _full_facts() -> dict:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_annual(2022, 100_000_000), _annual(2023, 130_000_000)]}},
                "ResearchAndDevelopmentExpense": {"units": {"USD": [_annual(2022, 10_000_000), _annual(2023, 9_000_000)]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_annual(2022, 5_000_000), _annual(2023, 7_000_000)]}
                },
                "GrossProfit": {"units": {"USD": [_annual(2023, 60_000_000)]}},
                "OperatingIncomeLoss": {"units": {"USD": [_annual(2022, 20_000_000), _annual(2023, 25_000_000)]}},
                "NetIncomeLoss": {"units": {"USD": [_annual(2023, 18_000_000)]}},
                "Assets": {"units": {"USD": [_annual(2022, 200_000_000), _annual(2023, 220_000_000)]}},
                "AssetsCurrent": {"units": {"USD": [_annual(2023, 90_000_000)]}},
                "LiabilitiesCurrent": {"units": {"USD": [_annual(2023, 60_000_000)]}},
                "StockholdersEquity": {"units": {"USD": [_annual(2023, 100_000_000)]}},
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_annual(2023, 40_000_000)]}},
                "LongTermDebtCurrent": {"units": {"USD": [_annual(2023, 5_000_000)]}},
                "LongTermDebtNoncurrent": {"units": {"USD": [_annual(2023, 45_000_000)]}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_annual(2023, 30_000_000)]}},
                # PropertyPlantAndEquipmentNet deliberately omitted: exercises
                # a genuine partial-coverage INSUFFICIENT_DATA case.
            }
        }
    }


@respx.mock
def test_fetch_financial_profile_live_computes_all_available_metrics(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=_full_facts())
    )
    client = SecEdgarClient(settings=settings)
    profile = client.fetch_financial_profile("AAPL")

    assert profile.ticker == "AAPL"
    assert profile.cik == "0000320193"
    assert profile.filing_source_url is not None
    assert len(profile.metrics) == 19

    revenue_growth = profile.metrics["revenue_growth"]
    assert revenue_growth.status == "AVAILABLE"
    assert revenue_growth.current_value == pytest.approx(130_000_000)
    assert revenue_growth.yoy_change_pct == pytest.approx(0.30, abs=1e-6)
    assert revenue_growth.fiscal_period == "FY2023"
    assert revenue_growth.xbrl_concept
    # A directly-disclosed raw XBRL fact -- LIVE, not DERIVED.
    assert revenue_growth.quality_state == "LIVE"

    gross_margin = profile.metrics["gross_margin"]
    assert gross_margin.status == "AVAILABLE"
    assert gross_margin.unit == "percent"
    assert gross_margin.current_value == pytest.approx(60_000_000 / 130_000_000)
    # A ratio computed from two raw facts -- DERIVED, not itself a
    # directly-disclosed fact. See core/data_quality_state.py.
    assert gross_margin.quality_state == "DERIVED"
    assert gross_margin.source == "derived:sec_edgar"

    total_debt = profile.metrics["total_debt"]
    assert total_debt.status == "AVAILABLE"
    assert total_debt.current_value == pytest.approx(50_000_000)  # LongTermDebtCurrent + Noncurrent
    assert total_debt.quality_state == "LIVE"

    # PP&E was never disclosed by this filer -> honestly INSUFFICIENT_DATA,
    # never a fabricated/synthetic value.
    ppe_growth = profile.metrics["ppe_growth"]
    assert ppe_growth.status == "INSUFFICIENT_DATA"
    assert ppe_growth.quality_state == "INSUFFICIENT_DATA"
    assert ppe_growth.current_value is None
    assert "insufficient sec data" in ppe_growth.note.lower()
    client.close()


def _full_facts_with_stale_split_debt_only() -> dict:
    """Reproduces the exact real-world shape that triggered the FY2013
    debt-metric staleness bug: this filer reported the SPLIT
    LongTermDebtCurrent/LongTermDebtNoncurrent pair together only in one
    old fiscal year (2013), while reporting the SINGLE combined
    LongTermDebt tag in the current/latest year (2023) alongside every
    other metric. Total Debt / Net Debt / Debt-to-Equity must resolve to
    FY2023 -- the latest valid comparable SEC observation -- never freeze
    on the old FY2013 split-tag year just because that's the only year
    where both split tags happened to co-occur.
    """
    facts = _full_facts()
    gaap = facts["facts"]["us-gaap"]
    gaap["LongTermDebtCurrent"] = {"units": {"USD": [_annual(2013, 1_000_000)]}}
    gaap["LongTermDebtNoncurrent"] = {"units": {"USD": [_annual(2013, 9_000_000)]}}
    gaap["LongTermDebt"] = {"units": {"USD": [_annual(2023, 50_000_000)]}}
    return facts


@respx.mock
def test_total_debt_uses_latest_valid_period_not_stale_fy2013_split_tags(settings):
    """Regression test for the reported bug: Revenue/R&D/CapEx/Assets/Cash
    all correctly tracked FY2025-equivalent (here FY2023) while Total
    Debt/Net Debt/Debt-to-Equity incorrectly froze on FY2013. See
    ``_debt_by_fy``'s per-fiscal-year merge fix in sec_edgar_client.py.
    """
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=_full_facts_with_stale_split_debt_only())
    )
    client = SecEdgarClient(settings=settings)
    profile = client.fetch_financial_profile("AAPL")

    # Sanity check: the other metrics on this same filer correctly track
    # the latest (2023) fiscal year, exactly matching the reported symptom.
    assert profile.metrics["revenue_growth"].fiscal_period == "FY2023"

    total_debt = profile.metrics["total_debt"]
    assert total_debt.status == "AVAILABLE"
    assert total_debt.fiscal_period == "FY2023"
    assert total_debt.fiscal_period != "FY2013"
    assert total_debt.current_value == pytest.approx(50_000_000)  # the single LongTermDebt tag for FY2023
    assert total_debt.quality_state == "LIVE"

    net_debt = profile.metrics["net_debt"]
    assert net_debt.status == "AVAILABLE"
    assert net_debt.fiscal_period == "FY2023"
    assert net_debt.fiscal_period != "FY2013"
    # cash at FY2023 is 40_000_000 (see _full_facts)
    assert net_debt.current_value == pytest.approx(50_000_000 - 40_000_000)
    assert net_debt.quality_state == "DERIVED"

    debt_to_equity = profile.metrics["debt_to_equity"]
    assert debt_to_equity.status == "AVAILABLE"
    assert debt_to_equity.fiscal_period == "FY2023"
    assert debt_to_equity.fiscal_period != "FY2013"
    # equity at FY2023 is 100_000_000 (see _full_facts)
    assert debt_to_equity.current_value == pytest.approx(50_000_000 / 100_000_000)
    assert debt_to_equity.quality_state == "DERIVED"
    client.close()


@respx.mock
def test_total_debt_returns_insufficient_data_when_no_valid_debt_observation_exists(settings):
    """No debt tag (split pair or single combined) is reported by this
    filer for ANY fiscal year -- must honestly report INSUFFICIENT_DATA,
    never select an old/unrelated value just to fill the field.
    """
    facts = _full_facts()
    gaap = facts["facts"]["us-gaap"]
    gaap.pop("LongTermDebtCurrent", None)
    gaap.pop("LongTermDebtNoncurrent", None)
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=facts)
    )
    client = SecEdgarClient(settings=settings)
    profile = client.fetch_financial_profile("AAPL")

    for key in ("total_debt", "net_debt", "debt_to_equity"):
        metric = profile.metrics[key]
        assert metric.status == "INSUFFICIENT_DATA"
        assert metric.quality_state == "INSUFFICIENT_DATA"
        assert metric.current_value is None
        assert metric.fiscal_period is None
        assert not metric.source.startswith("synthetic_fallback")
    client.close()


def test_debt_by_fy_merges_split_and_single_tags_per_fiscal_year_never_all_or_nothing():
    """Direct unit test of the fixed root-cause function. A filer that
    reported the split current/noncurrent pair together only in FY2013,
    and the single combined LongTermDebt tag in FY2024 and FY2025 (with
    FY2025 being the actual latest 10-K), must have ALL THREE years
    present in the merged result -- proving the merge is per-fiscal-year,
    not an all-or-nothing choice between the split-tag intersection and
    the single-tag series.
    """
    from orbitaliq.data.sec_edgar_client import _debt_by_fy

    gaap = {
        "LongTermDebtCurrent": {"units": {"USD": [_annual(2013, 1_000_000)]}},
        "LongTermDebtNoncurrent": {"units": {"USD": [_annual(2013, 9_000_000)]}},
        "LongTermDebt": {
            "units": {"USD": [_annual(2024, 48_000_000), _annual(2025, 55_000_000)]}
        },
    }
    merged = _debt_by_fy(gaap)
    assert merged == {2013: 10_000_000, 2024: 48_000_000, 2025: 55_000_000}
    # The latest fiscal year a downstream _metric_from_point-style consumer
    # would select is FY2025 -- the genuinely latest disclosed period.
    assert max(merged.keys()) == 2025


def test_debt_by_fy_empty_when_no_debt_tags_disclosed():
    from orbitaliq.data.sec_edgar_client import _debt_by_fy

    assert _debt_by_fy({}) == {}


def _full_facts_with_revenue_tag_migration() -> dict:
    """Reproduces the real-world shape confirmed live against SEC EDGAR
    for APLE (Apple Hospitality REIT, CIK0001418121): the filer tagged
    revenue under ``Revenues`` only through an old fiscal year, then
    switched to ``RevenueFromContractWithCustomerExcludingAssessedTax``
    (the ASC 606 tag) for every year since, including the actual latest
    10-K. Every other metric here uses its normal single, unmigrated tag
    and reports the current year, exactly matching the reported symptom:
    CapEx/Operating Income/FCF/Assets/ROA/ROE/Cash/Debt all correctly at
    the latest year while Revenue (and anything derived from it) froze on
    the old tag's last year.
    """
    facts = _full_facts()
    gaap = facts["facts"]["us-gaap"]
    gaap["Revenues"] = {"units": {"USD": [_annual(2016, 90_000_000), _annual(2017, 95_000_000)]}}
    gaap["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
        "units": {"USD": [_annual(2024, 140_000_000), _annual(2025, 150_000_000)]}
    }
    # GrossProfit also disclosed for FY2025 (in addition to _full_facts'
    # FY2023), so gross_margin has a common year with the now-complete
    # revenue series and can prove the ratio metric un-freezes too.
    gaap["GrossProfit"]["units"]["USD"].append(_annual(2025, 65_000_000))
    return facts


@respx.mock
def test_revenue_growth_uses_latest_valid_period_not_stale_pre_asc606_tag(settings):
    """Regression test for the reported bug: the dashboard's "Most recent
    fiscal period" showed FY2017 for an assessment whose other metrics
    (CapEx Growth, Operating Income Growth, FCF Trend, Asset Growth, ROA,
    ROE, Cash, Total Debt, Debt/Equity) were genuinely FY2025. Root cause:
    ``_by_fy`` stopped at the first candidate revenue tag with any data
    (the pre-ASC-606 ``Revenues`` tag, last reported FY2017) and never
    checked the later, currently-used tag that actually has FY2024/FY2025
    data. See ``_by_fy``'s merge-across-tags fix in sec_edgar_client.py.
    """
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=_full_facts_with_revenue_tag_migration())
    )
    client = SecEdgarClient(settings=settings)
    profile = client.fetch_financial_profile("AAPL")

    # Sanity check: the other metrics on this same filer correctly track
    # the latest (2023, per _full_facts) fiscal year -- confirming the
    # fixture reproduces "other metrics current, revenue stale" exactly,
    # matching the reported symptom before this fix.
    assert profile.metrics["operating_income_growth"].fiscal_period == "FY2023"

    revenue_growth = profile.metrics["revenue_growth"]
    assert revenue_growth.status == "AVAILABLE"
    assert revenue_growth.fiscal_period == "FY2025"
    assert revenue_growth.fiscal_period != "FY2017"
    assert revenue_growth.current_value == pytest.approx(150_000_000)
    assert revenue_growth.prior_value == pytest.approx(140_000_000)
    assert revenue_growth.prior_fiscal_period == "FY2024"

    # Metrics computed from revenue (ratios) also stop being pinned to the
    # stale tag once the underlying revenue series is complete.
    gross_margin = profile.metrics["gross_margin"]
    assert gross_margin.status == "AVAILABLE"
    assert gross_margin.fiscal_period == "FY2025"
    client.close()


def test_by_fy_merges_across_tags_never_stops_at_first_populated_tag():
    """Direct unit test of the fixed root-cause function. A filer that
    reported an old tag only for early years, and a newer replacement tag
    only for recent years (a genuine tag migration, e.g. the ASC 606
    revenue transition), must have BOTH tags' years present in the merged
    result -- proving _by_fy no longer returns as soon as the first
    candidate tag has any data at all.
    """
    from orbitaliq.data.sec_edgar_client import _by_fy

    gaap = {
        "Revenues": {"units": {"USD": [_annual(2016, 90_000_000), _annual(2017, 95_000_000)]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {"USD": [_annual(2024, 140_000_000), _annual(2025, 150_000_000)]}
        },
    }
    merged = _by_fy(gaap, [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ])
    assert merged == {2016: 90_000_000, 2017: 95_000_000, 2024: 140_000_000, 2025: 150_000_000}
    assert max(merged.keys()) == 2025


def test_by_fy_returns_first_tags_data_unchanged_when_no_migration():
    """Ordinary, non-migrated case (the vast majority of filers): only one
    candidate tag is ever populated. Confirms the merge fix is a strict
    superset behavior -- single-tag filers see no change at all.
    """
    from orbitaliq.data.sec_edgar_client import _by_fy

    gaap = {"Revenues": {"units": {"USD": [_annual(2022, 100_000_000), _annual(2023, 130_000_000)]}}}
    merged = _by_fy(gaap, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
    assert merged == {2022: 100_000_000, 2023: 130_000_000}


@respx.mock
def test_fetch_financial_profile_unknown_ticker_is_all_insufficient_never_fabricated(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    client = SecEdgarClient(settings=settings)
    profile = client.fetch_financial_profile("NOPE")

    assert profile.cik is None
    assert profile.filing_source_url is None
    assert len(profile.metrics) == 19
    assert all(m.status == "INSUFFICIENT_DATA" for m in profile.metrics.values())
    assert all(m.current_value is None for m in profile.metrics.values())
    assert all("not found" in m.note.lower() for m in profile.metrics.values())
    client.close()


@respx.mock
def test_fetch_financial_profile_never_falls_back_to_synthetic_source(settings):
    """FinancialProfile has no synthetic/fallback source at all — unlike
    FinancialSignal, an unreachable company-facts endpoint must surface as
    a typed gap (never a labeled-but-fabricated value).

    An unexpected exception while fetching (this test's ConnectError) is
    an unexpected fetch failure, not a documented "this filer doesn't
    disclose this concept" gap — so it must classify as
    DataQualityState.ERROR, distinct from the clean-absence
    DataQualityState.INSUFFICIENT_DATA case covered by
    test_fetch_financial_profile_unknown_ticker_is_all_insufficient_never_fabricated
    above. Both keep ``status == "INSUFFICIENT_DATA"`` (the coarse
    AVAILABLE/INSUFFICIENT_DATA flag every existing caller checks is
    unchanged); only the finer-grained ``source``/``quality_state`` differ.
    """
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        side_effect=httpx.ConnectError("boom")
    )
    client = SecEdgarClient(settings=settings)
    profile = client.fetch_financial_profile("AAPL")
    assert all(m.status == "INSUFFICIENT_DATA" for m in profile.metrics.values())
    # Honestly labeled "error:sec_edgar:ConnectError" — never
    # "synthetic_fallback:*" (which would imply a fabricated value was
    # substituted) and never the clean-absence "insufficient_data" label
    # either, since this genuinely errored rather than cleanly resolving
    # to "not disclosed."
    assert all(m.source == "error:sec_edgar:ConnectError" for m in profile.metrics.values())
    assert all(m.quality_state == "ERROR" for m in profile.metrics.values())
    assert all(not m.source.startswith("synthetic_fallback") for m in profile.metrics.values())
    client.close()


# --- HistoricalFinancials / fetch_historical_financials ---


@respx.mock
def test_fetch_historical_financials_only_includes_disclosed_years_and_fields(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=_full_facts())
    )
    client = SecEdgarClient(settings=settings)
    historical = client.fetch_historical_financials("AAPL", max_years=6)

    assert historical.cik == "0000320193"
    fiscal_years = {fy["fiscal_year"]: fy for fy in historical.fiscal_years}
    assert set(fiscal_years.keys()) == {2022, 2023}
    # R&D was disclosed for both years in this fixture.
    assert fiscal_years[2022]["rd"] == pytest.approx(10_000_000)
    assert fiscal_years[2023]["revenue"] == pytest.approx(130_000_000)
    assert fiscal_years[2023]["gross_margin"] == pytest.approx(60_000_000 / 130_000_000)
    # gross_profit was never disclosed for 2022 -> margin absent, not
    # interpolated or fabricated.
    assert fiscal_years[2022]["gross_margin"] is None
    client.close()


@respx.mock
def test_fetch_historical_financials_unknown_ticker_returns_empty_series(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    client = SecEdgarClient(settings=settings)
    historical = client.fetch_historical_financials("NOPE")
    assert historical.fiscal_years == []
    client.close()


@respx.mock
def test_fetch_historical_financials_respects_max_years(settings):
    many_years_revenue = [_annual(y, float(y) * 1000) for y in range(2015, 2024)]
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=_facts(revenue=many_years_revenue))
    )
    client = SecEdgarClient(settings=settings)
    historical = client.fetch_historical_financials("AAPL", max_years=3)
    assert len(historical.fiscal_years) == 3
    # Most recent 3 years, but returned in ascending order.
    assert [fy["fiscal_year"] for fy in historical.fiscal_years] == [2021, 2022, 2023]
    client.close()


# --- fetch_company_facts memoization cache ---


@respx.mock
def test_fetch_company_facts_is_cached_per_cik_across_calls(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    route = respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=_full_facts())
    )
    client = SecEdgarClient(settings=settings)

    # fetch_financial_signals (legacy) and fetch_financial_profile (new)
    # both need company-facts for the same CIK within one assessment --
    # the cache must ensure this issues exactly one live HTTP call.
    client.fetch_financial_signals("AAPL")
    client.fetch_financial_profile("AAPL")
    client.fetch_historical_financials("AAPL")

    assert route.call_count == 1
    client.close()


@respx.mock
def test_fetch_company_facts_cache_also_memoizes_a_failed_lookup(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    route = respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        side_effect=httpx.ConnectError("boom")
    )
    client = SecEdgarClient(settings=settings)
    client.fetch_financial_profile("AAPL")
    # _get() retries transient failures once (stop_after_attempt(2)), so the
    # FIRST logical fetch already makes 2 physical HTTP calls -- the cache
    # is proven by the SECOND logical fetch adding zero more.
    count_after_first_call = route.call_count
    assert count_after_first_call == 2
    client.fetch_financial_profile("AAPL")
    assert route.call_count == count_after_first_call  # the None result was cached too, not re-fetched
    client.close()


# --- fetch_business_address: real-data-only, never a guessed/partial address ---


@respx.mock
def test_fetch_business_address_returns_the_disclosed_business_address(settings):
    respx.get("https://data.sec.gov/submissions/CIK0001318605.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "addresses": {
                    "business": {
                        "street1": "1 Tesla Road", "street2": None, "city": "Austin",
                        "stateOrCountry": "TX", "zipCode": "78725",
                    },
                    "mailing": {
                        "street1": "PO Box 1", "street2": None, "city": "Austin",
                        "stateOrCountry": "TX", "zipCode": "78725",
                    },
                }
            },
        )
    )
    client = SecEdgarClient(settings=settings)
    address = client.fetch_business_address("0001318605")
    assert address is not None
    assert address.street1 == "1 Tesla Road"
    assert address.city == "Austin"
    assert address.state_or_country == "TX"
    assert address.zip_code == "78725"
    assert address.one_line() == "1 Tesla Road, Austin TX 78725"
    assert address.source_url.endswith("CIK0001318605.json")
    client.close()


@respx.mock
def test_fetch_business_address_falls_back_to_mailing_address_when_no_business_address(settings):
    respx.get("https://data.sec.gov/submissions/CIK0001318605.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "addresses": {
                    "mailing": {
                        "street1": "PO Box 1", "street2": None, "city": "Austin",
                        "stateOrCountry": "TX", "zipCode": "78725",
                    },
                }
            },
        )
    )
    client = SecEdgarClient(settings=settings)
    address = client.fetch_business_address("0001318605")
    assert address is not None
    assert address.street1 == "PO Box 1"
    client.close()


@respx.mock
def test_fetch_business_address_returns_none_when_no_address_disclosed_at_all(settings):
    respx.get("https://data.sec.gov/submissions/CIK0001318605.json").mock(
        return_value=httpx.Response(200, json={"addresses": {}})
    )
    client = SecEdgarClient(settings=settings)
    assert client.fetch_business_address("0001318605") is None
    client.close()


@respx.mock
def test_fetch_business_address_returns_none_on_network_failure_never_raises(settings):
    respx.get("https://data.sec.gov/submissions/CIK0001318605.json").mock(side_effect=httpx.ConnectError("boom"))
    client = SecEdgarClient(settings=settings)
    assert client.fetch_business_address("0001318605") is None
    client.close()


@respx.mock
def test_fetch_business_address_returns_none_for_malformed_address_field(settings):
    respx.get("https://data.sec.gov/submissions/CIK0001318605.json").mock(
        return_value=httpx.Response(200, json={"addresses": {"business": "not-a-dict"}})
    )
    client = SecEdgarClient(settings=settings)
    assert client.fetch_business_address("0001318605") is None
    client.close()


# --- get_ticker_directory / ticker_directory_error: read-only exposure for company_resolver.py ---


@respx.mock
def test_get_ticker_directory_returns_the_full_cached_directory(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    client = SecEdgarClient(settings=settings)
    directory = client.get_ticker_directory()
    assert set(directory.keys()) == {"AAPL", "TSLA"}
    assert directory["AAPL"]["title"] == "Apple Inc."
    client.close()


@respx.mock
def test_get_ticker_directory_reuses_the_same_cache_resolve_cik_populates(settings):
    route = respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKER_DIRECTORY)
    )
    client = SecEdgarClient(settings=settings)
    client.resolve_cik("AAPL")
    client.get_ticker_directory()
    assert route.call_count == 1  # no second fetch
    client.close()


@respx.mock
def test_get_ticker_directory_returns_empty_dict_on_fetch_failure_and_exposes_the_error(settings):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(side_effect=httpx.ConnectError("boom"))
    client = SecEdgarClient(settings=settings)
    assert client.get_ticker_directory() == {}
    assert client.ticker_directory_error is not None
    assert "ConnectError" in client.ticker_directory_error
    client.close()


def test_ticker_directory_error_is_none_before_any_fetch(settings):
    client = SecEdgarClient(settings=settings)
    assert client.ticker_directory_error is None
    client.close()
