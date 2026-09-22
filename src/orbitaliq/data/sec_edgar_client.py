"""Real financial data via SEC EDGAR — the U.S. Securities and Exchange
Commission's public filings system.

Every U.S. public company's XBRL-tagged financial statements are free,
keyless government data:

- **Ticker -> CIK mapping**: https://www.sec.gov/files/company_tickers.json
- **XBRL Company Facts**: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  Official docs: https://www.sec.gov/edgar/sec-api-documentation

No account, no API key, ever — SEC's only requirement (its "fair access"
policy, not authentication) is a descriptive `User-Agent` header identifying
the requester, which this client sets from `SEC_EDGAR_USER_AGENT`.

This is exactly the kind of primary-source data real equity-research
analysts and management consultants pull for competitive benchmarking —
revenue growth, R&D investment trend, capital-expenditure trend — straight
from a company's own audited 10-K filings, not a third-party estimate.

Different filers tag the same concept differently in XBRL (e.g. some use
`Revenues`, others `RevenueFromContractWithCustomerExcludingAssessedTax`),
so each metric tries a short list of known-equivalent GAAP tags and uses
whichever the filer actually reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from orbitaliq.config import Settings, get_settings
from orbitaliq.core.data_quality_state import DataQualityState, classify_state
from orbitaliq.core.logging_config import logger
from orbitaliq.utils.geo import deterministic_seed

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)

# Each metric tries these GAAP concept tags in order until one has data.
_REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
_RD_TAGS = ["ResearchAndDevelopmentExpense"]
_CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForCapitalImprovements",
    "PaymentsToAcquireProductiveAssets",
]

# (metric key, candidate tags, calibration: (low_growth_pct, high_growth_pct))
# Calibration maps a YoY growth rate onto a 0-1 index: low_growth_pct -> 0.0,
# high_growth_pct -> 1.0, clipped. Ranges are wider for naturally lumpier
# line items (R&D, capex) than for revenue.
_METRICS = [
    ("revenue_growth_signal", _REVENUE_TAGS, (-0.10, 0.30)),
    ("rd_investment_signal", _RD_TAGS, (-0.20, 0.40)),
    ("capex_growth_signal", _CAPEX_TAGS, (-0.30, 0.50)),
]

# --- Extended GAAP tag registry for the full, enterprise financial profile ---
# (used by fetch_financial_profile / fetch_historical_financials below —
# a superset of the three growth signals above, covering profitability,
# balance-sheet strength, and investment intensity). Every metric here is
# either a genuine SEC XBRL value or is explicitly marked
# INSUFFICIENT_DATA — never a synthetic substitute.
_OPERATING_INCOME_TAGS = ["OperatingIncomeLoss"]
_NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
_GROSS_PROFIT_TAGS = ["GrossProfit"]
_ASSETS_TAGS = ["Assets"]
_ASSETS_CURRENT_TAGS = ["AssetsCurrent"]
_LIABILITIES_CURRENT_TAGS = ["LiabilitiesCurrent"]
_EQUITY_TAGS = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
_CASH_TAGS = ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]
_PPE_TAGS = ["PropertyPlantAndEquipmentNet"]
_CFO_TAGS = ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
# Total debt has no single universally-tagged XBRL concept. This client's
# rule, stated here so it's fully auditable: prefer the sum of
# LongTermDebtCurrent + LongTermDebtNoncurrent for a fiscal year when BOTH
# are reported that year (avoids double-counting); otherwise fall back to
# the single combined LongTermDebt tag for that year. A filer reporting
# neither yields INSUFFICIENT_DATA rather than a guess.
_DEBT_CONCEPT_LABEL = "LongTermDebtCurrent + LongTermDebtNoncurrent (or LongTermDebt)"

# (key, label, category, unit, candidate tags for the "no data at all" case)
_ALL_PROFILE_METRIC_DEFS = [
    ("revenue_growth", "Revenue Growth", "growth", "USD", _REVENUE_TAGS),
    ("rd_growth", "R&D Growth", "growth", "USD", _RD_TAGS),
    ("capex_growth", "CapEx Growth", "growth", "USD", _CAPEX_TAGS),
    ("operating_income_growth", "Operating Income Growth", "growth", "USD", _OPERATING_INCOME_TAGS),
    ("free_cash_flow_trend", "Free Cash Flow Trend", "growth", "USD", _CFO_TAGS),
    ("gross_margin", "Gross Margin", "profitability", "percent", _GROSS_PROFIT_TAGS),
    ("operating_margin", "Operating Margin", "profitability", "percent", _OPERATING_INCOME_TAGS),
    ("net_margin", "Net Margin", "profitability", "percent", _NET_INCOME_TAGS),
    ("roa", "Return on Assets", "profitability", "percent", _NET_INCOME_TAGS),
    ("roe", "Return on Equity", "profitability", "percent", _NET_INCOME_TAGS),
    ("cash", "Cash & Equivalents", "balance_sheet", "USD", _CASH_TAGS),
    ("total_debt", "Total Debt", "balance_sheet", "USD", []),
    ("net_debt", "Net Debt", "balance_sheet", "USD", []),
    ("current_ratio", "Current Ratio", "balance_sheet", "ratio", _ASSETS_CURRENT_TAGS),
    ("debt_to_equity", "Debt / Equity", "balance_sheet", "ratio", []),
    ("capex_to_revenue", "CapEx / Revenue", "investment_intensity", "percent", _CAPEX_TAGS),
    ("rd_to_revenue", "R&D / Revenue", "investment_intensity", "percent", _RD_TAGS),
    ("asset_growth", "Asset Growth", "investment_intensity", "USD", _ASSETS_TAGS),
    ("ppe_growth", "PP&E Growth", "investment_intensity", "USD", _PPE_TAGS),
]


@dataclass
class FinancialSignal:
    """One 0-1 normalized growth-index input to the deterministic scoring
    engine (``core/intelligence_engine.py``). ``value`` is ``None`` exactly
    when ``status == "INSUFFICIENT_DATA"`` -- a live-mode SEC EDGAR fetch
    that failed for this metric. This is the ONLY thing a live-mode
    failure may produce; a fabricated numeric value is never substituted
    (see ``_insufficient_financial_signal`` vs. ``_synthetic_financial_signal``
    below -- the latter is reachable only from ``synthetic_financial_signals()``,
    which is used exclusively by offline demo mode and test fixtures).
    """

    value: float | None  # normalized 0-1 growth index; None iff status == "INSUFFICIENT_DATA"
    live: bool
    source: str
    summary: str
    status: str = "AVAILABLE"  # "AVAILABLE" | "INSUFFICIENT_DATA"
    # The canonical 5-state classification of `source` (see
    # core/data_quality_state.py) -- LIVE / CACHED_REAL / DERIVED /
    # INSUFFICIENT_DATA / ERROR. Always derived from `source` via
    # `classify_state()` at construction time (never set independently),
    # so it can never drift from what `source` actually says. Additive:
    # `status` (AVAILABLE/INSUFFICIENT_DATA) is unchanged and remains what
    # every existing caller checks; `quality_state` is the finer-grained
    # classification on top of it.
    quality_state: str = "LIVE"

    def as_dict(self) -> dict:
        return {
            "value": round(self.value, 4) if self.value is not None else None,
            "live": self.live,
            "source": self.source,
            "summary": self.summary,
            "status": self.status,
            "quality_state": self.quality_state,
        }


@dataclass
class FinancialMetric:
    """A single financial-intelligence line item for the enterprise
    dashboard: always either a genuine SEC-derived value with full
    provenance, or explicitly INSUFFICIENT_DATA. There is no third state —
    this type has no synthetic/fallback source, unlike ``FinancialSignal``.
    """

    key: str
    label: str
    category: str  # "growth" | "profitability" | "balance_sheet" | "investment_intensity"
    unit: str  # "USD" | "percent" | "ratio"
    status: str  # "AVAILABLE" | "INSUFFICIENT_DATA"
    current_value: float | None = None
    prior_value: float | None = None
    yoy_change_pct: float | None = None
    fiscal_period: str | None = None
    prior_fiscal_period: str | None = None
    xbrl_concept: str | None = None
    source: str = "sec_edgar_live"
    note: str | None = None
    # See FinancialSignal.quality_state above -- same contract, always
    # derived from `source` via classify_state() at construction time.
    quality_state: str = "LIVE"

    def as_dict(self) -> dict:
        def _r(v):
            return round(v, 6) if isinstance(v, float) else v

        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "unit": self.unit,
            "status": self.status,
            "current_value": _r(self.current_value),
            "prior_value": _r(self.prior_value),
            "yoy_change_pct": _r(self.yoy_change_pct),
            "fiscal_period": self.fiscal_period,
            "prior_fiscal_period": self.prior_fiscal_period,
            "xbrl_concept": self.xbrl_concept,
            "source": self.source,
            "note": self.note,
            "quality_state": self.quality_state,
        }


@dataclass
class BusinessAddress:
    """A company's real, SEC-registered principal business address, as
    disclosed in its own EDGAR submissions record — never geocoded or
    guessed here (see ``data/geocoding_client.py`` for that separate step).
    ``one_line()`` builds the free-text form the Census Bureau geocoder
    expects.
    """

    street1: str | None
    street2: str | None
    city: str | None
    state_or_country: str | None
    zip_code: str | None
    source_url: str

    def one_line(self) -> str:
        street = " ".join(p for p in (self.street1, self.street2) if p)
        tail = " ".join(p for p in (self.city, self.state_or_country, self.zip_code) if p)
        return ", ".join(p for p in (street, tail) if p)

    def as_dict(self) -> dict:
        return {
            "street1": self.street1,
            "street2": self.street2,
            "city": self.city,
            "state_or_country": self.state_or_country,
            "zip_code": self.zip_code,
            "one_line": self.one_line(),
            "source_url": self.source_url,
        }


@dataclass
class FinancialProfile:
    """The full real-data financial profile for one ticker: every metric in
    ``_ALL_PROFILE_METRIC_DEFS``, each independently AVAILABLE or
    INSUFFICIENT_DATA — used by the enterprise dashboard's Real Financial
    Intelligence view.
    """

    ticker: str
    company_name: str | None
    cik: str | None
    retrieved_at: str
    filing_source_url: str | None
    metrics: dict[str, FinancialMetric] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "cik": self.cik,
            "retrieved_at": self.retrieved_at,
            "filing_source_url": self.filing_source_url,
            "metrics": {k: v.as_dict() for k, v in self.metrics.items()},
        }


@dataclass
class HistoricalFinancials:
    """Real, disclosed-only multi-fiscal-year series for a ticker. Missing
    years or missing line items within a year are simply absent (``None``)
    — never interpolated or fabricated.
    """

    ticker: str
    company_name: str | None
    cik: str | None
    retrieved_at: str
    fiscal_years: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "cik": self.cik,
            "retrieved_at": self.retrieved_at,
            "fiscal_years": self.fiscal_years,
        }


def _by_fy(gaap: dict, tags: list[str]) -> dict[int, float]:
    """Return {fiscal_year: value} merged across EVERY candidate tag this
    filer reported — not just the first tag that happens to have any data
    at all — across ALL annual (10-K, FY) entries, not just the two most
    recent (unlike the legacy ``_annual_series`` used by
    ``fetch_financial_signals``).

    Filers commonly migrate which XBRL concept they tag a line item under
    — most visibly, many switched which revenue tag they use around the
    ASC 606 transition (~FY2018), e.g. from ``Revenues`` to
    ``RevenueFromContractWithCustomerExcludingAssessedTax``. Returning only
    the first candidate tag's years, even when a later candidate tag has
    the filer's actual most recent fiscal years, silently pins every
    metric derived from that series — and any "most recent fiscal period"
    display built from it — to a stale year while other metrics on the
    same assessment correctly track the latest disclosed period. This is
    the same class of bug ``_debt_by_fy`` already guards against for the
    debt tags (see its docstring); fixing it here, in the shared helper
    every profile/historical metric goes through, instead of duplicating
    the merge for each metric that could hit it.

    Within a tag, and across tags for any fiscal year more than one tag
    happens to report, later entries win — matching SEC's own
    "most-recently-filed value for that period" semantics (a later 10-K
    can restate an earlier fiscal year), and, for the rare cross-tag
    overlap, preferring the tag listed later (the more recent reporting
    convention).
    """
    merged: dict[int, float] = {}
    for tag in tags:
        concept = gaap.get(tag)
        if not concept:
            continue
        entries = concept.get("units", {}).get("USD", [])
        annual = {
            e["fy"]: e["val"]
            for e in entries
            if e.get("form") == "10-K" and e.get("fp") == "FY" and isinstance(e.get("val"), (int, float)) and e.get("fy")
        }
        merged.update(annual)
    return merged


def _debt_by_fy(gaap: dict) -> dict[int, float]:
    """Per fiscal year (never as an all-or-nothing choice across years):
    prefer the sum of LongTermDebtCurrent + LongTermDebtNoncurrent when
    BOTH are reported for that specific year, otherwise fall back to the
    single combined LongTermDebt tag reported for that same year. A year
    with neither is simply omitted (never a partial or guessed value).

    This must merge per year, not intersect the split tags' full year sets
    against the single tag's full year sets: a filer that only reported
    the split current/noncurrent pair together in one old fiscal year
    (e.g. only FY2013) while reporting the single combined LongTermDebt
    tag every year since (including the actual latest 10-K) would
    otherwise have every recent year silently discarded, because the old
    code returned ONLY the split-tag intersection whenever it was
    non-empty at all, never merging in the single-tag years alongside it.
    That was a real bug: it made Total Debt / Net Debt / Debt-to-Equity
    freeze on a stale historical fiscal year while every other metric on
    the same filer correctly tracked the latest disclosed period.
    """
    noncurrent = _by_fy(gaap, ["LongTermDebtNoncurrent"])
    current = _by_fy(gaap, ["LongTermDebtCurrent"])
    single = _by_fy(gaap, ["LongTermDebt", "DebtLongtermAndShorttermCombinedAmount"])

    result: dict[int, float] = {}
    for fy in set(noncurrent) | set(current) | set(single):
        if fy in noncurrent and fy in current:
            result[fy] = noncurrent[fy] + current[fy]
        elif fy in single:
            result[fy] = single[fy]
        # else: neither the split pair nor the single tag was reported for
        # this fiscal year -- correctly omitted, never guessed.
    return result


def _growth_from_series(series: dict[int, float]) -> dict:
    years = sorted(series.keys(), reverse=True)
    if len(years) < 2:
        note = "fewer than 2 distinct fiscal years reported" if years else "metric not disclosed by this filer"
        result = {"status": "insufficient", "note": note}
        if years:
            result["current_value"] = series[years[0]]
            result["current_fy"] = years[0]
        return result
    latest_fy, prior_fy = years[0], years[1]
    latest_val, prior_val = series[latest_fy], series[prior_fy]
    if prior_val == 0:
        return {
            "status": "insufficient",
            "note": "prior-year value was zero; growth undefined",
            "current_value": latest_val,
            "current_fy": latest_fy,
        }
    return {
        "status": "available",
        "current_value": latest_val,
        "current_fy": latest_fy,
        "prior_value": prior_val,
        "prior_fy": prior_fy,
        "yoy_change_pct": (latest_val - prior_val) / abs(prior_val),
    }


def _ratio_from_series(numer: dict[int, float], denom: dict[int, float]) -> dict:
    common_years = sorted(set(numer) & set(denom), reverse=True)
    if not common_years:
        return {"status": "insufficient", "note": "numerator and denominator not both disclosed for any common fiscal year"}
    fy = common_years[0]
    if denom[fy] == 0:
        return {"status": "insufficient", "note": f"FY{fy} denominator was zero; ratio undefined"}
    result = {"status": "available", "current_value": numer[fy] / denom[fy], "current_fy": fy}
    if len(common_years) >= 2:
        prior_fy = common_years[1]
        if denom[prior_fy] != 0:
            prior_ratio = numer[prior_fy] / denom[prior_fy]
            result["prior_value"] = prior_ratio
            result["prior_fy"] = prior_fy
            if prior_ratio != 0:
                result["yoy_change_pct"] = (result["current_value"] - prior_ratio) / abs(prior_ratio)
    return result


def _metric_from_growth(key: str, label: str, category: str, series: dict[int, float], tags: list[str]) -> FinancialMetric:
    g = _growth_from_series(series)
    xbrl = tags[0] if tags else None
    if g["status"] != "available":
        source = "insufficient_data"
        return FinancialMetric(
            key=key, label=label, category=category, unit="USD", status="INSUFFICIENT_DATA",
            current_value=g.get("current_value"),
            fiscal_period=(f"FY{g['current_fy']}" if g.get("current_fy") else None),
            xbrl_concept=xbrl, source=source,
            note=f"Insufficient SEC data — {g['note']}",
            quality_state=classify_state(source).value,
        )
    # A directly-disclosed XBRL fact for both fiscal years -- the YoY %
    # is a simple delta presentation of two raw facts, not a computed
    # ratio/margin, so this classifies as LIVE (see _metric_from_ratio
    # below for the DERIVED counterpart).
    source = "sec_edgar_live"
    return FinancialMetric(
        key=key, label=label, category=category, unit="USD", status="AVAILABLE",
        current_value=g["current_value"], prior_value=g["prior_value"], yoy_change_pct=g["yoy_change_pct"],
        fiscal_period=f"FY{g['current_fy']}", prior_fiscal_period=f"FY{g['prior_fy']}",
        xbrl_concept=xbrl, source=source,
        quality_state=classify_state(source).value,
    )


def _metric_from_ratio(
    key: str, label: str, category: str, numer: dict[int, float], denom: dict[int, float], unit: str, xbrl_concept: str
) -> FinancialMetric:
    r = _ratio_from_series(numer, denom)
    if r["status"] != "available":
        source = "insufficient_data"
        return FinancialMetric(
            key=key, label=label, category=category, unit=unit, status="INSUFFICIENT_DATA",
            xbrl_concept=xbrl_concept, source=source,
            note=f"Insufficient SEC data — {r['note']}",
            quality_state=classify_state(source).value,
        )
    # A ratio/margin computed from two directly-disclosed XBRL facts --
    # DERIVED, not itself a directly-disclosed fact (see
    # DataQualityState.DERIVED in core/data_quality_state.py).
    source = "derived:sec_edgar"
    return FinancialMetric(
        key=key, label=label, category=category, unit=unit, status="AVAILABLE",
        current_value=r["current_value"], prior_value=r.get("prior_value"), yoy_change_pct=r.get("yoy_change_pct"),
        fiscal_period=f"FY{r['current_fy']}",
        prior_fiscal_period=(f"FY{r['prior_fy']}" if r.get("prior_fy") else None),
        xbrl_concept=xbrl_concept, source=source,
        quality_state=classify_state(source).value,
    )


def _metric_from_point(key: str, label: str, category: str, series: dict[int, float], unit: str, xbrl_concept: str) -> FinancialMetric:
    years = sorted(series.keys(), reverse=True)
    if not years:
        source = "insufficient_data"
        return FinancialMetric(
            key=key, label=label, category=category, unit=unit, status="INSUFFICIENT_DATA",
            xbrl_concept=xbrl_concept, source=source,
            note="Insufficient SEC data — metric not disclosed by this filer",
            quality_state=classify_state(source).value,
        )
    fy = years[0]
    source = "sec_edgar_live"
    metric = FinancialMetric(
        key=key, label=label, category=category, unit=unit, status="AVAILABLE",
        current_value=series[fy], fiscal_period=f"FY{fy}", xbrl_concept=xbrl_concept, source=source,
        quality_state=classify_state(source).value,
    )
    if len(years) >= 2:
        prior_fy = years[1]
        prior_val = series[prior_fy]
        metric.prior_value = prior_val
        metric.prior_fiscal_period = f"FY{prior_fy}"
        if prior_val != 0:
            metric.yoy_change_pct = (metric.current_value - prior_val) / abs(prior_val)
    return metric


def _fcf_metric(cfo: dict[int, float], capex: dict[int, float]) -> FinancialMetric:
    common_years = sorted(set(cfo) & set(capex), reverse=True)
    xbrl = "NetCashProvidedByUsedInOperatingActivities - CapEx"
    if not common_years:
        source = "insufficient_data"
        return FinancialMetric(
            key="free_cash_flow_trend", label="Free Cash Flow Trend", category="growth", unit="USD",
            status="INSUFFICIENT_DATA", xbrl_concept=xbrl, source=source,
            note="Insufficient SEC data — operating cash flow and/or capex not both disclosed for a common fiscal year",
            quality_state=classify_state(source).value,
        )
    fy = common_years[0]
    fcf_current = cfo[fy] - capex[fy]
    # Computed as CFO - CapEx from two directly-disclosed facts -- DERIVED.
    source = "derived:sec_edgar"
    metric = FinancialMetric(
        key="free_cash_flow_trend", label="Free Cash Flow Trend", category="growth", unit="USD",
        status="AVAILABLE", current_value=fcf_current, fiscal_period=f"FY{fy}", xbrl_concept=xbrl, source=source,
        quality_state=classify_state(source).value,
    )
    if len(common_years) >= 2:
        prior_fy = common_years[1]
        fcf_prior = cfo[prior_fy] - capex[prior_fy]
        metric.prior_value = fcf_prior
        metric.prior_fiscal_period = f"FY{prior_fy}"
        if fcf_prior != 0:
            metric.yoy_change_pct = (fcf_current - fcf_prior) / abs(fcf_prior)
    return metric


def _net_debt_metric(debt: dict[int, float], cash: dict[int, float]) -> FinancialMetric:
    common_years = sorted(set(debt) & set(cash), reverse=True)
    xbrl = "Total Debt - Cash & Equivalents"
    if not common_years:
        source = "insufficient_data"
        return FinancialMetric(
            key="net_debt", label="Net Debt", category="balance_sheet", unit="USD",
            status="INSUFFICIENT_DATA", xbrl_concept=xbrl, source=source,
            note="Insufficient SEC data — total debt and/or cash not both disclosed for a common fiscal year",
            quality_state=classify_state(source).value,
        )
    fy = common_years[0]
    # Computed as Total Debt - Cash from two directly-disclosed facts -- DERIVED.
    source = "derived:sec_edgar"
    metric = FinancialMetric(
        key="net_debt", label="Net Debt", category="balance_sheet", unit="USD", status="AVAILABLE",
        current_value=debt[fy] - cash[fy], fiscal_period=f"FY{fy}", xbrl_concept=xbrl, source=source,
        quality_state=classify_state(source).value,
    )
    if len(common_years) >= 2:
        prior_fy = common_years[1]
        metric.prior_value = debt[prior_fy] - cash[prior_fy]
        metric.prior_fiscal_period = f"FY{prior_fy}"
    return metric


def _all_metrics_insufficient(reason: str) -> dict[str, FinancialMetric]:
    source = "insufficient_data"
    return {
        key: FinancialMetric(
            key=key, label=label, category=category, unit=unit, status="INSUFFICIENT_DATA",
            xbrl_concept=(tags[0] if tags else None), source=source,
            note=f"Insufficient SEC data — {reason}",
            quality_state=classify_state(source).value,
        )
        for key, label, category, unit, tags in _ALL_PROFILE_METRIC_DEFS
    }


def _all_metrics_error(reason: str, *, error_type: str) -> dict[str, FinancialMetric]:
    """Sibling of ``_all_metrics_insufficient`` for the *unexpected
    exception* case (see ``DataQualityState.ERROR``) -- used when the
    ticker-directory or company-facts fetch itself raised, rather than
    cleanly resolving to "not found"/"not disclosed."
    """
    source = f"error:sec_edgar:{error_type}"
    return {
        key: FinancialMetric(
            key=key, label=label, category=category, unit=unit, status="INSUFFICIENT_DATA",
            xbrl_concept=(tags[0] if tags else None), source=source,
            note=f"SEC EDGAR fetch error — {reason}",
            quality_state=classify_state(source).value,
        )
        for key, label, category, unit, tags in _ALL_PROFILE_METRIC_DEFS
    }


def _synthetic_financial_signal(ticker: str, metric: str, reason: str) -> FinancialSignal:
    """Deterministic per-(ticker, metric) fabricated value -- reproducible,
    clearly labeled ``synthetic_fallback:*``, and reachable ONLY from
    ``synthetic_financial_signals()`` below (the offline-demo-mode /
    test-fixture path -- see ``IngestionAgent.run()``'s
    ``ORBITALIQ_LIVE_DATA_MODE=false`` branch). A live-mode SEC EDGAR
    failure must never call this function -- see
    ``_insufficient_financial_signal`` immediately below, which is what
    ``fetch_financial_signals`` actually calls on failure.
    """
    rng_seed = deterministic_seed(ticker.upper(), metric)
    value = (rng_seed % 1000) / 1000.0
    logger.warning(f"SEC EDGAR fallback for '{metric}' ({ticker}): {reason}")
    source = f"synthetic_fallback:{metric}"
    # quality_state deliberately INSUFFICIENT_DATA (not LIVE) even though
    # status="AVAILABLE" (a placeholder numeric value exists) -- honestly
    # reflects that this is not a real production value; see
    # classify_state()'s treatment of the synthetic_fallback:* prefix.
    return FinancialSignal(
        value=value, live=False, source=source, summary=reason, status="AVAILABLE",
        quality_state=classify_state(source).value,
    )


def _insufficient_financial_signal(ticker: str, metric: str, reason: str) -> FinancialSignal:
    """A live-mode SEC EDGAR fetch found a documented, expected reason no
    value is available for this metric (ticker not in the directory, no
    usable XBRL tag, fewer than two comparable fiscal years, etc). Returns
    a typed ``value=None`` / ``status="INSUFFICIENT_DATA"`` result -- never
    a fabricated number. This is what every *documented-gap* failure path
    in ``fetch_financial_signals`` / ``_compute_growth_signal`` returns;
    see ``_error_financial_signal`` below for the sibling path used when
    the failure was an *unexpected* exception rather than a clean, known
    gap. ``core/intelligence_engine.py::compute_momentum_assessment()``
    guards every multiply against this ``None`` explicitly rather than
    treating it as zero.
    """
    logger.warning(f"SEC EDGAR insufficient data for '{metric}' ({ticker}): {reason}")
    source = "insufficient_data"
    return FinancialSignal(
        value=None, live=False, source=source, summary=reason, status="INSUFFICIENT_DATA",
        quality_state=classify_state(source).value,
    )


def _error_financial_signal(ticker: str, metric: str, reason: str, *, error_type: str) -> FinancialSignal:
    """A live-mode SEC EDGAR fetch failed with an *unexpected* exception
    (network failure, malformed response, JSON decode error) rather than a
    clean, documented gap -- see ``DataQualityState.ERROR`` in
    ``core/data_quality_state.py``. Still returns a typed ``value=None`` /
    ``status="INSUFFICIENT_DATA"`` result -- never a fabricated number --
    but the finer-grained ``quality_state``/``source`` disclose that this
    was an error, not a documented absence, so an operator investigating a
    Data Quality Center report can tell "SEC EDGAR doesn't have this" apart
    from "our fetch broke and should be retried/investigated."
    """
    logger.warning(f"SEC EDGAR ERROR for '{metric}' ({ticker}): {reason}")
    source = f"error:sec_edgar:{error_type}"
    return FinancialSignal(
        value=None, live=False, source=source, summary=reason, status="INSUFFICIENT_DATA",
        quality_state=classify_state(source).value,
    )


def _growth_index(growth_pct: float, low: float, high: float) -> float:
    return max(0.0, min((growth_pct - low) / (high - low), 1.0))


def synthetic_financial_signals(ticker: str) -> dict[str, FinancialSignal]:
    """All three metrics via the deterministic synthetic fallback, with no
    network attempt at all. Used by the ingestion agent's offline demo mode
    (``ORBITALIQ_LIVE_DATA_MODE=false``) so the test suite/CI never depends
    on network availability, reusing the exact same deterministic
    per-(ticker, metric) generator as a live client's partial-outage
    fallback.
    """
    return {metric: _synthetic_financial_signal(ticker, metric, "offline demo mode") for metric, _, _ in _METRICS}


class SecEdgarClient:
    def __init__(self, settings: Settings | None = None, http_client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=self._settings.live_data_timeout_seconds,
            headers={"User-Agent": self._settings.sec_edgar_user_agent},
        )
        self._ticker_cik_cache: dict[str, dict] | None = None
        # Set only when the ticker-directory fetch itself raised an
        # unexpected exception (network failure, bad JSON, etc) rather than
        # cleanly succeeding -- lets callers distinguish "ticker genuinely
        # not in a successfully-fetched directory" (INSUFFICIENT_DATA) from
        # "we couldn't even fetch the directory" (ERROR). See
        # resolve_cik()/fetch_financial_signals()/fetch_financial_profile().
        self._ticker_directory_error: str | None = None
        # Memoize the (large) company-facts JSON per CIK for the lifetime
        # of this client instance. fetch_financial_signals (the legacy
        # scoring-engine input) and fetch_financial_profile /
        # fetch_historical_financials (the enterprise dashboard's
        # real-data views) are frequently called back-to-back for the
        # same ticker within one assessment; without this cache that would
        # mean two identical live HTTP calls to data.sec.gov for no
        # reason, and — more importantly — a tiny risk of the two calls
        # seeing different data if SEC re-published between them. Caching
        # guarantees every view within one assessment is built from the
        # exact same underlying filing snapshot.
        self._company_facts_cache: dict[str, dict | None] = {}
        # cik -> "ExceptionClassName: message" whenever fetch_company_facts
        # hit an unexpected exception for that CIK (the only reason it ever
        # caches None for a CIK) -- see the ERROR-vs-INSUFFICIENT_DATA note
        # on _ticker_directory_error above; same distinction, one level down.
        self._company_facts_error: dict[str, str] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def _get(self, url: str) -> httpx.Response:
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def resolve_cik(self, ticker: str) -> tuple[str | None, str | None]:
        """Return (10-digit zero-padded CIK, company title) for a ticker, or (None, None)."""
        if self._ticker_cik_cache is None:
            try:
                response = self._get(f"{self._settings.sec_edgar_base_url}/files/company_tickers.json")
                raw = response.json()
                self._ticker_cik_cache = {
                    str(row["ticker"]).upper(): row for row in raw.values() if isinstance(row, dict) and "ticker" in row
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"SEC EDGAR ticker directory unavailable: {exc}")
                self._ticker_cik_cache = {}
                self._ticker_directory_error = f"{type(exc).__name__}: {exc}"

        row = self._ticker_cik_cache.get(ticker.upper())
        if not row:
            return None, None
        return f"{int(row['cik_str']):010d}", row.get("title")

    @property
    def ticker_directory_error(self) -> str | None:
        """Exposes ``_ticker_directory_error`` read-only for
        ``data/company_resolver.py``, so a free-text company-name search
        can tell "the SEC directory itself failed to fetch" (a real error)
        apart from "the directory loaded fine and genuinely has no match
        for this query" (see ``get_ticker_directory`` below).
        """
        return self._ticker_directory_error

    def get_ticker_directory(self) -> dict[str, dict]:
        """The full, cached SEC ticker -> ``{ticker, cik_str, title}``
        directory ``resolve_cik`` uses internally, exposed read-only for
        ``data/company_resolver.py``'s free-text company-name search.
        Triggers the same lazy fetch/cache ``resolve_cik`` does if the
        directory hasn't been loaded yet in this client instance. Returns
        ``{}`` if the directory genuinely couldn't be fetched (see
        ``ticker_directory_error``) — never partial or fabricated data.
        """
        if self._ticker_cik_cache is None:
            # Triggers the same lazy fetch/cache resolve_cik() performs;
            # this placeholder ticker is never expected to match a real
            # row, it's only here for the fetch side effect.
            self.resolve_cik("__orbitaliq_directory_warm__")
        return dict(self._ticker_cik_cache or {})

    def fetch_company_facts(self, cik: str) -> dict | None:
        if cik in self._company_facts_cache:
            return self._company_facts_cache[cik]

        url = f"{self._settings.sec_edgar_data_base_url}/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            response = self._get(url)
            facts = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SEC EDGAR company facts unavailable for CIK{cik}: {exc}")
            facts = None
            self._company_facts_error[cik] = f"{type(exc).__name__}: {exc}"

        self._company_facts_cache[cik] = facts
        return facts

    def fetch_business_address(self, cik: str) -> BusinessAddress | None:
        """The company's real, SEC-registered principal business address
        from its EDGAR submissions record
        (``https://data.sec.gov/submissions/CIK##########.json``) — never
        wired into any other method on this client, only used by
        ``data/company_resolver.py`` as the sole authoritative input to
        geocoding a company's facility coordinates for the "resolve by
        company name" workflow. Returns ``None`` on any failure (not
        disclosed by this filer, malformed response, or a network error) —
        never a guessed or partial address; the caller reports this
        honestly as insufficient rather than falling back to any other
        source.
        """
        url = f"{self._settings.sec_edgar_data_base_url}/submissions/CIK{cik}.json"
        try:
            response = self._get(url)
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SEC EDGAR submissions/business-address unavailable for CIK{cik}: {exc}")
            return None

        addresses = payload.get("addresses") or {}
        # Prefer the registered business address; a mailing address is a
        # real, disclosed fallback only when no business address exists —
        # never a synthetic substitute, just a different real SEC-disclosed
        # field on the same filer.
        raw = addresses.get("business") or addresses.get("mailing")
        if not raw or not isinstance(raw, dict):
            return None

        address = BusinessAddress(
            street1=raw.get("street1"),
            street2=raw.get("street2"),
            city=raw.get("city"),
            state_or_country=raw.get("stateOrCountry") or raw.get("stateOrCountryDescription"),
            zip_code=raw.get("zipCode"),
            source_url=url,
        )
        if not address.one_line():
            return None
        return address

    def fetch_financial_signals(self, ticker: str) -> dict[str, FinancialSignal]:
        cik, title = self.resolve_cik(ticker)
        if cik is None:
            if self._ticker_directory_error:
                return {
                    metric: _error_financial_signal(
                        ticker, metric,
                        f"SEC EDGAR ticker directory fetch failed: {self._ticker_directory_error}",
                        error_type=self._ticker_directory_error.split(":", 1)[0],
                    )
                    for metric, _, _ in _METRICS
                }
            return {
                metric: _insufficient_financial_signal(ticker, metric, f"ticker '{ticker}' not found in SEC EDGAR directory")
                for metric, _, _ in _METRICS
            }

        facts = self.fetch_company_facts(cik)
        if facts is None:
            facts_error = self._company_facts_error.get(cik)
            if facts_error:
                return {
                    metric: _error_financial_signal(
                        ticker, metric,
                        f"SEC EDGAR company facts request failed for CIK{cik}: {facts_error}",
                        error_type=facts_error.split(":", 1)[0],
                    )
                    for metric, _, _ in _METRICS
                }
            return {
                metric: _insufficient_financial_signal(ticker, metric, f"SEC EDGAR company facts request failed for CIK{cik}")
                for metric, _, _ in _METRICS
            }

        gaap = facts.get("facts", {}).get("us-gaap", {})
        results: dict[str, FinancialSignal] = {}
        for metric, tags, (low, high) in _METRICS:
            results[metric] = self._compute_growth_signal(ticker, metric, gaap, tags, low, high)
        return results

    @staticmethod
    def _annual_series(gaap: dict, tags: list[str]) -> list[dict] | None:
        for tag in tags:
            concept = gaap.get(tag)
            if not concept:
                continue
            usd_entries = concept.get("units", {}).get("USD", [])
            annual = [
                e
                for e in usd_entries
                if e.get("form") == "10-K" and e.get("fp") == "FY" and isinstance(e.get("val"), (int, float)) and e.get("fy")
            ]
            if len(annual) >= 2:
                return annual
        return None

    def _compute_growth_signal(
        self, ticker: str, metric: str, gaap: dict, tags: list[str], low: float, high: float
    ) -> FinancialSignal:
        annual = self._annual_series(gaap, tags)
        if not annual:
            return _insufficient_financial_signal(ticker, metric, f"no usable XBRL tag among {tags} for this filer")

        by_fy: dict[int, float] = {}
        for entry in annual:
            by_fy[entry["fy"]] = entry["val"]  # last entry per fiscal year wins (latest-filed restatement)
        years = sorted(by_fy.keys(), reverse=True)
        if len(years) < 2:
            return _insufficient_financial_signal(ticker, metric, "fewer than 2 distinct fiscal years reported")

        latest_fy, prior_fy = years[0], years[1]
        latest_val, prior_val = by_fy[latest_fy], by_fy[prior_fy]
        if prior_val == 0:
            return _insufficient_financial_signal(ticker, metric, f"prior-year {metric} value was zero; growth undefined")

        growth_pct = (latest_val - prior_val) / abs(prior_val)
        index = _growth_index(growth_pct, low, high)

        source = "sec_edgar_live"
        return FinancialSignal(
            value=index,
            live=True,
            source=source,
            summary=(
                f"FY{latest_fy} vs FY{prior_fy}: {growth_pct * 100:+.1f}% YoY "
                f"(${latest_val:,.0f} vs ${prior_val:,.0f})"
            ),
            status="AVAILABLE",
            quality_state=classify_state(source).value,
        )

    def fetch_financial_profile(self, ticker: str) -> FinancialProfile:
        """The full enterprise financial profile: growth, profitability,
        balance-sheet, and investment-intensity metrics, each independently
        AVAILABLE (from a genuine SEC filing) or INSUFFICIENT_DATA. Unlike
        ``fetch_financial_signals`` (used by the deterministic scoring
        engine and preserved unchanged for backward compatibility), this
        method has **no synthetic fallback whatsoever** — it is the
        strict, real-data-only source for the enterprise dashboard's
        Real Financial Intelligence view.
        """
        cik, title = self.resolve_cik(ticker)
        retrieved_at = datetime.now(timezone.utc).isoformat()

        if cik is None:
            if self._ticker_directory_error:
                return FinancialProfile(
                    ticker=ticker.upper(), company_name=None, cik=None, retrieved_at=retrieved_at,
                    filing_source_url=None,
                    metrics=_all_metrics_error(
                        f"SEC EDGAR ticker directory fetch failed: {self._ticker_directory_error}",
                        error_type=self._ticker_directory_error.split(":", 1)[0],
                    ),
                )
            return FinancialProfile(
                ticker=ticker.upper(), company_name=None, cik=None, retrieved_at=retrieved_at,
                filing_source_url=None,
                metrics=_all_metrics_insufficient(f"ticker '{ticker}' not found in SEC EDGAR company directory"),
            )

        filing_url = f"{self._settings.sec_edgar_data_base_url}/api/xbrl/companyfacts/CIK{cik}.json"
        facts = self.fetch_company_facts(cik)
        if facts is None:
            facts_error = self._company_facts_error.get(cik)
            if facts_error:
                return FinancialProfile(
                    ticker=ticker.upper(), company_name=title, cik=cik, retrieved_at=retrieved_at,
                    filing_source_url=filing_url,
                    metrics=_all_metrics_error(
                        f"SEC EDGAR company facts request failed for CIK{cik}: {facts_error}",
                        error_type=facts_error.split(":", 1)[0],
                    ),
                )
            return FinancialProfile(
                ticker=ticker.upper(), company_name=title, cik=cik, retrieved_at=retrieved_at,
                filing_source_url=filing_url,
                metrics=_all_metrics_insufficient(f"SEC EDGAR company facts request failed for CIK{cik}"),
            )

        gaap = facts.get("facts", {}).get("us-gaap", {})
        revenue = _by_fy(gaap, _REVENUE_TAGS)
        rd = _by_fy(gaap, _RD_TAGS)
        capex = _by_fy(gaap, _CAPEX_TAGS)
        operating_income = _by_fy(gaap, _OPERATING_INCOME_TAGS)
        net_income = _by_fy(gaap, _NET_INCOME_TAGS)
        gross_profit = _by_fy(gaap, _GROSS_PROFIT_TAGS)
        assets = _by_fy(gaap, _ASSETS_TAGS)
        assets_current = _by_fy(gaap, _ASSETS_CURRENT_TAGS)
        liabilities_current = _by_fy(gaap, _LIABILITIES_CURRENT_TAGS)
        equity = _by_fy(gaap, _EQUITY_TAGS)
        cash = _by_fy(gaap, _CASH_TAGS)
        debt = _debt_by_fy(gaap)
        ppe = _by_fy(gaap, _PPE_TAGS)
        cfo = _by_fy(gaap, _CFO_TAGS)

        metrics: dict[str, FinancialMetric] = {
            "revenue_growth": _metric_from_growth("revenue_growth", "Revenue Growth", "growth", revenue, _REVENUE_TAGS),
            "rd_growth": _metric_from_growth("rd_growth", "R&D Growth", "growth", rd, _RD_TAGS),
            "capex_growth": _metric_from_growth("capex_growth", "CapEx Growth", "growth", capex, _CAPEX_TAGS),
            "operating_income_growth": _metric_from_growth(
                "operating_income_growth", "Operating Income Growth", "growth", operating_income, _OPERATING_INCOME_TAGS
            ),
            "free_cash_flow_trend": _fcf_metric(cfo, capex),
            "gross_margin": _metric_from_ratio(
                "gross_margin", "Gross Margin", "profitability", gross_profit, revenue, "percent", "GrossProfit / Revenues"
            ),
            "operating_margin": _metric_from_ratio(
                "operating_margin", "Operating Margin", "profitability", operating_income, revenue, "percent",
                "OperatingIncomeLoss / Revenues",
            ),
            "net_margin": _metric_from_ratio(
                "net_margin", "Net Margin", "profitability", net_income, revenue, "percent", "NetIncomeLoss / Revenues"
            ),
            "roa": _metric_from_ratio(
                "roa", "Return on Assets", "profitability", net_income, assets, "percent", "NetIncomeLoss / Assets"
            ),
            "roe": _metric_from_ratio(
                "roe", "Return on Equity", "profitability", net_income, equity, "percent", "NetIncomeLoss / StockholdersEquity"
            ),
            "cash": _metric_from_point("cash", "Cash & Equivalents", "balance_sheet", cash, "USD", _CASH_TAGS[0]),
            "total_debt": _metric_from_point("total_debt", "Total Debt", "balance_sheet", debt, "USD", _DEBT_CONCEPT_LABEL),
            "net_debt": _net_debt_metric(debt, cash),
            "current_ratio": _metric_from_ratio(
                "current_ratio", "Current Ratio", "balance_sheet", assets_current, liabilities_current, "ratio",
                "AssetsCurrent / LiabilitiesCurrent",
            ),
            "debt_to_equity": _metric_from_ratio(
                "debt_to_equity", "Debt / Equity", "balance_sheet", debt, equity, "ratio",
                f"{_DEBT_CONCEPT_LABEL} / StockholdersEquity",
            ),
            "capex_to_revenue": _metric_from_ratio(
                "capex_to_revenue", "CapEx / Revenue", "investment_intensity", capex, revenue, "percent", "CapEx / Revenues"
            ),
            "rd_to_revenue": _metric_from_ratio(
                "rd_to_revenue", "R&D / Revenue", "investment_intensity", rd, revenue, "percent",
                "ResearchAndDevelopmentExpense / Revenues",
            ),
            "asset_growth": _metric_from_growth("asset_growth", "Asset Growth", "investment_intensity", assets, _ASSETS_TAGS),
            "ppe_growth": _metric_from_growth("ppe_growth", "PP&E Growth", "investment_intensity", ppe, _PPE_TAGS),
        }

        return FinancialProfile(
            ticker=ticker.upper(), company_name=title, cik=cik, retrieved_at=retrieved_at,
            filing_source_url=filing_url, metrics=metrics,
        )

    def fetch_historical_financials(self, ticker: str, max_years: int = 6) -> HistoricalFinancials:
        """Real, disclosed-only multi-fiscal-year series for the Historical
        Analysis view. Only fiscal years and line items the filer actually
        reported appear — no interpolation, no fabricated years.
        """
        cik, title = self.resolve_cik(ticker)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        if cik is None:
            return HistoricalFinancials(ticker=ticker.upper(), company_name=None, cik=None, retrieved_at=retrieved_at)

        facts = self.fetch_company_facts(cik)
        if facts is None:
            return HistoricalFinancials(ticker=ticker.upper(), company_name=title, cik=cik, retrieved_at=retrieved_at)

        gaap = facts.get("facts", {}).get("us-gaap", {})
        revenue = _by_fy(gaap, _REVENUE_TAGS)
        rd = _by_fy(gaap, _RD_TAGS)
        capex = _by_fy(gaap, _CAPEX_TAGS)
        gross_profit = _by_fy(gaap, _GROSS_PROFIT_TAGS)
        operating_income = _by_fy(gaap, _OPERATING_INCOME_TAGS)
        net_income = _by_fy(gaap, _NET_INCOME_TAGS)
        assets = _by_fy(gaap, _ASSETS_TAGS)

        candidate_years = sorted(set(revenue) | set(rd) | set(capex) | set(assets), reverse=True)[:max_years]
        fiscal_years = []
        for fy in sorted(candidate_years):
            rev = revenue.get(fy)
            fiscal_years.append(
                {
                    "fiscal_year": fy,
                    "revenue": revenue.get(fy),
                    "rd": rd.get(fy),
                    "capex": capex.get(fy),
                    "assets": assets.get(fy),
                    "gross_margin": (gross_profit[fy] / rev) if fy in gross_profit and rev else None,
                    "operating_margin": (operating_income[fy] / rev) if fy in operating_income and rev else None,
                    "net_margin": (net_income[fy] / rev) if fy in net_income and rev else None,
                }
            )

        return HistoricalFinancials(
            ticker=ticker.upper(), company_name=title, cik=cik, retrieved_at=retrieved_at, fiscal_years=fiscal_years
        )
