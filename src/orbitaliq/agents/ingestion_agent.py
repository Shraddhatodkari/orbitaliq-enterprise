"""Ingestion Agent — gathers financial-growth signals and a bi-temporal
satellite tile pair for a company's facility.

When ``ORBITALIQ_LIVE_DATA_MODE`` is true (the default), every signal is
fetched from a genuine public data source:

- revenue / R&D / capex growth  <- SEC EDGAR XBRL company facts (live, keyless)
- satellite tile pair (current + prior)  <- NASA GIBS real VIIRS/MODIS imagery (live, keyless)

Each source degrades independently and transparently to a typed
``value=None`` / ``status="INSUFFICIENT_DATA"`` result when it can't
produce usable live data -- never a fabricated substitute (see
``data/sec_edgar_client.py`` and ``data/imagery_provider.py``). The agent
always returns a per-signal ``data_sources`` provenance map alongside the
financial signals and tile pair, so every assessment can honestly disclose
exactly what was live.

Setting ``ORBITALIQ_LIVE_DATA_MODE=false`` skips all network calls and uses
the deterministic synthetic generators directly — this is what the test
suite and CI use, so they never depend on third-party API availability.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from orbitaliq.agents.types import CompanyInput
from orbitaliq.config import Settings, get_settings
from orbitaliq.core.data_quality_state import classify_state
from orbitaliq.core.logging_config import logger
from orbitaliq.data.imagery_provider import ChangeTilePairResult, NasaGibsImageryProvider
from orbitaliq.data.repository import lookup_cached_financial_value
from orbitaliq.data.sec_edgar_client import FinancialProfile, FinancialSignal, SecEdgarClient, synthetic_financial_signals
from orbitaliq.data.synthetic_data import generate_satellite_tile
from orbitaliq.utils.geo import validate_lat_lon

OFFLINE_SOURCE_LABELS = {
    "revenue_growth_signal": "offline_demo_mode",
    "rd_investment_signal": "offline_demo_mode",
    "capex_growth_signal": "offline_demo_mode",
    "satellite_imagery_current": "offline_demo_mode",
    "satellite_imagery_prior": "offline_demo_mode",
}

# Used for both satellite data_sources keys when a company/assessment has
# no verified facility coordinates at all (see data/company_resolver.py) --
# satellite evidence was never attempted, so this is a documented,
# INSUFFICIENT_DATA-classifying gap (core/data_quality_state.py), never a
# live-mode failure and never a guessed tile.
_NO_FACILITY_SOURCE = "insufficient_data:no_verified_facility_coordinates"


class IngestionAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        sec_client: SecEdgarClient | None = None,
        imagery_provider: NasaGibsImageryProvider | None = None,
        cache_lookup: Callable[[str, str], tuple[float, str] | None] | None = lookup_cached_financial_value,
    ) -> None:
        self._settings = settings or get_settings()
        self._sec = sec_client or SecEdgarClient(self._settings)
        self._imagery = imagery_provider or NasaGibsImageryProvider(self._settings)
        # Genuine CACHED_REAL support (core/data_quality_state.py): when a
        # live financial-signal fetch fails, this callable is tried before
        # giving up and reporting INSUFFICIENT_DATA/ERROR -- see
        # _apply_cached_real_fallback() below. Defaults to the real,
        # repository-backed lookup (data/repository.py) so production
        # assessments get this "for free"; tests pass a fake or None to
        # control/disable it explicitly. Never raises -- a lookup failure
        # degrades to "no cached value found", never blocks an assessment.
        self._cache_lookup = cache_lookup
        # Populated by run() in live mode so fetch_enriched_evidence() can
        # reuse the exact same bi-temporal tile pair (with its full
        # observation-date/resolution/coordinates metadata) for the
        # dashboard's Satellite Change Detection View, instead of issuing a
        # second, redundant round of NASA GIBS requests for the same
        # assessment. None in offline demo mode, since no ImageryResult
        # objects are produced there.
        self._last_change_pair: ChangeTilePairResult | None = None

    def close(self) -> None:
        for client in (self._sec, self._imagery):
            try:
                client.close()
            except Exception:  # noqa: BLE001 - closing must never blow up shutdown
                pass

    def run(self, company: CompanyInput) -> tuple[dict[str, FinancialSignal], np.ndarray | None, np.ndarray | None, dict]:
        has_coordinates = company.latitude is not None and company.longitude is not None
        if has_coordinates:
            validate_lat_lon(company.latitude, company.longitude)
        lat, lon = company.latitude, company.longitude

        if not self._settings.orbitaliq_live_data_mode:
            logger.debug(f"IngestionAgent: offline demo mode for {company.ticker} / {company.facility_name}")
            financial_signals = synthetic_financial_signals(company.ticker)
            self._last_change_pair = None
            if not has_coordinates:
                return financial_signals, None, None, {**OFFLINE_SOURCE_LABELS, **self._no_facility_satellite_sources()}
            current_tile = generate_satellite_tile(lat, lon, seed_salt="current_offline")
            prior_tile = generate_satellite_tile(lat, lon, seed_salt="prior_offline")
            return financial_signals, current_tile, prior_tile, dict(OFFLINE_SOURCE_LABELS)

        logger.debug(
            f"IngestionAgent: fetching SEC EDGAR financials"
            f"{' + NASA GIBS change pair' if has_coordinates else ' (no verified facility coordinates -- satellite skipped)'} "
            f"for {company.ticker} / {company.facility_name}"
        )

        financial_signals = self._sec.fetch_financial_signals(company.ticker)
        financial_signals = self._apply_cached_real_fallback(company.ticker, financial_signals)

        if not has_coordinates:
            # No verified facility location exists for this assessment
            # (see data/company_resolver.py) -- satellite/vision evidence
            # is never attempted, and NEVER stands in with a guessed
            # coordinate or a synthetic tile. This is a documented,
            # INSUFFICIENT_DATA-classifying gap, not a live-mode failure.
            self._last_change_pair = None
            data_sources = {
                "revenue_growth_signal": financial_signals["revenue_growth_signal"].source,
                "rd_investment_signal": financial_signals["rd_investment_signal"].source,
                "capex_growth_signal": financial_signals["capex_growth_signal"].source,
                **self._no_facility_satellite_sources(),
            }
            live_count = sum(1 for v in data_sources.values() if str(v).endswith("_live"))
            logger.info(
                f"IngestionAgent: {live_count}/{len(data_sources)} signals live for {company.ticker} "
                "(satellite not attempted -- no verified facility coordinates)"
            )
            return financial_signals, None, None, data_sources

        pair = self._imagery.fetch_change_pair(lat, lon)
        self._last_change_pair = pair

        data_sources = {
            "revenue_growth_signal": financial_signals["revenue_growth_signal"].source,
            "rd_investment_signal": financial_signals["rd_investment_signal"].source,
            "capex_growth_signal": financial_signals["capex_growth_signal"].source,
            "satellite_imagery_current": pair.current.source,
            "satellite_imagery_prior": pair.prior.source,
        }

        live_count = sum(1 for v in data_sources.values() if v.endswith("_live"))
        logger.info(
            f"IngestionAgent: {live_count}/{len(data_sources)} signals live for {company.ticker} "
            f"({pair.summary()})"
        )

        return financial_signals, pair.current.tile, pair.prior.tile, data_sources

    @staticmethod
    def _no_facility_satellite_sources() -> dict[str, str]:
        return {
            "satellite_imagery_current": _NO_FACILITY_SOURCE,
            "satellite_imagery_prior": _NO_FACILITY_SOURCE,
        }

    def _apply_cached_real_fallback(
        self, ticker: str, financial_signals: dict[str, FinancialSignal]
    ) -> dict[str, FinancialSignal]:
        """For every signal whose live fetch genuinely failed
        (``status == "INSUFFICIENT_DATA"``), try ``self._cache_lookup`` for
        a genuine, previously-observed value for this exact
        (ticker, signal) before accepting the gap -- see
        ``DataQualityState.CACHED_REAL`` in core/data_quality_state.py.

        Never touches a signal that already succeeded live, and never
        touches the disclosed offline-demo/synthetic-fallback path (those
        keep ``status == "AVAILABLE"`` by construction -- see
        ``data/sec_edgar_client.py`` -- so they never reach this method's
        replacement branch at all). A cache miss (or no cache_lookup
        configured) leaves the original INSUFFICIENT_DATA/ERROR result
        untouched -- this is a strictly additive enrichment, never a new
        way to lose information.
        """
        if self._cache_lookup is None:
            return financial_signals

        updated = dict(financial_signals)
        for key, signal in financial_signals.items():
            if signal.status != "INSUFFICIENT_DATA":
                continue
            try:
                hit = self._cache_lookup(ticker, key)
            except Exception as exc:  # noqa: BLE001 - cache lookup is best-effort only
                logger.warning(f"cache_lookup raised for {ticker}/{key}: {exc}")
                continue
            if hit is None:
                continue
            cached_value, original_retrieved_at = hit
            source = f"cached_real:sec_edgar:{original_retrieved_at}"
            updated[key] = FinancialSignal(
                value=cached_value,
                live=False,
                source=source,
                summary=(
                    f"Live SEC EDGAR fetch failed ({signal.summary}); reusing this platform's own last "
                    f"observed live value from {original_retrieved_at} — STALE, verify before citing."
                ),
                status="AVAILABLE",
                quality_state=classify_state(source).value,
            )
            logger.info(f"IngestionAgent: CACHED_REAL fallback used for {ticker}/{key} (original: {original_retrieved_at})")
        return updated

    def fetch_enriched_evidence(
        self, company: CompanyInput
    ) -> tuple[FinancialProfile | None, ChangeTilePairResult | None]:
        """Optional, additive fetch for the enterprise dashboard's real-data
        surfaces (Real Financial Intelligence, Satellite Change Detection
        View) — never called by the core scoring pipeline itself.

        This is a *separate* method rather than a change to ``run()``'s
        return contract specifically so ``AssessmentOrchestrator`` can call
        it defensively (``getattr(..., None)`` + ``callable()``) without
        requiring every existing ``IngestionAgent`` test fake (which only
        implements ``run()``) to be updated. Returns ``(None, None)`` in
        offline demo mode, since no live ``FinancialProfile`` or
        ``ChangeTilePairResult`` exists to report — callers must treat both
        as optional rather than assuming either succeeds.
        """
        if not self._settings.orbitaliq_live_data_mode:
            return None, None

        financial_profile: FinancialProfile | None = None
        try:
            financial_profile = self._sec.fetch_financial_profile(company.ticker)
        except Exception as exc:  # noqa: BLE001 - this view is optional; never fail the assessment over it
            logger.warning(f"fetch_enriched_evidence: financial profile unavailable for {company.ticker}: {exc}")

        # Reuses the exact tile pair already fetched by run() for this
        # instance/assessment (see the comment on self._last_change_pair) —
        # avoids a second, redundant round of NASA GIBS requests.
        return financial_profile, self._last_change_pair
