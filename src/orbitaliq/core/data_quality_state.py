"""The canonical 5-state data-quality vocabulary for every live-path
signal this platform produces.

Every financial signal, financial metric, and imagery result the
*production* (``ORBITALIQ_LIVE_DATA_MODE=true``) pipeline emits resolves to
**exactly one** of these five states — never a fabricated value standing in
for one of them:

- ``LIVE`` — a directly-disclosed value from a genuine, just-fetched
  authoritative source (a real SEC EDGAR XBRL fact, a real NASA GIBS tile).
- ``CACHED_REAL`` — the live fetch failed *this time*, but a genuine,
  previously-fetched live value for the same (ticker, signal) exists in
  this platform's own assessment history and is reused, with its original
  retrieval timestamp disclosed and flagged as potentially stale. Still a
  real, previously-observed value — never invented.
- ``DERIVED`` — computed *from* one or more LIVE/CACHED_REAL inputs (a
  margin, a ratio, a growth rate) rather than itself a directly-disclosed
  fact. Still fully real and traceable, just one calculation step removed
  from the raw filing.
- ``INSUFFICIENT_DATA`` — a documented, expected reason why no value is
  available (ticker not found, filer doesn't report this concept, fewer
  than two comparable fiscal years, etc.) — a clean, understood gap.
- ``ERROR`` — an unexpected failure (network exception, malformed
  response, parse failure) while attempting the fetch, distinguished from
  ``INSUFFICIENT_DATA`` because it is *not* a documented data gap: the data
  may well exist, but this attempt to retrieve it broke unexpectedly.

This module is the single source of truth for that vocabulary and for
classifying the free-form provenance strings already used throughout the
codebase (``data_sources`` maps, ``FinancialSignal.source``,
``ImageryResult.source``, etc.) into it, so ``core/data_quality.py``'s
bucketing and ``core/intelligence_engine.py``'s trust gate can never
disagree about what a given label means.

The disclosed, non-production offline/demo path (``synthetic_fallback:*``,
``offline_demo_mode`` — see ``data/sec_edgar_client.py`` and
``agents/ingestion_agent.py``) is deliberately **not** one of these five
states: it is a separate, clearly-labeled non-production concept (local
dev/CI only, confined to ``ORBITALIQ_LIVE_DATA_MODE=false`` and test
fixtures — see ``docs/FINAL_VALIDATION_REPORT.md``), and collapsing it into
this vocabulary would blur exactly the line this contract exists to keep
sharp: "did we actually try a live source and get a real answer" vs.
"was this a disclosed, non-production placeholder."
"""
from __future__ import annotations

from enum import Enum


class DataQualityState(str, Enum):
    LIVE = "LIVE"
    CACHED_REAL = "CACHED_REAL"
    DERIVED = "DERIVED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ERROR = "ERROR"


# States that represent a real, usable numeric value -- eligible to be
# trusted/weighed by the scoring engine. INSUFFICIENT_DATA and ERROR never
# carry a real value and are never trusted.
TRUSTED_STATES = frozenset({DataQualityState.LIVE, DataQualityState.CACHED_REAL, DataQualityState.DERIVED})

# Non-production, fully-disclosed demo/test-fixture labels -- never
# produced by a live-mode failure path (see module docstring). Recognized
# here, not folded into DataQualityState, so callers can still tell "this
# ran in disclosed offline demo mode" apart from any of the five real
# production states.
#
# These two labels are deliberately NOT treated the same way by
# ``classify_state``/``is_trusted`` below, preserving a distinction that
# predates this module and is exercised by existing tests:
# ``offline_demo_mode`` (the whole-assessment offline demo mode switch --
# see ``agents/ingestion_agent.py::OFFLINE_SOURCE_LABELS``) is trusted for
# scoring, since it's a deliberate, fully-disclosed stand-in for the entire
# live pipeline, not a live-mode failure. ``synthetic_fallback:*`` (the
# per-signal deterministic generator in ``data/sec_edgar_client.py`` --
# reachable only via ``synthetic_financial_signals()``, i.e. also
# offline-demo-mode/test-fixture-only, never a live-mode failure path) is
# kept recognized as a demo/non-production label but is NOT trusted for
# scoring, for backward compatibility with older persisted assessments and
# direct test fixtures that use that label to exercise the exclusion path.
_DEMO_FALLBACK_PREFIXES = ("synthetic_fallback",)
_DEMO_FALLBACK_EXACT = frozenset({"offline_demo_mode"})
_DEMO_FALLBACK_TRUSTED_EXACT = frozenset({"offline_demo_mode"})


def is_demo_fallback(source: str | None) -> bool:
    if source is None:
        return False
    s = str(source)
    return s in _DEMO_FALLBACK_EXACT or s.startswith(_DEMO_FALLBACK_PREFIXES)


def classify_state(source: str | None) -> DataQualityState:
    """Classify a provenance string into one of the five canonical states.

    ``source is None`` classifies as ``LIVE`` (trusted) -- this preserves
    the pre-existing behavior of every caller that predates this module
    (a direct unit-test call with no ``data_sources`` map at all means
    "trust everything", exactly as before).

    Recognizes both the pre-existing labels already in use
    (``*_live``, ``insufficient_data*``, ``NOT_AVAILABLE``) and the new
    labels this contract introduces (``cached_real:*``, ``derived:*``,
    ``error:*``). An unrecognized label defaults to ``INSUFFICIENT_DATA``
    (the conservative choice -- never silently trusted).
    """
    if source is None:
        return DataQualityState.LIVE

    s = str(source)

    if s in _DEMO_FALLBACK_TRUSTED_EXACT:
        # offline_demo_mode: deliberate, fully disclosed stand-in for the
        # entire live pipeline -- trusted for scoring (see module comment
        # above _DEMO_FALLBACK_PREFIXES). Classifies as LIVE for the
        # purposes of the 5-state contract since it was never claimed to be
        # a live source but is not a live-mode failure either; callers that
        # need to tell it apart from a genuine live fetch should check
        # `is_demo_fallback()` explicitly.
        return DataQualityState.LIVE
    if is_demo_fallback(s):
        # synthetic_fallback:* -- a disclosed demo/test-fixture label, but
        # NOT trusted for scoring (see module comment above). Classifies as
        # INSUFFICIENT_DATA: a real value was not legitimately produced.
        return DataQualityState.INSUFFICIENT_DATA
    if s.startswith("cached_real"):
        return DataQualityState.CACHED_REAL
    if s.startswith("derived"):
        return DataQualityState.DERIVED
    if s.startswith("error"):
        return DataQualityState.ERROR
    if s == "NOT_AVAILABLE" or s.startswith("insufficient_data"):
        return DataQualityState.INSUFFICIENT_DATA
    if s.endswith("_live"):
        return DataQualityState.LIVE
    # Unrecognized label -- conservative default, never silently trusted.
    return DataQualityState.INSUFFICIENT_DATA


def is_trusted(source: str | None) -> bool:
    """Whether a signal with this provenance label should be weighed by
    the deterministic scoring engine. Delegates entirely to
    ``classify_state`` so this can never drift from the 5-state contract.
    """
    return classify_state(source) in TRUSTED_STATES
