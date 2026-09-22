"""Streamlit app configuration.

The only thing this module configures is *where the FastAPI backend is* —
no business rules, thresholds, or scoring logic live here or anywhere else
in ``streamlit_app``.
"""
from __future__ import annotations

import os

# Overridable via an environment variable (e.g. if the backend runs on a
# different host/port) or Streamlit secrets (``.streamlit/secrets.toml``
# -> ``ORBITALIQ_API_BASE_URL = "http://localhost:8000"``). Defaults to
# the same localhost:8000 the existing dashboard/README already document.
DEFAULT_API_BASE_URL = "http://localhost:8000"


def get_api_base_url() -> str:
    env_value = os.environ.get("ORBITALIQ_API_BASE_URL")
    if env_value:
        return env_value.rstrip("/")
    try:
        import streamlit as st

        secret_value = st.secrets.get("ORBITALIQ_API_BASE_URL")  # type: ignore[union-attr]
        if secret_value:
            return str(secret_value).rstrip("/")
    except Exception:
        # No secrets.toml configured, or Streamlit not fully initialized
        # (e.g. this module imported outside a running Streamlit app) —
        # fall through to the default. Never a fatal error: the app must
        # still start and simply show connection errors if the backend
        # really is unreachable at the default URL.
        pass
    return DEFAULT_API_BASE_URL


# How long the Streamlit client waits for a backend response before
# surfacing a clear timeout error. Generous on purpose: a live assessment
# makes several real SEC EDGAR / NASA GIBS network calls plus a CNN
# forward pass, which can legitimately take 10-30s on a slower connection
# or an older CPU — this must not be confused with a crash.
REQUEST_TIMEOUT_SECONDS = 90.0

APP_TITLE = "OrbitalIQ Enterprise — Competitive Expansion Intelligence Platform"
PAGE_ICON = "\U0001F6F0"  # satellite emoji, matches the existing dashboard's theme

MOMENTUM_LEVELS = ("STABLE", "EMERGING", "STRONG", "AGGRESSIVE_EXPANSION")
WATCHLIST_TIERS = ("NONE", "WATCH", "REVIEW", "HIGH_SIGNAL")
