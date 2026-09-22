"""Centralized application configuration.

All runtime configuration is sourced from environment variables (with a
``.env`` file as a local-dev convenience) so the same container image can
be promoted across dev / staging / production without code changes —
a standard enterprise twelve-factor pattern.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "OrbitalIQ Enterprise — Competitive Expansion Intelligence Platform"
    orbitaliq_env: Literal["development", "staging", "production", "test"] = Field(
        default="development", alias="ORBITALIQ_ENV"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- NVIDIA NIM (LLM inference microservice) ---
    nvidia_api_key: str = Field(default="nvapi-REPLACE_ME", alias="NVIDIA_API_KEY")
    nvidia_nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_NIM_BASE_URL"
    )
    nvidia_nim_chat_model: str = Field(
        default="meta/llama-3.1-70b-instruct", alias="NVIDIA_NIM_CHAT_MODEL"
    )
    nvidia_nim_embed_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5", alias="NVIDIA_NIM_EMBED_MODEL"
    )
    nvidia_nim_timeout_seconds: float = Field(default=30.0, alias="NVIDIA_NIM_TIMEOUT_SECONDS")

    # When true (default for local dev / CI), the agent pipeline never makes
    # a network call to NVIDIA NIM and instead uses deterministic,
    # unit-testable mock generations. Set to false and provide a real
    # NVIDIA_API_KEY to run against the live NIM endpoint.
    orbitaliq_offline_mode: bool = Field(default=True, alias="ORBITALIQ_OFFLINE_MODE")

    # --- Vision model ---
    # "auto" selects CUDA when a GPU is present (e.g. an NVIDIA A10/T4/L4 in
    # production) and transparently falls back to CPU otherwise (e.g. this
    # sandbox, a laptop, or a CI runner).
    orbitaliq_vision_device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto", alias="ORBITALIQ_VISION_DEVICE"
    )
    vision_model_checkpoint: str | None = Field(default=None, alias="VISION_MODEL_CHECKPOINT")

    # --- Live external data sources ---
    # When true (the honest, "real data" default), the ingestion agent
    # calls genuine, entirely free, keyless public APIs — SEC EDGAR
    # (real XBRL-tagged financial filings) and NASA GIBS (real MODIS/VIIRS
    # satellite imagery) — for every signal. Any source that's unreachable
    # or lacks coverage for a given ticker/location falls back
    # automatically and independently (see data/sec_edgar_client.py and
    # data/imagery_provider.py); every assessment records exactly which
    # signals were live vs. fallback in its `data_sources` field, so
    # nothing is silently faked. Set to false to force the fully offline,
    # deterministic synthetic-data demo mode (used by the test suite/CI so
    # it never depends on network availability). Neither source has any
    # paid tier or usage cost — this pipeline's data layer is free to run
    # indefinitely.
    orbitaliq_live_data_mode: bool = Field(default=True, alias="ORBITALIQ_LIVE_DATA_MODE")
    live_data_timeout_seconds: float = Field(default=6.0, alias="LIVE_DATA_TIMEOUT_SECONDS")

    # --- Strict real-data (enterprise) mode ---
    # When true, the ingestion/scoring/report layers NEVER substitute a
    # synthetic value to "complete" a metric. Any signal that cannot be
    # retrieved from a genuine live source is surfaced explicitly as
    # NOT_AVAILABLE / "Insufficient data" rather than filled in — this is
    # the mode the enterprise dashboard runs in by default. It requires
    # ORBITALIQ_LIVE_DATA_MODE=true (strict mode with no live attempt at
    # all would mean every metric is trivially "not available", which is
    # rarely what's wanted) but is otherwise independent: turning it off
    # restores the original demo-friendly behavior (used by the existing
    # test suite / CI), where an unreachable source degrades to a labeled
    # synthetic fallback instead of an explicit gap.
    orbitaliq_strict_real_data_mode: bool = Field(default=False, alias="ORBITALIQ_STRICT_REAL_DATA_MODE")

    # SEC EDGAR requires no API key — only a descriptive User-Agent header
    # per its fair-access policy: https://www.sec.gov/os/webmaster-faq#developers
    # Set this to "Your Name your.email@example.com" before heavy use.
    sec_edgar_user_agent: str = Field(
        default="OrbitalIQ-Portfolio-Project research@example.com", alias="SEC_EDGAR_USER_AGENT"
    )
    sec_edgar_base_url: str = Field(default="https://www.sec.gov", alias="SEC_EDGAR_BASE_URL")
    sec_edgar_data_base_url: str = Field(default="https://data.sec.gov", alias="SEC_EDGAR_DATA_BASE_URL")

    # NASA GIBS (Global Imagery Browse Services) — free, keyless, real
    # MODIS/VIIRS satellite imagery: https://nasa-gibs.github.io/gibs-api-docs/
    nasa_gibs_base_url: str = Field(default="https://gibs.earthdata.nasa.gov/wmts", alias="NASA_GIBS_BASE_URL")

    # US Census Bureau Geocoder — free, keyless, US-government-authoritative
    # street-address -> lat/lon geocoding: https://geocoding.geo.census.gov/geocoder/
    # Used only to turn a company's real SEC-EDGAR-registered business
    # address (data/sec_edgar_client.py::SecEdgarClient.fetch_business_address)
    # into verified coordinates for the "resolve by company name" workflow
    # (data/company_resolver.py) -- never used to guess or interpolate a
    # coordinate. US addresses only; a non-US or unmatched address correctly
    # yields INSUFFICIENT_DATA rather than a guess.
    census_geocoder_base_url: str = Field(
        default="https://geocoding.geo.census.gov/geocoder", alias="CENSUS_GEOCODER_BASE_URL"
    )

    # --- Storage ---
    database_url: str = Field(default="sqlite:///./orbitaliq.db", alias="DATABASE_URL")

    # --- Watchlist escalation webhook ---
    # No default: unset means the WatchlistAgent's dispatch step is a
    # logged stub (see agents/watchlist_agent.py), which is disclosed to
    # every caller via WatchlistDecision.webhook_status /
    # AssessmentResponse.watchlist_webhook_status rather than silently
    # pretending an escalation was sent. Set this to a real Slack/Teams
    # incoming-webhook URL to make escalations operational with no other
    # code change.
    watchlist_webhook_url: str | None = Field(default=None, alias="WATCHLIST_WEBHOOK_URL")


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
