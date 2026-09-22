"""Watchlist Agent — classifies a completed assessment into a documented,
deterministic escalation tier and (optionally) dispatches to a
notification channel.

**Tier rules are deterministic and derive from nothing but the already-
computed, already-tested ``MomentumLevel`` bucket** (see
``core/intelligence_engine.py::MOMENTUM_LEVEL_THRESHOLDS``) — no new
scoring logic, weighting, or LLM judgment is introduced here:

    STABLE (0-30)              -> NONE        (not on the watchlist)
    EMERGING (30-55)           -> WATCH        (early signal, monitor)
    STRONG (55-75)              -> REVIEW       (warrants analyst review)
    AGGRESSIVE_EXPANSION (75+) -> HIGH_SIGNAL  (escalate immediately)

**Webhook dispatch is honestly disclosed as stubbed unless configured.**
``WATCHLIST_WEBHOOK_URL`` is unset by default, so every escalation's
``webhook_status`` reads ``STUBBED_NO_WEBHOOK_CONFIGURED`` — the escalation
is logged, never silently dropped, but no external system is actually
notified. Setting ``WATCHLIST_WEBHOOK_URL`` to a real Slack/Teams incoming
webhook makes dispatch operational with no other code change; the status
then reads ``DISPATCHED`` or ``DISPATCH_FAILED`` (network error) so a
caller never has to guess whether an escalation actually went out.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from orbitaliq.config import Settings, get_settings
from orbitaliq.core.intelligence_engine import MomentumLevel
from orbitaliq.core.logging_config import logger

TIER_BY_MOMENTUM_LEVEL: dict[MomentumLevel, str] = {
    MomentumLevel.STABLE: "NONE",
    MomentumLevel.EMERGING: "WATCH",
    MomentumLevel.STRONG: "REVIEW",
    MomentumLevel.AGGRESSIVE_EXPANSION: "HIGH_SIGNAL",
    # A score that couldn't be legitimately calculated (see
    # core/intelligence_engine.py) is never escalated -- there is nothing
    # real to escalate on.
    MomentumLevel.INSUFFICIENT_DATA: "NONE",
}

CHANNEL_BY_TIER: dict[str, str] = {
    "WATCH": "competitive-intel-watch",
    "REVIEW": "competitive-intel-review",
    "HIGH_SIGNAL": "competitive-intel-priority",
}

STUBBED_STATUS = "STUBBED_NO_WEBHOOK_CONFIGURED"
DISPATCHED_STATUS = "DISPATCHED"
DISPATCH_FAILED_STATUS = "DISPATCH_FAILED"


@dataclass
class WatchlistDecision:
    flagged: bool
    channel: str | None
    reason: str
    # "NONE" | "WATCH" | "REVIEW" | "HIGH_SIGNAL"
    tier: str
    # "STUBBED_NO_WEBHOOK_CONFIGURED" | "DISPATCHED" | "DISPATCH_FAILED" | "NOT_APPLICABLE"
    webhook_status: str

    def as_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "channel": self.channel,
            "reason": self.reason,
            "tier": self.tier,
            "webhook_status": self.webhook_status,
        }


class WatchlistAgent:
    def __init__(self, settings: Settings | None = None, http_client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        self._owns_client = http_client is None
        self._client = http_client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    def run(self, company_name: str, momentum_level: MomentumLevel, momentum_score: float) -> WatchlistDecision:
        tier = TIER_BY_MOMENTUM_LEVEL[momentum_level]

        if tier == "NONE":
            reason = (
                "Competitive Expansion Signal could not be legitimately calculated (insufficient live data) "
                "— not eligible for the watchlist"
                if momentum_level is MomentumLevel.INSUFFICIENT_DATA
                else "Competitive Expansion Signal STABLE — below the WATCH tier's EMERGING threshold"
            )
            return WatchlistDecision(
                flagged=False,
                channel=None,
                reason=reason,
                tier=tier,
                webhook_status="NOT_APPLICABLE",
            )

        channel = CHANNEL_BY_TIER[tier]
        webhook_status = self._dispatch(company_name, momentum_level, momentum_score, tier, channel)
        return WatchlistDecision(
            flagged=True,
            channel=channel,
            reason=f"momentum level {momentum_level.value} -> {tier} tier",
            tier=tier,
            webhook_status=webhook_status,
        )

    def _dispatch(
        self, company_name: str, momentum_level: MomentumLevel, momentum_score: float, tier: str, channel: str
    ) -> str:
        message = (
            f"WATCHLIST [{tier}] flagged to #{channel}: {company_name} "
            f"momentum_level={momentum_level.value} score={momentum_score:.1f}"
        )
        webhook_url = self._settings.watchlist_webhook_url
        if not webhook_url:
            logger.warning(f"{message} (webhook stub — set WATCHLIST_WEBHOOK_URL to make this operational)")
            return STUBBED_STATUS

        try:
            client = self._client or httpx.Client(timeout=5.0)
            try:
                response = client.post(webhook_url, json={"text": message, "tier": tier, "channel": channel})
                response.raise_for_status()
            finally:
                if self._client is None:
                    client.close()
            logger.info(f"{message} (dispatched to configured webhook)")
            return DISPATCHED_STATUS
        except Exception as exc:  # noqa: BLE001 - a failed escalation must never fail the assessment
            logger.warning(f"{message} (webhook dispatch FAILED: {exc})")
            return DISPATCH_FAILED_STATUS
