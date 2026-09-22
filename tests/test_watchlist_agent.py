from __future__ import annotations

import httpx
import pytest
import respx

from orbitaliq.agents.watchlist_agent import (
    DISPATCH_FAILED_STATUS,
    DISPATCHED_STATUS,
    STUBBED_STATUS,
    WatchlistAgent,
)
from orbitaliq.config import Settings
from orbitaliq.core.intelligence_engine import MomentumLevel


@pytest.mark.parametrize(
    "level,expected_tier,expected_flagged",
    [
        (MomentumLevel.STABLE, "NONE", False),
        (MomentumLevel.EMERGING, "WATCH", True),
        (MomentumLevel.STRONG, "REVIEW", True),
        (MomentumLevel.AGGRESSIVE_EXPANSION, "HIGH_SIGNAL", True),
    ],
)
def test_tier_assignment_is_deterministic_from_momentum_level(level, expected_tier, expected_flagged):
    agent = WatchlistAgent(settings=Settings())
    decision = agent.run("Acme Corp", level, 50.0)
    assert decision.tier == expected_tier
    assert decision.flagged is expected_flagged


def test_stable_is_never_flagged_and_channel_is_none():
    agent = WatchlistAgent(settings=Settings())
    decision = agent.run("Acme Corp", MomentumLevel.STABLE, 10.0)
    assert decision.flagged is False
    assert decision.channel is None
    assert decision.webhook_status == "NOT_APPLICABLE"


def test_each_tier_has_a_distinct_documented_channel():
    agent = WatchlistAgent(settings=Settings())
    channels = {
        agent.run("A", MomentumLevel.EMERGING, 40.0).channel,
        agent.run("A", MomentumLevel.STRONG, 60.0).channel,
        agent.run("A", MomentumLevel.AGGRESSIVE_EXPANSION, 90.0).channel,
    }
    assert len(channels) == 3
    assert None not in channels


def test_webhook_dispatch_is_stubbed_when_no_url_configured():
    agent = WatchlistAgent(settings=Settings())  # WATCHLIST_WEBHOOK_URL unset by default
    decision = agent.run("Acme Corp", MomentumLevel.STRONG, 60.0)
    assert decision.webhook_status == STUBBED_STATUS


@respx.mock
def test_webhook_dispatch_succeeds_when_url_configured():
    route = respx.post("https://hooks.example.com/watchlist").mock(return_value=httpx.Response(200, json={"ok": True}))
    settings = Settings(WATCHLIST_WEBHOOK_URL="https://hooks.example.com/watchlist")
    agent = WatchlistAgent(settings=settings)
    decision = agent.run("Acme Corp", MomentumLevel.AGGRESSIVE_EXPANSION, 90.0)
    assert decision.webhook_status == DISPATCHED_STATUS
    assert route.called


@respx.mock
def test_webhook_dispatch_failure_is_disclosed_never_raises():
    respx.post("https://hooks.example.com/watchlist").mock(side_effect=httpx.ConnectError("boom"))
    settings = Settings(WATCHLIST_WEBHOOK_URL="https://hooks.example.com/watchlist")
    agent = WatchlistAgent(settings=settings)
    decision = agent.run("Acme Corp", MomentumLevel.AGGRESSIVE_EXPANSION, 90.0)  # must not raise
    assert decision.webhook_status == DISPATCH_FAILED_STATUS
    assert decision.flagged is True  # a failed escalation never blocks the assessment result


def test_as_dict_shape():
    agent = WatchlistAgent(settings=Settings())
    d = agent.run("Acme Corp", MomentumLevel.STRONG, 60.0).as_dict()
    assert set(d.keys()) == {"flagged", "channel", "reason", "tier", "webhook_status"}


def test_insufficient_data_is_never_flagged_and_reason_says_why():
    """A score that couldn't be legitimately calculated (see
    core/intelligence_engine.py's data-sufficiency gate) must never be
    escalated, and the reason must say plainly why -- not the generic
    "momentum level STABLE" wording used for a genuinely low real score.
    """
    agent = WatchlistAgent(settings=Settings())
    decision = agent.run("Acme Corp", MomentumLevel.INSUFFICIENT_DATA, 0.0)
    assert decision.flagged is False
    assert decision.tier == "NONE"
    assert decision.channel is None
    assert decision.webhook_status == "NOT_APPLICABLE"
    assert "insufficient" in decision.reason.lower()
    assert "STABLE" not in decision.reason
