import httpx
import pytest
import respx

from orbitaliq.config import Settings
from orbitaliq.nvidia.nim_client import NimClientError, NvidiaNimClient


def _settings(**overrides) -> Settings:
    defaults = dict(
        NVIDIA_API_KEY="nvapi-REAL-TEST-KEY",
        NVIDIA_NIM_BASE_URL="https://fake-nim.test/v1",
        ORBITALIQ_OFFLINE_MODE=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _briefing_kwargs(**overrides):
    defaults = dict(
        company_name="Tesla, Inc.",
        ticker="TSLA",
        facility_name="Gigafactory Nevada",
        momentum_score=72.5,
        momentum_level="STRONG",
        signal_breakdown={"revenue_growth": 60.0, "construction_expansion": 55.0},
        vision_findings={"overall_change_score": 0.55},
    )
    defaults.update(overrides)
    return defaults


def test_offline_mode_never_makes_network_call():
    settings = _settings(ORBITALIQ_OFFLINE_MODE=True)
    client = NvidiaNimClient(settings=settings)
    assert client.is_offline is True
    result = client.generate_intelligence_briefing(**_briefing_kwargs())
    assert result.source == "derived_from_verified_evidence"
    assert "Tesla, Inc." in result.text
    assert "72.5" in result.text
    client.close()


def test_placeholder_api_key_forces_offline_even_if_flag_false():
    settings = _settings(ORBITALIQ_OFFLINE_MODE=False, NVIDIA_API_KEY="nvapi-REPLACE_ME")
    client = NvidiaNimClient(settings=settings)
    assert client.is_offline is True
    client.close()


@respx.mock
def test_live_mode_calls_chat_completions_and_parses_response():
    settings = _settings()
    route = respx.post("https://fake-nim.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "meta/llama-3.1-70b-instruct",
                "choices": [{"message": {"content": "  Executive summary text.  "}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
            },
        )
    )
    client = NvidiaNimClient(settings=settings)
    result = client.generate_intelligence_briefing(**_briefing_kwargs())

    assert route.called
    assert result.source == "nvidia_nim"
    assert result.text == "Executive summary text."
    assert result.usage["total_tokens"] == 160
    client.close()


@respx.mock
def test_auth_failure_raises_nim_client_error_without_retry_storm():
    settings = _settings()
    route = respx.post("https://fake-nim.test/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    client = NvidiaNimClient(settings=settings)
    with pytest.raises(NimClientError):
        client.generate_intelligence_briefing(**_briefing_kwargs())
    assert route.call_count == 1  # auth errors should not be retried
    client.close()


@respx.mock
def test_transient_5xx_is_retried_then_succeeds():
    settings = _settings()
    route = respx.post("https://fake-nim.test/v1/chat/completions")
    route.side_effect = [
        httpx.Response(503, json={"error": "unavailable"}),
        httpx.Response(
            200,
            json={
                "model": "meta/llama-3.1-70b-instruct",
                "choices": [{"message": {"content": "Recovered after retry."}}],
                "usage": {},
            },
        ),
    ]
    client = NvidiaNimClient(settings=settings)
    result = client.generate_intelligence_briefing(**_briefing_kwargs())
    assert result.text == "Recovered after retry."
    assert route.call_count == 2
    client.close()


@respx.mock
def test_persistent_5xx_exhausts_retries_and_raises():
    settings = _settings()
    respx.post("https://fake-nim.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "down"})
    )
    client = NvidiaNimClient(settings=settings)
    with pytest.raises(httpx.HTTPStatusError):
        client.generate_intelligence_briefing(**_briefing_kwargs())
    client.close()


def test_context_manager_closes_owned_client():
    settings = _settings(ORBITALIQ_OFFLINE_MODE=True)
    with NvidiaNimClient(settings=settings) as client:
        assert client.is_offline is True
    # closing twice should not raise
    client.close()
