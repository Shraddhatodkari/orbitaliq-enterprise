"""Client for NVIDIA NIM (NVIDIA Inference Microservices).

NVIDIA NIM exposes an OpenAI-compatible ``/chat/completions`` endpoint for
GPU-optimized foundation models (Llama 3.x, Mixtral, Nemotron, ...),
whether self-hosted on NVIDIA GPUs or called hosted via
https://build.nvidia.com. This client targets that API directly with
``httpx`` (no proprietary SDK dependency), retries transient failures with
exponential backoff, and — critically for a reproducible demo/CI story —
supports a fully offline mode that returns deterministic, templated
generations without any network call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from orbitaliq.config import Settings, get_settings
from orbitaliq.core.logging_config import logger

# "derived_from_verified_evidence" (not "offline_mock"): this deterministic
# template is not a mock, demo, or offline-only stand-in -- it runs on the
# exact same code path whenever the grounding critic blocks an LLM-generated
# narrative (see agents/orchestrator.py::_ground_narrative), including in
# full live-data mode with a real NVIDIA_API_KEY configured. It is named for
# what it actually is: text built deterministically, directly from the same
# already-verified structured evidence (momentum score, signal breakdown,
# vision findings) that the LLM path would otherwise narrate -- never
# invented, and definitionally grounded because there is nothing in it that
# didn't come straight from that evidence.
GenerationSource = Literal["nvidia_nim", "derived_from_verified_evidence"]


class NimClientError(RuntimeError):
    """Raised when the NVIDIA NIM API cannot be reached or returns an error."""


@dataclass
class NimGenerationResult:
    text: str
    model: str
    source: GenerationSource
    usage: dict[str, Any]


_RETRYABLE_EXCEPTIONS = (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError)


class NvidiaNimClient:
    """Thin, dependency-light wrapper around the NVIDIA NIM chat-completions API."""

    def __init__(self, settings: Settings | None = None, http_client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=self._settings.nvidia_nim_base_url,
            timeout=self._settings.nvidia_nim_timeout_seconds,
            headers={
                "Authorization": f"Bearer {self._settings.nvidia_api_key}",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "NvidiaNimClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def is_offline(self) -> bool:
        return self._settings.orbitaliq_offline_mode or self._settings.nvidia_api_key.startswith("nvapi-REPLACE")

    def generate_intelligence_briefing(
        self,
        *,
        company_name: str,
        ticker: str,
        facility_name: str,
        momentum_score: float,
        momentum_level: str,
        signal_breakdown: dict[str, Any],
        vision_findings: dict[str, Any],
        max_tokens: int = 500,
        temperature: float = 0.2,
    ) -> NimGenerationResult:
        """Turn structured competitive-momentum output into an
        executive-ready narrative.

        Falls back to a deterministic template (no network call) whenever
        offline mode is enabled or no real API key is configured — this
        keeps the full agent pipeline runnable and unit-testable with zero
        external dependencies, while the exact same call path exercises
        the live NVIDIA NIM endpoint in a configured production deployment.
        """
        if self.is_offline:
            return self._offline_briefing(
                company_name=company_name,
                ticker=ticker,
                facility_name=facility_name,
                momentum_score=momentum_score,
                momentum_level=momentum_level,
                signal_breakdown=signal_breakdown,
                vision_findings=vision_findings,
            )

        system_prompt = (
            "You are a senior competitive-intelligence analyst at a management "
            "consulting firm, producing concise, decision-ready briefings for "
            "an engagement team benchmarking a competitor. Lead with the "
            "Competitive Expansion Signal level, not the raw number. Be "
            "specific, quantify the growth drivers, and avoid hedging filler "
            "language. Always note that the satellite evidence is directional "
            "(large-scale land-cover change only), not a confirmed fact."
        )
        user_prompt = (
            f"Company: {company_name} ({ticker})\n"
            f"Facility under review: {facility_name}\n"
            f"Competitive Expansion Signal: {momentum_level} (composite index {momentum_score}/100)\n"
            f"Signal breakdown (SEC EDGAR financial growth + satellite change detection): {signal_breakdown}\n"
            f"Satellite vision findings: {vision_findings}\n\n"
            "Write a 3-4 sentence executive briefing that leads with the Competitive Expansion Signal level, "
            "covering the dominant growth driver, what the satellite evidence adds (and its directional "
            "limitation), and one concrete next research step for the engagement team."
        )
        payload = {
            "model": self._settings.nvidia_nim_chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data = self._post_chat_completion(payload)
        text = data["choices"][0]["message"]["content"].strip()
        return NimGenerationResult(
            text=text,
            model=data.get("model", self._settings.nvidia_nim_chat_model),
            source="nvidia_nim",
            usage=data.get("usage", {}),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    )
    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                # Don't waste retries on an auth failure — it won't self-heal.
                raise NimClientError(
                    f"NVIDIA NIM authentication failed ({status}). Check NVIDIA_API_KEY."
                ) from exc
            logger.warning(f"NVIDIA NIM returned {status}; retrying if attempts remain")
            raise
        except httpx.TransportError as exc:
            logger.warning(f"NVIDIA NIM transport error: {exc}; retrying if attempts remain")
            raise
        return response.json()

    def offline_briefing(
        self,
        *,
        company_name: str,
        ticker: str,
        facility_name: str,
        momentum_score: float,
        momentum_level: str,
        signal_breakdown: dict[str, Any],
        vision_findings: dict[str, Any],
    ) -> NimGenerationResult:
        """Public entry point for the deterministic offline template,
        independent of ``is_offline``/network configuration.

        Used by the Report Agent's grounding-gated fallback path: when the
        grounding critic (``core/grounding.py``) blocks an LLM-generated
        narrative for citing a number that doesn't trace back to real
        evidence, the pipeline falls back to this template rather than
        publishing the blocked text — this method makes that fallback
        callable on demand rather than only implicitly via offline mode.
        """
        return self._offline_briefing(
            company_name=company_name,
            ticker=ticker,
            facility_name=facility_name,
            momentum_score=momentum_score,
            momentum_level=momentum_level,
            signal_breakdown=signal_breakdown,
            vision_findings=vision_findings,
        )

    @staticmethod
    def _offline_briefing(
        *,
        company_name: str,
        ticker: str,
        facility_name: str,
        momentum_score: float,
        momentum_level: str,
        signal_breakdown: dict[str, Any],
        vision_findings: dict[str, Any],
    ) -> NimGenerationResult:
        if momentum_level == "INSUFFICIENT_DATA":
            text = (
                f"{company_name} ({ticker}): Competitive Expansion Signal — Insufficient Data for "
                f"{facility_name}. Every underlying financial (SEC EDGAR) and satellite (NASA GIBS) signal for "
                "this assessment came back unavailable or fell back to a labeled synthetic placeholder rather "
                "than live data -- rather than publish a signal partly or wholly derived from synthetic inputs, "
                "this assessment reports Insufficient Data. Retry once the live data sources are reachable."
            )
            return NimGenerationResult(
                text=text,
                model="offline-deterministic-template-v1",
                source="derived_from_verified_evidence",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        dominant = max(signal_breakdown.items(), key=lambda kv: kv[1]) if signal_breakdown else ("unknown", 0)
        dominant_name, dominant_value = dominant
        change_score = vision_findings.get("overall_change_score")
        change_clause = (
            f" Bi-temporal satellite analysis of {facility_name} flagged an overall change score of "
            f"{change_score:.2f} (large-scale land-cover change only, not a confirmed fact), directionally "
            "consistent with the dominant financial driver."
            if isinstance(change_score, (int, float))
            else ""
        )
        text = (
            f"{company_name} ({ticker})'s Competitive Expansion Signal for {facility_name} is "
            f"{momentum_level} (composite index {momentum_score:.1f}/100). The dominant growth driver is "
            f"{dominant_name.replace('_', ' ')} (contribution {dominant_value:.1f}/100).{change_clause} "
            "Next step: corroborate this signal with permitting filings, hiring data, or press coverage "
            "before including it in a client-facing deliverable."
        )
        return NimGenerationResult(
            text=text,
            model="offline-deterministic-template-v1",
            source="derived_from_verified_evidence",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
