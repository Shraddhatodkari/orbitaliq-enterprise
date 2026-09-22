#!/usr/bin/env python3
"""End-to-end command-line demo of the OrbitalIQ pipeline.

Runs the full ingestion -> vision -> intelligence-scoring -> report ->
watchlist agent pipeline against a handful of real, publicly-traded
companies and their well-known facilities, with no server or database
required, and prints a formatted competitive-momentum briefing for each.

100% free to run: SEC EDGAR and NASA GIBS are both free, keyless public
APIs with no usage cost, and the vision CNN runs fine on CPU (no GPU
required — it's intentionally small, ~90K params).

Usage:
    python scripts/run_demo.py
    ORBITALIQ_OFFLINE_MODE=false NVIDIA_API_KEY=nvapi-... python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbitaliq.agents.orchestrator import AssessmentOrchestrator  # noqa: E402
from orbitaliq.agents.types import CompanyInput  # noqa: E402
from orbitaliq.config import get_settings  # noqa: E402
from orbitaliq.core.logging_config import configure_logging  # noqa: E402

# Real, publicly-traded companies and real facility coordinates — chosen
# because each has a well-documented, real-world expansion story that a
# recruiter or interviewer can independently sanity-check.
DEMO_COMPANIES = [
    CompanyInput(
        ticker="TSLA",
        company_name="Tesla, Inc.",
        facility_name="Gigafactory Nevada",
        latitude=39.5380,
        longitude=-119.4425,
        industry="automotive",
        market_cap_usd=800_000_000_000,
    ),
    CompanyInput(
        ticker="AMZN",
        company_name="Amazon.com, Inc.",
        facility_name="CVG1 Fulfillment/Air Hub, Hebron KY",
        latitude=39.0470,
        longitude=-84.6621,
        industry="e-commerce logistics",
        market_cap_usd=1_900_000_000_000,
    ),
    CompanyInput(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        facility_name="Santa Clara Headquarters / Voyager & Endeavor",
        latitude=37.3722,
        longitude=-121.9664,
        industry="semiconductors",
        market_cap_usd=3_000_000_000_000,
    ),
    CompanyInput(
        ticker="TSM",
        company_name="Taiwan Semiconductor Manufacturing Company",
        facility_name="Fab 21, Phoenix, Arizona",
        latitude=33.6011,
        longitude=-112.1449,
        industry="semiconductor foundry",
        market_cap_usd=900_000_000_000,
    ),
]


def _print_divider() -> None:
    print("=" * 88)


def main() -> None:
    configure_logging()
    settings = get_settings()

    print()
    _print_divider()
    print(f"  {settings.app_name} — Competitive & Market Intelligence Demo")
    print(f"  LLM narrative:      {'OFFLINE (deterministic template)' if settings.orbitaliq_offline_mode else 'LIVE NVIDIA NIM'}")
    print(f"  Financial/satellite: {'LIVE (SEC EDGAR / NASA GIBS)' if settings.orbitaliq_live_data_mode else 'OFFLINE (synthetic demo generator)'}")
    print("  Cost: $0.00 — every data source used here is free and keyless.")
    _print_divider()

    with AssessmentOrchestrator() as orchestrator:
        results = []
        for company in DEMO_COMPANIES:
            result = orchestrator.run(company)
            results.append(result)

            print(f"\n{company.ticker}: {company.company_name}  —  {company.facility_name}")
            print(f"  Composite Momentum Score: {result.momentum_score:.1f}/100  [{result.momentum_level}]")
            print("  Signal breakdown:")
            for signal, score in sorted(result.signal_breakdown.items(), key=lambda kv: -kv[1]):
                bar = "#" * int(score / 5)
                print(f"    {signal:<24s} {score:5.1f}  {bar}")
            print(f"  Satellite change score: {result.vision_findings['overall_change_score']:.2f} "
                  f"(device={result.vision_findings['device_used']})")
            live_count = sum(1 for v in result.data_sources.values() if str(v).endswith("_live"))
            print(f"  Data provenance: {live_count}/{len(result.data_sources)} signals live")
            for signal, source in result.data_sources.items():
                tag = "LIVE " if source.endswith("_live") else "demo "
                print(f"    [{tag}] {signal:<26s} {source}")
            print(f"  Watchlist flagged: {result.watchlist_flagged} "
                  f"{'-> #' + result.watchlist_channel if result.watchlist_channel else ''}")
            print("  Executive briefing:")
            print(f"    {result.narrative_report}")
            print("  Recommended next steps:")
            for action in result.recommended_actions:
                print(f"    - {action}")
            print("  Agent trace:")
            for step in result.agent_trace:
                print(f"    [{step.status:>5s}] {step.agent:<26s} {step.duration_ms:6.2f} ms  {step.summary}")

    _print_divider()
    ranked = sorted(results, key=lambda r: -r.momentum_score)
    print("  Companies ranked by expansion momentum (highest first):")
    for r in ranked:
        print(f"    {r.momentum_score:5.1f}  {r.momentum_level:<22s} {r.company_input.ticker}  {r.company_input.facility_name}")
    _print_divider()
    print()


if __name__ == "__main__":
    main()
