#!/usr/bin/env python3
"""Seed the database with a set of demo assessments via the running API.

Usage (with the API already running, e.g. `docker compose up` or
`uvicorn orbitaliq.main:app`):

    python scripts/seed_data.py --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import sys

import httpx

COMPANIES = [
    {"ticker": "TSLA", "company_name": "Tesla, Inc.", "facility_name": "Gigafactory Nevada",
     "latitude": 39.5380, "longitude": -119.4425, "industry": "automotive", "market_cap_usd": 800_000_000_000},
    {"ticker": "AMZN", "company_name": "Amazon.com, Inc.", "facility_name": "CVG1 Fulfillment/Air Hub, Hebron KY",
     "latitude": 39.0470, "longitude": -84.6621, "industry": "e-commerce logistics", "market_cap_usd": 1_900_000_000_000},
    {"ticker": "NVDA", "company_name": "NVIDIA Corporation", "facility_name": "Santa Clara Headquarters",
     "latitude": 37.3722, "longitude": -121.9664, "industry": "semiconductors", "market_cap_usd": 3_000_000_000_000},
    {"ticker": "TSM", "company_name": "Taiwan Semiconductor Manufacturing Company", "facility_name": "Fab 21, Phoenix, Arizona",
     "latitude": 33.6011, "longitude": -112.1449, "industry": "semiconductor foundry", "market_cap_usd": 900_000_000_000},
    {"ticker": "MSFT", "company_name": "Microsoft Corporation", "facility_name": "Mount Pleasant, Wisconsin Data Center",
     "latitude": 42.7261, "longitude": -87.8998, "industry": "cloud infrastructure", "market_cap_usd": 3_100_000_000_000},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print(f"Connected: {health.json()}")

        for company in COMPANIES:
            resp = client.post("/api/v1/assessments", json=company)
            if resp.status_code != 201:
                print(f"FAILED {company['ticker']}: {resp.status_code} {resp.text}", file=sys.stderr)
                continue
            body = resp.json()
            print(f"Created {body['id']}: {body['ticker']} / {body['facility_name']} "
                  f"-> {body['momentum_score']:.1f} ({body['momentum_level']})")


if __name__ == "__main__":
    main()
