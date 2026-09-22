#!/usr/bin/env python3
"""Standalone proof that OrbitalIQ's data clients hit REAL public APIs --
driven end-to-end by a single company-name query, exactly the way the
normal product workflow is (company name in, everything else resolved).

THE FLOW IS STRICT AND ONE-DIRECTIONAL:

    company name
        |
        v
    company resolver (SEC EDGAR company directory)
        |
        v
    verified identity (legal name / ticker / CIK) -- or AMBIGUOUS / COMPANY_NOT_FOUND
        |
        v
    verified location, IF an authoritative source confirms one (else none)
        |
        v
    only THEN: SEC EDGAR company-facts check for the RESOLVED ticker,
               NASA GIBS check for the RESOLVED coordinate (skipped,
               never guessed, if no coordinate was verified)

There is no other path into the SEC or GIBS checks. In particular, neither
check has a fallback ticker or fallback coordinate baked in: passing
``--company Microsoft`` can never surface Tesla's ticker or Tesla's
coordinates, because nothing in this script has a Tesla-specific default
any more -- the only per-run identity/location data comes from resolving
the ``--company`` argument itself, or from an explicit ``--ticker``/
``--lat``/``--lon`` override (see below), never from a silent default.

Also printed, clearly labeled and NOT conflated: a generic, company-
independent "is the Census geocoder endpoint reachable at all" connectivity
check (``CENSUS ENDPOINT``) against a fixed, well-known address, versus
whether THIS run's company actually got a verified coordinate
(``COMPANY LOCATION``). The first says the government endpoint is up. The
second says this specific company's address was geocoded. They are
different facts and this script never lets one stand in for the other.

"LIVE" is printed ONLY when a signal actually came back with ``live=True``
/ a source ending in ``_live`` -- a genuine, successfully-parsed response
from the real public endpoint. A signal that fell back to a typed
``INSUFFICIENT_DATA`` result (network failure, unknown ticker, no usable
data for that filer/date, unverified address, etc.) is NEVER reported as
LIVE, and this script never substitutes or fabricates a value -- including
never substituting one company's ticker or coordinates for another's --
to make a source look live or a company look resolved. See
``data/sec_edgar_client.py``'s ``_insufficient_financial_signal`` and
``data/imagery_provider.py``'s equivalent for NASA GIBS, which is exactly
what a failed fetch returns, and ``data/company_resolver.py`` for the
RESOLVED / RESOLVED_NO_FACILITY / AMBIGUOUS / COMPANY_NOT_FOUND contract.

This sandbox this project was originally built in has restricted outbound
network access (package registries only, confirmed via direct `curl`
tests against data.sec.gov and gibs.earthdata.nasa.gov -- both return a
proxy-level connection rejection), so the live code paths could be
written and unit-tested against realistic mocked responses (see
tests/test_sec_edgar_client.py and tests/test_imagery_provider.py) but
never exercised against the real internet FROM THAT SANDBOX. Run this
script on a machine with normal internet access -- your laptop, a cloud
VM, a GitHub Actions runner -- to see the genuine live calls succeed (or
honestly fail) and print exactly what came back. This script's own
printed output, run on such a machine, is the actual live-verification
evidence -- not a claim made without running it.

Usage:
    python scripts/verify_live_apis.py --company Tesla
    python scripts/verify_live_apis.py --company Microsoft
    python scripts/verify_live_apis.py --company "Tata Motors"
    python scripts/verify_live_apis.py --company Corp            # AMBIGUOUS
    python scripts/verify_live_apis.py --company "not-a-real-co" # COMPANY_NOT_FOUND

    # Advanced / analyst override -- bypasses resolution for the SEC/GIBS
    # checks only (never for the resolver step itself, which always runs
    # against the literal --company text so its own output stays honest):
    python scripts/verify_live_apis.py --company Tesla --ticker TSLA --lat 39.538 --lon -119.4425

No API key is required for any source. All are entirely free with no
usage cost: SEC EDGAR only requires a descriptive User-Agent string
(SEC_EDGAR_USER_AGENT), and NASA GIBS / the Census geocoder are fully
keyless.

Exit codes:
    0  full verified chain: company RESOLVED with a verified location, and
       SEC EDGAR + NASA GIBS + the Census connectivity check were all LIVE.
    1  every network check failed (SEC EDGAR, NASA GIBS, and the Census
       connectivity check were all UNAVAILABLE).
    2  mixed / partial -- includes the common, entirely honest case of a
       real company that RESOLVED but has no verified facility (financial
       checks LIVE, satellite correctly skipped), AMBIGUOUS, or
       COMPANY_NOT_FOUND.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbitaliq.config import Settings  # noqa: E402
from orbitaliq.data.company_resolver import CompanyResolution, resolve_company  # noqa: E402
from orbitaliq.data.geocoding_client import CensusGeocodingClient  # noqa: E402
from orbitaliq.data.imagery_provider import NasaGibsImageryProvider  # noqa: E402
from orbitaliq.data.sec_edgar_client import SecEdgarClient  # noqa: E402


def _resolve(sec: SecEdgarClient, geocoder: CensusGeocodingClient, settings: Settings, query: str) -> CompanyResolution:
    """Step 1 of the flow, always run first, always driven only by the
    literal ``--company`` text -- this is the one and only place a
    ticker/CIK/coordinate can enter the rest of this script."""
    print("=" * 78)
    print("  STEP 1/3 -- Company resolution (SEC EDGAR company directory + Census geocoding)")
    print("=" * 78)
    print(f"  Company query: {query!r}")
    start = time.perf_counter()
    resolution = resolve_company(query, sec_client=sec, geocoder=geocoder, settings=settings)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"  status: {resolution.status}  ({elapsed_ms:.0f}ms)")
    if resolution.status == "AMBIGUOUS":
        print(f"  reason: {resolution.reason}")
        print(f"  candidates ({len(resolution.candidates)}):")
        for c in resolution.candidates[:10]:
            print(f"    - {c.ticker}  CIK{c.cik}  {c.company_name}")
    elif resolution.status == "COMPANY_NOT_FOUND":
        print(f"  reason: {resolution.reason}")
    else:
        print(f"  Resolved company : {resolution.company_name}")
        print(f"  Ticker           : {resolution.ticker}")
        print(f"  CIK              : {resolution.cik}")
        if resolution.latitude is not None and resolution.longitude is not None:
            print(f"  Location         : VERIFIED @ ({resolution.latitude}, {resolution.longitude})  source={resolution.facility_source}")
        else:
            print("  Location         : unavailable")
            if resolution.facility_note:
                print(f"  Location note    : {resolution.facility_note}")
    return resolution


def _verify_sec(sec: SecEdgarClient, ticker: str) -> tuple[bool, list[str]]:
    print(f"\n--- SEC EDGAR XBRL Company Facts (keyless) -- for the RESOLVED ticker {ticker!r} ---")
    print("    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json")
    lines: list[str] = []
    start = time.perf_counter()
    try:
        signals = sec.fetch_financial_signals(ticker)
    except Exception as exc:  # noqa: BLE001 - this script's whole job is to surface exactly what happened
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"    EXCEPTION after {elapsed_ms:.0f}ms: {exc}")
        return False, [f"EXCEPTION: {exc}"]
    elapsed_ms = (time.perf_counter() - start) * 1000
    any_live = False
    for key, signal in signals.items():
        status_word = "LIVE" if signal.live else "UNAVAILABLE"
        any_live = any_live or signal.live
        line = f"    {key}: {status_word}  (source={signal.source}, {elapsed_ms:.0f}ms)"
        print(line)
        print(f"       -> {signal.summary}")
        lines.append(f"{key}={status_word} ({signal.source}): {signal.summary}")
        # Defense-in-depth self-check: a signal reported live must never
        # carry the synthetic_fallback:* / insufficient_data label, and
        # vice versa -- if this script itself ever mislabels a fallback as
        # LIVE, that is a bug in THIS script, not just the client.
        assert signal.live == (signal.source == "sec_edgar_live"), (
            f"verify_live_apis.py internal inconsistency for {key}: "
            f"live={signal.live} but source={signal.source!r}"
        )
    return any_live, lines


def _verify_gibs(gibs: NasaGibsImageryProvider, lat: float, lon: float) -> tuple[bool, list[str]]:
    print(f"\n--- NASA GIBS real bi-temporal satellite tile pair (keyless) -- for the RESOLVED coordinate ({lat}, {lon}) ---")
    print("    https://gibs.earthdata.nasa.gov/wmts/...")
    lines: list[str] = []
    start = time.perf_counter()
    try:
        pair = gibs.fetch_change_pair(lat, lon)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"    EXCEPTION after {elapsed_ms:.0f}ms: {exc}")
        return False, [f"EXCEPTION: {exc}"]
    elapsed_ms = (time.perf_counter() - start) * 1000
    any_live = False
    for label, tile_result in (("current", pair.current), ("prior", pair.prior)):
        status_word = "LIVE" if tile_result.live else "UNAVAILABLE"
        any_live = any_live or tile_result.live
        print(f"    {label}: {status_word}  (source={tile_result.source}, observation_date={tile_result.observation_date}, {elapsed_ms:.0f}ms)")
        print(f"       -> {tile_result.summary}")
        lines.append(f"{label}={status_word} ({tile_result.source}): {tile_result.summary}")
        assert tile_result.live == (tile_result.source == "nasa_gibs_live"), (
            f"verify_live_apis.py internal inconsistency for {label}: "
            f"live={tile_result.live} but source={tile_result.source!r}"
        )
    return any_live, lines


def _verify_census_endpoint(geocoder: CensusGeocodingClient, address: str) -> tuple[bool, list[str]]:
    """A generic, company-INDEPENDENT connectivity check: is the Census
    Bureau's geocoder reachable and returning real results at all, for a
    fixed, well-known address? This proves the endpoint is live. It is
    NOT evidence that any particular company's facility address geocoded
    -- that fact is ``COMPANY LOCATION`` in the verdict, computed only
    from the resolver's own output, never from this check.
    """
    print("\n--- Census Geocoder Endpoint Connectivity Test (keyless, company-independent) ---")
    print("    https://geocoding.geo.census.gov/geocoder/locations/onelineaddress")
    print("    NOTE: this checks the endpoint is up using a fixed reference address --")
    print("    it does NOT mean the run's company address was geocoded (see COMPANY LOCATION below).")
    lines: list[str] = []
    start = time.perf_counter()
    result = geocoder.geocode_address(address)
    elapsed_ms = (time.perf_counter() - start) * 1000
    is_live = result.status == "AVAILABLE" and result.source == "census_geocoder_live"
    print(f"    reference address='{address}' -> {'LIVE' if is_live else 'UNAVAILABLE'}  (source={result.source}, {elapsed_ms:.0f}ms)")
    if is_live:
        print(f"       -> matched '{result.matched_address}' @ ({result.latitude}, {result.longitude})")
    else:
        print(f"       -> {result.note}")
    lines.append(f"address={address!r} status={result.status} source={result.source}: {result.note or 'matched'}")
    return is_live, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--company", default="Tesla",
        help="Free-text company name to resolve end-to-end (ticker/CIK + facility + coordinates) (default: Tesla)",
    )
    parser.add_argument(
        "--address", default="1600 Pennsylvania Ave NW, Washington, DC 20500",
        help="Fixed reference address for the company-independent Census endpoint connectivity test "
             "(default: a well-known verifiable US address). Unrelated to --company's own location.",
    )
    parser.add_argument(
        "--ticker", default=None,
        help="Advanced / analyst override: use this ticker for the SEC EDGAR check instead of the one "
             "the resolver returns for --company. Does NOT affect company resolution itself. Rarely needed.",
    )
    parser.add_argument(
        "--lat", type=float, default=None,
        help="Advanced / analyst override: use this latitude for the NASA GIBS check instead of the "
             "resolver's verified coordinate. Requires --lon too. Does NOT affect company resolution itself.",
    )
    parser.add_argument(
        "--lon", type=float, default=None,
        help="Advanced / analyst override: use this longitude for the NASA GIBS check instead of the "
             "resolver's verified coordinate. Requires --lat too. Does NOT affect company resolution itself.",
    )
    args = parser.parse_args()
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be given together")

    settings = Settings(ORBITALIQ_LIVE_DATA_MODE=True)

    sec = SecEdgarClient(settings=settings)
    geocoder = CensusGeocodingClient(settings)

    # STEP 1: resolve the company FIRST. Nothing below this line uses any
    # ticker or coordinate that did not come from `resolution` (or from an
    # explicit --ticker/--lat/--lon override the user typed for THIS run).
    resolution = _resolve(sec, geocoder, settings, args.company)

    effective_ticker = args.ticker or resolution.ticker
    if args.lat is not None and args.lon is not None:
        effective_lat, effective_lon = args.lat, args.lon
        location_source = "manual override (--lat/--lon)"
    elif resolution.latitude is not None and resolution.longitude is not None:
        effective_lat, effective_lon = resolution.latitude, resolution.longitude
        location_source = "resolver-verified"
    else:
        effective_lat = effective_lon = None
        location_source = None

    print("\n" + "=" * 78)
    print("  STEP 2/3 -- Company-specific live checks (driven only by the resolution above)")
    print("=" * 78)

    if effective_ticker:
        sec_live, _sec_lines = _verify_sec(sec, effective_ticker)
        sec_ran = True
    else:
        print(f"\n--- SEC EDGAR check SKIPPED: no ticker was resolved for {args.company!r} "
              f"(status={resolution.status}) -- never substituting another company's ticker ---")
        sec_live, sec_ran = False, False

    if effective_lat is not None and effective_lon is not None:
        print(f"    (coordinate source: {location_source})")
        gibs_live, _gibs_lines = _verify_gibs(gibs := NasaGibsImageryProvider(settings=settings), effective_lat, effective_lon)
        gibs.close()
        gibs_ran = True
    else:
        print("\n--- SATELLITE: INSUFFICIENT_DATA ---")
        print(f"    reason: no verified company/facility coordinate available for {args.company!r} "
              f"(status={resolution.status}) -- NASA GIBS was never called, and no other company's "
              "coordinate was substituted.")
        gibs_live, gibs_ran = False, False

    print("\n" + "=" * 78)
    print("  STEP 3/3 -- Company-independent endpoint connectivity check")
    print("=" * 78)
    census_endpoint_live, _census_lines = _verify_census_endpoint(geocoder, args.address)

    sec.close()
    geocoder.close()

    company_location_verified = resolution.latitude is not None and resolution.longitude is not None

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  COMPANY RESOLUTION : {resolution.status}"
          + (f"  ({resolution.company_name} / {resolution.ticker} / CIK{resolution.cik})" if resolution.ticker else ""))
    print(f"  COMPANY LOCATION   : {'VERIFIED' if company_location_verified else 'UNVERIFIED'}"
          " -- whether THIS company's own address was geocoded (never another company's).")
    print(f"  SEC EDGAR          : {'LIVE' if sec_live else ('SKIPPED (no ticker)' if not sec_ran else 'UNAVAILABLE')}")
    print(f"  NASA GIBS          : {'LIVE' if gibs_live else ('SKIPPED (no verified coordinate)' if not gibs_ran else 'UNAVAILABLE')}")
    print(f"  CENSUS ENDPOINT    : {'LIVE' if census_endpoint_live else 'UNAVAILABLE'}"
          " -- connectivity only, using a fixed reference address, not this company's.")
    print()
    print("  'LIVE'/'VERIFIED' means the real, keyless public endpoint genuinely")
    print("  responded with usable data in THIS run, on THIS machine, just now, for")
    print("  the company actually requested via --company. 'UNAVAILABLE'/'SKIPPED'")
    print("  means a typed INSUFFICIENT_DATA / COMPANY_NOT_FOUND / RESOLVED_NO_FACILITY")
    print("  result was returned -- never a fabricated value, and never another")
    print("  company's ticker or coordinate substituted in its place.")
    print("=" * 78)

    if resolution.status in ("AMBIGUOUS", "COMPANY_NOT_FOUND"):
        return 2
    full_chain = sec_live and gibs_live and census_endpoint_live and company_location_verified
    all_down = not (sec_live or gibs_live or census_endpoint_live)
    if full_chain:
        return 0
    if all_down:
        return 1
    return 2  # mixed result -- e.g. resolved with no facility, satellite correctly skipped


if __name__ == "__main__":
    raise SystemExit(main())
