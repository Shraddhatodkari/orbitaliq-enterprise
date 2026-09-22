"""Grounding critic — the last gate an LLM-generated narrative passes
through before it reaches the dashboard.

The Report Agent's LLM narrative is only ever supposed to *describe*
numbers that were already computed deterministically elsewhere in the
pipeline (financial metrics, satellite change scores, the momentum score
itself, fiscal years). This module is the enforcement mechanism for that
rule: it extracts every numeric claim from the generated text — dollar
amounts, percentages, "X/100" scores, decimals, and large integers — and
checks each one against an explicit "grounded" set of allowed values built
from the same evidence the LLM was given, within a small rounding
tolerance. Any number that doesn't match anything in that set gets the
narrative BLOCKED; the pipeline then falls back to the deterministic
offline template, which is definitionally grounded (it is built directly
from the numbers, never generated).

This is a deterministic critic, not a second LLM call: no added latency,
no added cost, fully unit-testable, and it cannot itself hallucinate.

Known limitation, stated honestly: this is a heuristic numeric-claims
extractor, not a semantic fact-checker. It cannot catch an LLM that
misattributes a real, grounded number to the wrong company or the wrong
direction (e.g. calling a decline an "increase") — only that every number
it states traces back to real evidence. Pair it with a human skim of the
narrative for anything going into a real client deliverable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

_MONEY_RE = re.compile(r"\$\s*-?\d[\d,]*\.?\d*")
_PERCENT_RE = re.compile(r"-?\d[\d,]*\.?\d*\s*%")
_SCORE_RE = re.compile(r"-?\d[\d,]*\.?\d*\s*/\s*100")
_DECIMAL_RE = re.compile(r"-?\d+\.\d+")
_LARGE_INT_RE = re.compile(r"(?<!\.)\b\d{3,}(?:,\d{3})*\b(?!\.\d)")

DEFAULT_TOLERANCE = 0.6


@dataclass
class GroundingResult:
    passed: bool
    status: str  # "GROUNDED" | "BLOCKED_UNGROUNDED_NUMBER" | "GROUNDED_NO_NUMERIC_CLAIMS"
    flagged_values: list[str] = field(default_factory=list)
    checked_numbers: int = 0

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "status": self.status,
            "flagged_values": self.flagged_values,
            "checked_numbers": self.checked_numbers,
        }


def _extract_numeric_claims(text: str) -> list[float]:
    """Extract every numeric claim, processing "X/100" score mentions
    first and skipping any span already consumed — otherwise the literal
    "100" denominator in "35.4/100" would itself get parsed as a
    standalone number and (correctly, but for the wrong reason) fail to
    match anything grounded.
    """
    consumed: list[tuple[int, int]] = []
    values: list[float] = []

    def _overlaps(span: tuple[int, int]) -> bool:
        return any(a < span[1] and span[0] < b for a, b in consumed)

    for match in _SCORE_RE.finditer(text):
        span = match.span()
        if _overlaps(span):
            continue
        consumed.append(span)
        numerator_text = match.group().split("/")[0]
        raw = re.sub(r"[^\d.\-]", "", numerator_text)
        if raw and raw not in {"-", "."}:
            try:
                values.append(float(raw))
            except ValueError:
                pass

    for pattern in (_MONEY_RE, _PERCENT_RE, _DECIMAL_RE, _LARGE_INT_RE):
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span):
                continue
            consumed.append(span)
            raw = re.sub(r"[^\d.\-]", "", match.group())
            if not raw or raw in {"-", "."}:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return values


def build_grounded_values(*value_groups: Iterable[float | int | None] | float | int | None) -> set[float]:
    """Flatten any number of numeric values / iterables of numeric evidence
    (the momentum score, every financial-metric current/prior/YoY value,
    satellite change scores, fiscal years, coordinates) into one set of
    numbers the critic will accept the narrative citing.
    """
    grounded: set[float] = set()
    for group in value_groups:
        if group is None:
            continue
        if isinstance(group, (int, float)):
            grounded.add(round(float(group), 2))
            continue
        for v in group:
            if v is None:
                continue
            if isinstance(v, (int, float)):
                grounded.add(round(float(v), 2))
    return grounded


def _is_grounded(num: float, grounded_values: set[float], tolerance: float) -> bool:
    # Tolerance scales with magnitude (1% relative, floored at the fixed
    # `tolerance`) so a $383,285,000,000 figure isn't held to the same
    # absolute bar as a 44.13 margin — a flat absolute tolerance would
    # either be far too loose for small ratio-scale numbers or far too
    # tight for large dollar figures. Deliberately NOT auto-rescaling by
    # x100/÷100 here: percent<->ratio conversions are ambiguous to infer
    # blindly (a stray "%" sign) and would make this check too permissive.
    # Callers that pass ratio-unit metrics are expected to add both the
    # raw fraction AND its x100 percent form to `grounded_values` — see
    # ReportAgent's grounded-value construction.
    for g in grounded_values:
        allowed = max(tolerance, 0.01 * max(abs(g), abs(num), 1.0))
        if abs(num - g) <= allowed or abs(-num - g) <= allowed:
            return True
    return False


def check_narrative_grounding(
    text: str, grounded_values: set[float], *, tolerance: float = DEFAULT_TOLERANCE
) -> GroundingResult:
    candidates = _extract_numeric_claims(text)
    if not candidates:
        return GroundingResult(passed=True, status="GROUNDED_NO_NUMERIC_CLAIMS", checked_numbers=0)

    flagged: list[str] = []
    for num in candidates:
        if not _is_grounded(num, grounded_values, tolerance):
            flagged.append(f"{num:g}")

    if flagged:
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        deduped = [v for v in flagged if not (v in seen or seen.add(v))]
        return GroundingResult(
            passed=False, status="BLOCKED_UNGROUNDED_NUMBER", flagged_values=deduped, checked_numbers=len(candidates)
        )
    return GroundingResult(passed=True, status="GROUNDED", checked_numbers=len(candidates))
