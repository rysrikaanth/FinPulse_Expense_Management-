"""
Hallucination checker — verifies LLM output numbers against KPI context.

Extracts all numeric values from the LLM response, compares them to the
``UserKPIContext`` source of truth, and produces a ``GroundingReport``
with a score and verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from insights.kpi_builder import UserKPIContext


@dataclass
class GroundingReport:
    """Result of checking an LLM response against grounding data."""

    numbers_in_response: list[str] = field(default_factory=list)
    numbers_grounded: list[str] = field(default_factory=list)
    numbers_ungrounded: list[str] = field(default_factory=list)
    grounding_score: float = 1.0
    verdict: str = "grounded"  # "grounded" | "partial" | "hallucinated"


def check_grounding(
    insight_text: str,
    kpi_context: UserKPIContext,
) -> GroundingReport:
    """Verify that every number in the LLM output is grounded in the KPI data.

    Args:
        insight_text: The raw text returned by the LLM.
        kpi_context:  The KPI snapshot that was injected into the prompt.

    Returns:
        GroundingReport with match details and a score in [0, 1].
    """
    # ── Step 1: Extract all numbers from the LLM response ────────────────
    raw_numbers: set[str] = set()

    # Single pass with priority: currency → comma-numbers → percentage → plain
    # The pattern uses alternation: first try ₹ amounts, then comma-numbers,
    # then percentages, then plain integers/decimals
    combined_pattern = re.compile(
        r"₹[\d,]+(?:\.\d{1,2})?"      # ₹1,234.56 or ₹1234
        r"|\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?"  # 1,234 or 1,234.56
        r"|\d+(?:\.\d{1,2})?%"        # 42.3%
        r"|\d+(?:\.\d{1,2})?"         # 42 or 1234.56
    )
    for m in combined_pattern.finditer(insight_text):
        raw_numbers.add(m.group())

    if not raw_numbers:
        return GroundingReport(
            numbers_in_response=[],
            numbers_grounded=[],
            numbers_ungrounded=[],
            grounding_score=1.0,
            verdict="grounded",
        )

    # Normalise: strip ₹, commas, %
    def _normalise(s: str) -> float | None:
        s = s.replace("₹", "").replace(",", "").replace("%", "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    # ── Step 2: Build reference number set from KPI context ──────────────
    kpi_numbers: set[float] = set()

    kpi_numbers.add(kpi_context.total_spend_this_month)

    total = kpi_context.total_spend_this_month or 1.0
    for cat, v in kpi_context.spend_by_category.items():
        kpi_numbers.add(v)
        # Also add percentage of total (the LLM may report this)
        kpi_numbers.add(round(v / total * 100, 1))

    for vendor in kpi_context.top_vendors:
        t = vendor.get("total", 0)
        c = vendor.get("count", 0) or 1
        kpi_numbers.add(t)
        kpi_numbers.add(float(vendor.get("count", 0)))
        # Add average per visit — LLM correctly derives this
        kpi_numbers.add(round(t / c, 2))

    for a in kpi_context.anomalies_this_month:
        kpi_numbers.add(a.get("amount", 0))
        kpi_numbers.add(a.get("score", 0))

    for b in kpi_context.budget_alerts:
        kpi_numbers.add(b.get("monthly_limit", 0))
        kpi_numbers.add(b.get("spent_so_far", 0))
        kpi_numbers.add(b.get("predicted_total", 0))
        kpi_numbers.add(b.get("percent_used", 0))
        # Remaining budget
        lim = b.get("monthly_limit", 0)
        spent = b.get("spent_so_far", 0)
        if lim > 0:
            kpi_numbers.add(round(lim - spent, 2))

    for f in kpi_context.forecast_summary:
        kpi_numbers.add(f.get("predicted_total", 0))
        if f.get("days_until_breach") is not None:
            kpi_numbers.add(float(f["days_until_breach"]))

    for v in kpi_context.month_over_month_change.values():
        kpi_numbers.add(abs(v))  # absolute value since sign may vary

    for u in kpi_context.unusual_transactions:
        kpi_numbers.add(u.get("amount", 0))
        kpi_numbers.add(u.get("anomaly_score", 0))

    # Add count of anomalies, top vendors count
    kpi_numbers.add(float(len(kpi_context.anomalies_this_month)))
    kpi_numbers.add(float(len(kpi_context.top_vendors)))

    # Discard zeros and None
    kpi_numbers.discard(0.0)

    # ── Step 3: Match each extracted number ──────────────────────────────
    grounded: list[str] = []
    ungrounded: list[str] = []

    for raw in sorted(raw_numbers):
        val = _normalise(raw)
        if val is None:
            continue

        # Skip trivial numbers (0, 1, 2 ... common in prose)
        if val in (0, 1, 2):
            grounded.append(raw)
            continue

        matched = False
        for kpi_val in kpi_numbers:
            if kpi_val == 0:
                continue
            tolerance = abs(val - kpi_val) / (abs(kpi_val) + 1e-6)
            if tolerance <= 0.05:
                matched = True
                break

        if matched:
            grounded.append(raw)
        else:
            ungrounded.append(raw)

    # ── Step 4: Score and verdict ────────────────────────────────────────
    total = len(grounded) + len(ungrounded)
    if total == 0:
        score = 1.0
    else:
        score = len(grounded) / total

    if score >= 0.8:
        verdict = "grounded"
    elif score >= 0.5:
        verdict = "partial"
    else:
        verdict = "hallucinated"

    return GroundingReport(
        numbers_in_response=sorted(raw_numbers),
        numbers_grounded=grounded,
        numbers_ungrounded=ungrounded,
        grounding_score=round(score, 4),
        verdict=verdict,
    )
