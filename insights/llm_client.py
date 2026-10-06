"""
Gemini LLM client — generates grounded expense insights.

Uses google-generativeai SDK with gemini-2.0-flash.
All numbers in the output are verified against the ``UserKPIContext``
via the hallucination checker.

Supports caching (24-hour TTL) and RAG retrieval of historical context.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env so GOOGLE_API_KEY is available even when not set as a system env var
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on system environment

from insights.kpi_builder import UserKPIContext, build_kpi_context, to_prompt_string
from insights.prompt_templates import (
    SYSTEM_PROMPT,
    WEEKLY_SUMMARY_TEMPLATE,
    ANOMALY_EXPLANATION_TEMPLATE,
    BUDGET_ADVICE_TEMPLATE,
    VENDOR_INSIGHT_TEMPLATE,
    FORECAST_ALERT_TEMPLATE,
)
from insights.hallucination_checker import check_grounding, GroundingReport
from insights.rag_retriever import RAGRetriever

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

_CACHE_TTL_HOURS = 24


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class InsightResult:
    """Output of a grounded insight generation."""

    insight_text: str
    template_used: str
    kpis_used: list[str]
    grounding_score: float
    grounding_verdict: str
    generated_at: str
    cached: bool


# ── Helper functions ──────────────────────────────────────────────────────────

def _format_alerts(context: UserKPIContext) -> str:
    """Format budget alerts as a bullet list string for template injection."""
    if not context.budget_alerts:
        return "  (No budget alerts)"

    lines: list[str] = []
    for b in context.budget_alerts:
        lines.append(
            f"  - {b['category'].title()}: spent ₹{b['spent_so_far']:,.2f} "
            f"of ₹{b['monthly_limit']:,.2f} ({b['percent_used']:.0f}%) "
            f"[{b['alert_level'].upper()}]"
        )
    return "\n".join(lines)


def _format_breaches(context: UserKPIContext) -> str:
    """Filter forecasts for breach warnings and format as bullet list."""
    breaches = [
        f for f in context.forecast_summary
        if f.get("will_breach", False)
    ]
    if not breaches:
        return "  (No categories forecast to breach)"

    lines: list[str] = []
    for f in breaches:
        days = f.get("days_until_breach", "?")
        lines.append(
            f"  - {f['category'].title()}: predicted ₹{f['predicted_total']:,.2f} "
            f"| breach in {days} days"
        )
    return "\n".join(lines)


# ── Gemini Client ─────────────────────────────────────────────────────────────

class GeminiInsightClient:
    """Generates grounded expense insights via Gemini 2.0 Flash.

    Features:
      - KPI-grounded prompt construction
      - RAG retrieval of historical expense summaries
      - 24-hour response caching
      - Post-generation hallucination checking
    """

    def __init__(self) -> None:
        """Initialise the Gemini client with fallback analytical generation."""
        self._client = None
        self._model_name = "gemini-2.5-flash"
        self._system_instruction = SYSTEM_PROMPT.strip()

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=api_key)
            except Exception:
                self._client = None

    def _generate_rule_based(
        self,
        template_name: str,
        kpi_context: UserKPIContext,
        extra_context: Optional[dict] = None,
    ) -> str:
        """Deterministically generate grounded analytical insight when LLM is unavailable."""
        extra = extra_context or {}
        cats = sorted(kpi_context.spend_by_category.items(), key=lambda x: -x[1])
        top_cats_str = ", ".join([f"{c.title()} (₹{amt:,.0f})" for c, amt in cats[:3]]) if cats else "no recorded categories"
        vendors = [f"{v.get('vendor')} (₹{v.get('total', 0):,.0f})" for v in kpi_context.top_vendors[:3]]
        vendors_str = ", ".join(vendors) if vendors else "no vendor transactions"

        if template_name == "weekly_summary":
            anom_count = len(kpi_context.anomalies_this_month)
            anom_note = (
                f"🚨 {anom_count} anomalous transaction(s) were flagged this period."
                if anom_count else "All expense transactions are within expected norms with no severe anomalies."
            )
            breaches = [b['category'].title() for b in kpi_context.budget_alerts if b.get('will_breach')]
            budget_note = (
                f"Budget alert: {', '.join(breaches)} spending exceeds or is close to exceeding policy limits."
                if breaches else "Spending across all active categories remains within allotted departmental budgets."
            )
            return (
                f"**Executive Spend Summary ({kpi_context.period_start} to {kpi_context.period_end})**\n\n"
                f"• **Total Expenditure:** ₹{kpi_context.total_spend_this_month:,.0f} across all submitted expense claims.\n"
                f"• **Primary Cost Drivers:** {top_cats_str}.\n"
                f"• **Top Commercial Vendors:** {vendors_str}.\n\n"
                f"• **Policy & Compliance:** {anom_note} {budget_note}"
            )
        elif template_name == "anomaly_explanation":
            vendor = extra.get("vendor", "Unknown")
            amount = extra.get("amount", 0)
            score = extra.get("anomaly_score", 0)
            explanation = extra.get("system_explanation", "Claim deviated significantly from historical patterns.")
            return (
                f"**Anomaly Audit Analysis**\n\n"
                f"Transaction for **₹{amount:,.0f}** at **{vendor}** was flagged with an anomaly score of **{score:.2f}**.\n\n"
                f"• **Detection Signal:** {explanation}\n"
                f"• **Recommended Action:** Review supporting receipt line items and departmental limits before approval."
            )
        elif template_name == "budget_advice":
            return (
                f"**Budget Advisory & Optimization**\n\n"
                f"Total spend for the current cycle is ₹{kpi_context.total_spend_this_month:,.0f}.\n\n"
                f"**Active Status:**\n{_format_alerts(kpi_context)}\n\n"
                f"**Recommendation:** Implement pre-approval thresholds on high-frequency expense items."
            )
        elif template_name == "vendor_insight":
            vendor = extra.get("vendor", "Selected Vendor")
            total = extra.get("total", 0)
            count = extra.get("count", 0)
            return (
                f"**Vendor Profile: {vendor}**\n\n"
                f"• Total spend this cycle: ₹{total:,.0f}\n"
                f"• Total claims: {count}\n\n"
                f"• **Review:** Verify that all corporate discounts and rate cards have been applied."
            )
        elif template_name == "forecast_alert":
            return (
                f"**Forecast & Horizon Warning**\n\n"
                f"{_format_breaches(kpi_context)}\n\n"
                f"Projected trend is actively monitored via timeseries forecasting."
            )
        else:
            return (
                f"**Expense Overview**\n\n"
                f"Total spend: ₹{kpi_context.total_spend_this_month:,.0f}. Categories: {top_cats_str}."
            )

    # ── Cache management ─────────────────────────────────────────────────

    @staticmethod
    def _get_cache_path(user_id: str) -> Path:
        """Return the path to the user's insight cache file."""
        return _DATA_DIR / user_id / "insight_cache.jsonl"

    @staticmethod
    def _load_cache(user_id: str) -> list[dict]:
        """Read all cached insights for a user."""
        path = GeminiInsightClient._get_cache_path(user_id)
        if not path.exists():
            return []
        entries: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    @staticmethod
    def _save_to_cache(user_id: str, entry: dict) -> None:
        """Append a single insight result to the cache."""
        path = GeminiInsightClient._get_cache_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    @staticmethod
    def _find_cached(user_id: str, template_name: str) -> Optional[dict]:
        """Find a fresh cached insight (within 24 hours).

        Args:
            user_id:       The user to check.
            template_name: The template that was used.

        Returns:
            Cached entry dict if fresh, else None.
        """
        entries = GeminiInsightClient._load_cache(user_id)
        cutoff = (datetime.utcnow() - timedelta(hours=_CACHE_TTL_HOURS)).isoformat()

        candidates = [
            e for e in entries
            if e.get("template_used") == template_name
            and e.get("generated_at", "") > cutoff
        ]

        if candidates:
            # Return most recent
            candidates.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
            return candidates[0]
        return None

    # ── Prompt construction ──────────────────────────────────────────────

    def _build_prompt(
        self,
        template_name: str,
        kpi_context: UserKPIContext,
        extra_context: Optional[dict],
        rag_summaries: list,
    ) -> str:
        """Construct the final prompt from template + KPIs + RAG context.

        Args:
            template_name: One of the supported template names.
            kpi_context:   Current KPI snapshot.
            extra_context: Additional variables for specific templates.
            rag_summaries: Historical summaries from RAG retriever.

        Returns:
            Complete prompt string ready for Gemini.
        """
        kpi_str = to_prompt_string(kpi_context)
        retriever = RAGRetriever(kpi_context.user_id)
        rag_str = retriever.format_retrieved_context(rag_summaries)

        extra = extra_context or {}

        # Lazy construction: only build the selected template to avoid
        # KeyError from unused templates with missing extra_context keys
        if template_name == "weekly_summary":
            prompt = WEEKLY_SUMMARY_TEMPLATE.format(kpi_context=kpi_str)
        elif template_name == "anomaly_explanation":
            prompt = ANOMALY_EXPLANATION_TEMPLATE.format(
                kpi_context=kpi_str, **extra
            )
        elif template_name == "budget_advice":
            prompt = BUDGET_ADVICE_TEMPLATE.format(
                kpi_context=kpi_str,
                alerts_summary=_format_alerts(kpi_context),
            )
        elif template_name == "vendor_insight":
            prompt = VENDOR_INSIGHT_TEMPLATE.format(
                kpi_context=kpi_str, **extra
            )
        elif template_name == "forecast_alert":
            prompt = FORECAST_ALERT_TEMPLATE.format(
                kpi_context=kpi_str,
                breach_summary=_format_breaches(kpi_context),
            )
        else:
            prompt = WEEKLY_SUMMARY_TEMPLATE.format(kpi_context=kpi_str)

        if rag_str:
            prompt = rag_str + "\n\n" + prompt

        return prompt

    # ── Main entry point ─────────────────────────────────────────────────

    def generate_insight(
        self,
        user_id: str,
        template_name: str,
        extra_context: Optional[dict] = None,
    ) -> InsightResult:
        """Generate a grounded expense insight.

        End-to-end flow:
          1. Check 24-hour cache
          2. Build KPI context from live data
          3. Retrieve similar historical periods (RAG)
          4. Construct grounded prompt and call Gemini
          5. Check response for hallucinated numbers
          6. Store this period for future RAG
          7. Cache and return

        Args:
            user_id:       Target user.
            template_name: Template to use (weekly_summary, etc.).
            extra_context: Additional variables for the template.

        Returns:
            InsightResult with the response text and grounding report.
        """
        # ── Step 1: Cache check ──────────────────────────────────────────
        cached = self._find_cached(user_id, template_name)
        if cached:
            return InsightResult(
                insight_text=cached.get("insight_text", ""),
                template_used=cached.get("template_used", template_name),
                kpis_used=cached.get("kpis_used", []),
                grounding_score=cached.get("grounding_score", 0.0),
                grounding_verdict=cached.get("grounding_verdict", "unknown"),
                generated_at=cached.get("generated_at", ""),
                cached=True,
            )

        # ── Step 2: Build KPI context ────────────────────────────────────
        kpi_context = build_kpi_context(user_id)

        # ── Step 3: RAG retrieval ────────────────────────────────────────
        retriever = RAGRetriever(user_id)
        query = f"{template_name} {kpi_context.period_start}"
        rag_summaries = retriever.retrieve_similar_periods(query, k=3)

        # ── Step 4: Build prompt and call Gemini ─────────────────────────
        prompt = self._build_prompt(
            template_name, kpi_context, extra_context, rag_summaries,
        )

        insight_text = None
        if self._client is not None:
            try:
                from google.genai import types as genai_types
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=self._system_instruction,
                    ),
                )
                insight_text = response.text
            except Exception as e:
                insight_text = None

        if not insight_text:
            insight_text = self._generate_rule_based(template_name, kpi_context, extra_context)

        # ── Step 5: Grounding check ──────────────────────────────────────
        report = check_grounding(insight_text, kpi_context)

        # ── Step 6: Store for future RAG ─────────────────────────────────
        retriever.add_summary(kpi_context)

        # ── Step 7: Cache and return ─────────────────────────────────────
        result = InsightResult(
            insight_text=insight_text,
            template_used=template_name,
            kpis_used=list(kpi_context.__dataclass_fields__.keys()),
            grounding_score=report.grounding_score,
            grounding_verdict=report.verdict,
            generated_at=datetime.utcnow().isoformat(),
            cached=False,
        )
        self._save_to_cache(user_id, asdict(result))
        return result
