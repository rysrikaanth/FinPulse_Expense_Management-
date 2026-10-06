"""
CLI entry point — generates expense insights for a user.

Usage::

    python insights/generate.py --user_id eval_user_001 --type weekly_summary
    python insights/generate.py --user_id eval_user_001 --type vendor_insight --vendor Swiggy
    python insights/generate.py --user_id eval_user_001 --type budget_advice
    python insights/generate.py --user_id eval_user_001 --type forecast_alert
    python insights/generate.py --user_id eval_user_001 --type anomaly_explanation \\
        --anomaly_id <receipt_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from insights.kpi_builder import build_kpi_context, to_prompt_string
from insights.llm_client import GeminiInsightClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def _load_anomaly_by_id(user_id: str, anomaly_id: str) -> dict | None:
    """Look up a specific anomaly entry from the log by receipt_id."""
    log_path = _DATA_DIR / user_id / "anomaly_log.jsonl"
    if not log_path.exists():
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("receipt_id") == anomaly_id:
                    return entry
            except json.JSONDecodeError:
                continue
    return None


def _build_extra_context(
    args: argparse.Namespace,
    user_id: str,
) -> dict | None:
    """Build the extra_context dict required by specific templates."""
    if args.type == "vendor_insight":
        if not args.vendor:
            print("\033[91m✗ --vendor is required for vendor_insight\033[0m")
            sys.exit(1)

        # Look up vendor stats from KPI context
        kpi = build_kpi_context(user_id)
        vendor_info = next(
            (v for v in kpi.top_vendors if v["vendor"].lower() == args.vendor.lower()),
            None,
        )
        if vendor_info:
            return {
                "vendor": vendor_info["vendor"],
                "total": vendor_info["total"],
                "count": vendor_info["count"],
            }
        else:
            # Vendor exists but not in top 5 — compute from spend
            from storage.expense_store import load_expenses
            from datetime import datetime

            expenses = load_expenses(user_id)
            now = datetime.now()
            total = 0.0
            count = 0
            for exp in expenses:
                if exp.date and exp.total_amount is not None:
                    try:
                        d = datetime.fromisoformat(exp.date)
                        if (d.year == now.year and d.month == now.month
                                and (exp.vendor or "").lower() == args.vendor.lower()):
                            total += exp.total_amount
                            count += 1
                    except ValueError:
                        continue
            return {
                "vendor": args.vendor,
                "total": round(total, 2),
                "count": count,
            }

    elif args.type == "anomaly_explanation":
        if not args.anomaly_id:
            print("\033[91m✗ --anomaly_id is required for anomaly_explanation\033[0m")
            sys.exit(1)

        entry = _load_anomaly_by_id(user_id, args.anomaly_id)
        if entry is None:
            print(f"\033[91m✗ Anomaly {args.anomaly_id} not found in log\033[0m")
            sys.exit(1)

        return {
            "vendor": entry.get("vendor", "Unknown"),
            "date": entry.get("date", ""),
            "amount": entry.get("amount", 0),
            "category": entry.get("category", ""),
            "anomaly_score": entry.get("anomaly_score", 0),
            "system_explanation": entry.get("explanation", ""),
        }

    return None


def _print_result(result, args: argparse.Namespace) -> None:
    """Print the InsightResult with a formatted box."""
    w = 60
    title = args.type.replace("_", " ").upper()

    print()
    print("╔" + "═" * w + "╗")
    print(f"║  EXPENSE INSIGHT — {title:<{w - 21}}║")
    print(f"║  User: {args.user_id:<{w - 28}}| Generated: {result.generated_at[:10]}  ║")
    print(f"║  Cached: {str(result.cached):<{w - 11}}║")
    print("╠" + "═" * w + "╣")
    print(f"║  {'INSIGHT':<{w - 2}}║")
    print("╠" + "═" * w + "╣")
    print()

    # Print insight text
    for line in result.insight_text.strip().split("\n"):
        print(f"  {line}")

    print()
    print("╠" + "═" * w + "╣")
    print(f"║  {'GROUNDING REPORT':<{w - 2}}║")
    print("╠" + "═" * w + "╣")
    print(f"║  Score   : {result.grounding_score:.2f}{' ' * (w - 16)}║")
    print(f"║  Verdict : {result.grounding_verdict:<{w - 13}}║")
    print("╚" + "═" * w + "╝")

    if result.grounding_verdict == "hallucinated":
        print("\033[91m⚠ WARNING: LLM output contains ungrounded numbers. "
              "Review carefully.\033[0m")
    elif result.grounding_verdict == "partial":
        print("\033[93m⚠ Some numbers may not be grounded. Verify against data.\033[0m")

    print()


def main() -> None:
    """CLI entry point for insight generation."""
    parser = argparse.ArgumentParser(
        description="Generate grounded expense insights via Gemini 2.0 Flash",
    )
    parser.add_argument(
        "--user_id", required=True, type=str,
        help="User ID to generate insights for",
    )
    parser.add_argument(
        "--type", required=True, type=str,
        choices=[
            "weekly_summary",
            "anomaly_explanation",
            "budget_advice",
            "vendor_insight",
            "forecast_alert",
        ],
        help="Type of insight to generate",
    )
    parser.add_argument(
        "--vendor", type=str, default=None,
        help="Vendor name (required for vendor_insight)",
    )
    parser.add_argument(
        "--anomaly_id", type=str, default=None,
        help="Receipt ID of the anomaly (required for anomaly_explanation)",
    )

    args = parser.parse_args()

    # Build extra context
    extra_context = _build_extra_context(args, args.user_id)

    # Generate
    print(f"\n  ⏳ Generating {args.type} for user {args.user_id}...")

    try:
        client = GeminiInsightClient()
    except EnvironmentError as e:
        print(f"\033[91m✗ {e}\033[0m")
        sys.exit(1)

    result = client.generate_insight(
        user_id=args.user_id,
        template_name=args.type,
        extra_context=extra_context,
    )

    _print_result(result, args)


if __name__ == "__main__":
    main()
