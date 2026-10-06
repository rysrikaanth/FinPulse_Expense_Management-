"""
Per-user category history tracker.

Stores categorization results in ``data/{user_id}/category_history.jsonl``
to track spending patterns, vendor frequency, and enable future per-user
model adaptation.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from categorization.classifier import CategoryResult

# ── Paths ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def _history_path(user_id: str) -> Path:
    """Return (and lazily create) the JSONL history file path for a user."""
    user_dir = _DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "category_history.jsonl"


# ── Public API ───────────────────────────────────────────────────────────────

def update_history(
    user_id: str,
    expense_id: str,
    category_result: CategoryResult,
    vendor: str | None = None,
    amount: float | None = None,
) -> None:
    """Append a categorization event to the user's history.

    Args:
        user_id:         Owner user ID.
        expense_id:      The receipt_id of the categorized expense.
        category_result: Output from the classifier.
        vendor:          Vendor name (optional, stored for analytics).
        amount:          Transaction amount (optional, stored for analytics).
    """
    record = {
        "expense_id": expense_id,
        "main_category": category_result.main_category,
        "subcategory": category_result.subcategory,
        "confidence": category_result.confidence,
        "method": category_result.method,
        "vendor": vendor,
        "amount": amount,
    }

    path = _history_path(user_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_history(user_id: str) -> list[dict[str, Any]]:
    """Load all history entries for a user (internal helper)."""
    path = _history_path(user_id)
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_user_category_distribution(user_id: str) -> dict[str, int]:
    """Count how many expenses fall in each main category for a user.

    Args:
        user_id: The user to query.

    Returns:
        Dict mapping category name → count, sorted descending by count.
    """
    entries = _load_history(user_id)
    counter: Counter[str] = Counter()
    for e in entries:
        counter[e["main_category"]] += 1
    return dict(counter.most_common())


def get_user_top_vendors(
    user_id: str,
    category: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the user's most frequent vendors, optionally filtered by category.

    Args:
        user_id:  The user to query.
        category: If provided, only count vendors in this main category.
        limit:    Maximum number of vendors to return (default 10).

    Returns:
        List of dicts ``{vendor, count, total_spent}``, sorted by count desc.
    """
    entries = _load_history(user_id)

    vendor_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_spent": 0.0}
    )

    for e in entries:
        vendor = e.get("vendor")
        if not vendor:
            continue
        if category and e.get("main_category") != category:
            continue

        vendor_stats[vendor]["count"] += 1
        vendor_stats[vendor]["total_spent"] += e.get("amount", 0.0) or 0.0

    # Sort by count descending
    sorted_vendors = sorted(
        vendor_stats.items(), key=lambda kv: kv[1]["count"], reverse=True
    )

    return [
        {"vendor": v, "count": s["count"], "total_spent": round(s["total_spent"], 2)}
        for v, s in sorted_vendors[:limit]
    ]
