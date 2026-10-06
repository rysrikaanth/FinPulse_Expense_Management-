"""
Duplicate resolution — decides what action to take for a potential duplicate.

Actions:
  - ``reject``  — exact duplicate, do not save again
  - ``flag``    — fuzzy match with high confidence, save with flagged=True
  - ``accept``  — no duplicates found, save normally
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deduplication.detector import DuplicateMatch, detector
from models.expense import ExpenseRecord


@dataclass
class DuplicateResolution:
    """Decision output from the resolver."""
    is_duplicate: bool
    duplicate_of: Optional[str]     # receipt_id of the matched duplicate
    action: str                     # "reject" | "flag" | "accept"
    reason: str


def resolve_duplicates(
    expense: ExpenseRecord,
    user_id: str,
    fuzzy_threshold: float = 0.12,
    flag_similarity: float = 0.85,
) -> DuplicateResolution:
    """Determine whether an expense is a duplicate and decide the action.

    Decision logic:
      1. Exact match found          → action = ``reject``
      2. Fuzzy match score ≥ 0.85   → action = ``flag``  (needs human review)
      3. No match / low similarity  → action = ``accept``

    Args:
        expense:          The new (unsaved) ExpenseRecord.
        user_id:          Owner user ID.
        fuzzy_threshold:  L2 distance threshold passed to the detector.
        flag_similarity:  Minimum similarity score to trigger a ``flag``.

    Returns:
        A DuplicateResolution with the recommended action.
    """
    matches = detector.find_duplicates(expense, user_id, threshold=fuzzy_threshold)

    if not matches:
        return DuplicateResolution(
            is_duplicate=False,
            duplicate_of=None,
            action="accept",
            reason="No duplicate found in user history",
        )

    best = matches[0]  # sorted by similarity desc

    # ── Exact match → reject ─────────────────────────────────────────────
    if best.match_type == "exact":
        return DuplicateResolution(
            is_duplicate=True,
            duplicate_of=best.matched_expense_id,
            action="reject",
            reason=f"Exact duplicate of {best.matched_expense_id}: {best.reason}",
        )

    # ── High-confidence fuzzy match → flag for review ────────────────────
    if best.similarity_score >= flag_similarity:
        return DuplicateResolution(
            is_duplicate=True,
            duplicate_of=best.matched_expense_id,
            action="flag",
            reason=(
                f"Potential duplicate of {best.matched_expense_id} "
                f"(similarity={best.similarity_score:.2%}): {best.reason}"
            ),
        )

    # ── Low-confidence fuzzy match → accept ──────────────────────────────
    return DuplicateResolution(
        is_duplicate=False,
        duplicate_of=None,
        action="accept",
        reason=f"Closest match {best.matched_expense_id} below flag threshold ({best.similarity_score:.2%})",
    )
