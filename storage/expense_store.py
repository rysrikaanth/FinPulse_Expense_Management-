"""
Flat-file, per-user expense store using JSON-Lines (.jsonl).

Each user's expenses are stored at::

    data/{user_id}/expenses.jsonl

One JSON object per line — easy to append, stream-read, and debug.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from models.expense import ExpenseRecord

# Resolve `data/` relative to the project root (one level up from storage/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def _user_dir(user_id: str) -> Path:
    """Return (and lazily create) the per-user data directory."""
    d = _DATA_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonl_path(user_id: str) -> Path:
    """Return the JSONL file path for a given user."""
    return _user_dir(user_id) / "expenses.jsonl"


# ── Public API ───────────────────────────────────────────────────────────────

def save_expense(record: ExpenseRecord) -> Path:
    """Append an ExpenseRecord as a single JSON line to the user's store.

    Args:
        record: A validated ExpenseRecord.

    Returns:
        The Path to the JSONL file that was written to.
    """
    path = _jsonl_path(record.user_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    return path


def load_expenses(user_id: str) -> list[ExpenseRecord]:
    """Load all expense records for a user.

    Args:
        user_id: The user whose expenses to load.

    Returns:
        List of ExpenseRecord objects (empty list if file doesn't exist).
    """
    path = _jsonl_path(user_id)
    if not path.exists():
        return []

    records: list[ExpenseRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ExpenseRecord.model_validate_json(line))
    return records


def get_expense_by_id(
    user_id: str, receipt_id: str
) -> Optional[ExpenseRecord]:
    """Look up a single expense by its receipt_id.

    Args:
        user_id:    Owner of the record.
        receipt_id: The UUID string to match.

    Returns:
        The matching ExpenseRecord, or None if not found.
    """
    for record in load_expenses(user_id):
        if record.receipt_id == receipt_id:
            return record
    return None
