"""
Deterministic fingerprinting for duplicate receipt detection.

Two complementary strategies:
  - **Exact key**: SHA-256 of (user_id + normalized vendor + date + total_amount).
    Catches re-uploads of the identical receipt.
  - **Fuzzy embedding**: SBERT vector of vendor + sorted line-item descriptions.
    Catches near-duplicates (different photo angles, slight OCR variation).
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from categorization.embedder import embed_text_array

if TYPE_CHECKING:
    from models.expense import ExpenseRecord


def generate_exact_key(expense: "ExpenseRecord") -> str:
    """Produce a deterministic SHA-256 hex digest for exact duplicate matching.

    Key components (all lowercased / stripped):
        user_id  +  vendor_normalized  +  date  +  total_amount

    Args:
        expense: A validated ExpenseRecord.

    Returns:
        64-char hex string.
    """
    vendor = (expense.vendor or "").strip().lower()
    date = (expense.date or "").strip()
    amount = f"{expense.total_amount:.2f}" if expense.total_amount is not None else ""
    user = expense.user_id.strip().lower()

    raw = f"{user}|{vendor}|{date}|{amount}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_fuzzy_embedding(expense: "ExpenseRecord") -> np.ndarray:
    """Produce a 384-dim SBERT embedding for fuzzy duplicate matching.

    Combines vendor name + alphabetically-sorted line-item descriptions
    so that item ordering doesn't affect the vector.

    Args:
        expense: A validated ExpenseRecord.

    Returns:
        np.ndarray of shape (384,), L2-normalized.
    """
    parts: list[str] = []

    if expense.vendor:
        parts.append(expense.vendor.strip())

    descriptions = sorted(
        (li.description.strip() for li in expense.line_items if li.description),
    )
    parts.extend(descriptions)

    combined = " ".join(parts).strip()
    if not combined:
        return np.zeros(384, dtype=np.float32)

    return embed_text_array(combined)
