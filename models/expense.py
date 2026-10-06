"""
Canonical expense record model for the multi-user expense intelligence system.

Uses Pydantic v2 for validation, serialization, and schema enforcement.
Every record is user-scoped via `user_id` and uniquely identified by `receipt_id`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


# ── Sub-models ───────────────────────────────────────────────────────────────

class LineItem(BaseModel):
    """A single purchased item on the receipt."""

    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None


class TaxEntry(BaseModel):
    """A single tax line (e.g. SGST 2.5%)."""

    rate: Optional[float] = None
    amount: Optional[float] = None


# ── Confidence mapping ───────────────────────────────────────────────────────

CONFIDENCE_MAP: dict[str, float] = {
    "Certain": 1.0,
    "High": 0.85,
    "Medium": 0.6,
    "Low": 0.3,
}


def confidence_str_to_float(label: str | None) -> float:
    """Convert a Mindee confidence string to a numeric weight.

    Args:
        label: One of "Certain", "High", "Medium", "Low", or None.

    Returns:
        Float weight between 0.0 and 1.0.  Defaults to 0.0 for unknowns.
    """
    if label is None:
        return 0.0
    return CONFIDENCE_MAP.get(label, 0.0)


# ── Main record ──────────────────────────────────────────────────────────────

class ExpenseRecord(BaseModel):
    """Canonical, user-scoped expense record produced by the normalization pipeline."""

    # ── Identity ─────────────────────────────────────────────────────────
    user_id: str
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ── Core fields ──────────────────────────────────────────────────────
    vendor: Optional[str] = None
    date: Optional[str] = None          # ISO YYYY-MM-DD
    time: Optional[str] = None          # HH:MM
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    receipt_number: Optional[str] = None

    # ── Itemized breakdown ───────────────────────────────────────────────
    line_items: list[LineItem] = Field(default_factory=list)
    taxes: list[TaxEntry] = Field(default_factory=list)

    # ── Confidence / quality ─────────────────────────────────────────────
    raw_confidence: dict[str, str] = Field(default_factory=dict)
    overall_confidence: float = 0.0
    needs_review: bool = False

    # ── Deduplication flags ──────────────────────────────────────────────
    flagged: bool = False
    flagged_reason: Optional[str] = None

    # ── Provenance ───────────────────────────────────────────────────────
    source_file: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Claim status (set by admin review) ───────────────────────────────
    claim_status: str = "pending"      # "pending" | "accepted" | "rejected"
    reviewed_by: Optional[str] = None  # admin username who actioned it
    reviewed_at: Optional[str] = None  # ISO timestamp of review
    rejection_note: Optional[str] = None
