"""
Normalization pipeline: raw Mindee V2 JSON → canonical ExpenseRecord.

Handles the nested `_raw_http.inference.result.fields` structure from the
Mindee V2 API and gracefully degrades when fields are missing or malformed.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from models.expense import (
    ExpenseRecord,
    LineItem,
    TaxEntry,
    confidence_str_to_float,
)


# ── Vendor name normalization ────────────────────────────────────────────────

_STORE_NUMBER_RE = re.compile(r"\s*[#\-]\s*\d+\s*$")


def normalize_vendor(raw: str | None) -> str | None:
    """Clean up a vendor / supplier name.

    * Strips trailing store numbers  (e.g. "#342", "- 1024")
    * Converts to title case while preserving common brand quirks
    * Strips surrounding whitespace

    Args:
        raw: The raw supplier_name string from Mindee.

    Returns:
        A cleaned vendor string, or None if input is falsy.
    """
    if not raw:
        return None

    name = raw.strip()
    # Remove trailing store / franchise numbers
    name = _STORE_NUMBER_RE.sub("", name)
    # Title-case (handles "MCDONALD'S" → "Mcdonald'S" → fix apostrophe)
    name = name.title()
    # Fix apostrophe casing: "Mcdonald'S" → "Mcdonald's"
    name = re.sub(r"'S\b", "'s", name)
    return name


# ── Safe field accessors ─────────────────────────────────────────────────────

def _get_field(fields: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return a field dict from the Mindee fields map, or None."""
    return fields.get(key)


def _get_value(fields: dict[str, Any], key: str) -> Any:
    """Extract .value from a Mindee field, or None if missing."""
    field = _get_field(fields, key)
    if field is None:
        return None
    return field.get("value")


def _get_confidence(fields: dict[str, Any], key: str) -> str | None:
    """Extract .confidence from a Mindee field, or None."""
    field = _get_field(fields, key)
    if field is None:
        return None
    return field.get("confidence")


# ── Nested-field extractors ──────────────────────────────────────────────────

def _parse_line_items(fields: dict[str, Any]) -> list[LineItem]:
    """Parse the `line_items.items[].fields` structure into LineItem objects."""
    container = fields.get("line_items")
    if container is None:
        return []

    items_list = container.get("items", [])
    result: list[LineItem] = []
    for item in items_list:
        item_fields = item.get("fields", {})
        result.append(
            LineItem(
                description=_get_value(item_fields, "description"),
                quantity=_get_value(item_fields, "quantity"),
                unit_price=_get_value(item_fields, "unit_price"),
                total_price=_get_value(item_fields, "total_price"),
            )
        )
    return result


def _parse_taxes(fields: dict[str, Any]) -> list[TaxEntry]:
    """Parse the `taxes.items[].fields` structure into TaxEntry objects."""
    container = fields.get("taxes")
    if container is None:
        return []

    items_list = container.get("items", [])
    result: list[TaxEntry] = []
    for item in items_list:
        item_fields = item.get("fields", {})
        result.append(
            TaxEntry(
                rate=_get_value(item_fields, "rate"),
                amount=_get_value(item_fields, "amount"),
            )
        )
    return result


def _parse_currency(fields: dict[str, Any]) -> str | None:
    """Extract currency from the nested `locale.fields.currency.value` path."""
    locale = fields.get("locale")
    if locale is None:
        return None
    locale_fields = locale.get("fields", {})
    currency_field = locale_fields.get("currency")
    if currency_field is None:
        return None
    return currency_field.get("value")


# ── Confidence aggregation ───────────────────────────────────────────────────

_TRACKED_FIELDS = [
    "supplier_name",
    "date",
    "total_amount",
    "purchase_category",
    "receipt_number",
    "time",
]

_CRITICAL_FIELDS = ["supplier_name", "total_amount"]


def _build_confidence(
    fields: dict[str, Any],
) -> tuple[dict[str, str], float, bool]:
    """Compute per-field confidence dict, overall score, and needs_review flag.

    Returns:
        (raw_confidence, overall_confidence, needs_review)
    """
    raw: dict[str, str] = {}
    scores: list[float] = []

    for key in _TRACKED_FIELDS:
        conf_str = _get_confidence(fields, key)
        if conf_str is not None:
            raw[key] = conf_str
            scores.append(confidence_str_to_float(conf_str))

    overall = round(sum(scores) / len(scores), 4) if scores else 0.0

    needs_review = any(
        confidence_str_to_float(raw.get(f)) < 0.85
        for f in _CRITICAL_FIELDS
        if f in raw
    )

    return raw, overall, needs_review


# ── Public API ───────────────────────────────────────────────────────────────

def normalize_mindee_json(
    raw_json: dict[str, Any],
    user_id: str,
    source_file: str,
) -> ExpenseRecord:
    """Convert raw Mindee V2 API output into a canonical ExpenseRecord.

    The expected input shape is the dict produced by
    ``json.load("mindee_output.json")``, which has the structure::

        {
            "_raw_http": {
                "inference": {
                    "result": {
                        "fields": { ... }
                    }
                }
            }
        }

    Args:
        raw_json: Full Mindee response dict (as saved by ``ocr_extractor.py``).
        user_id:  Owner of this expense record.
        source_file: Original receipt filename.

    Returns:
        A validated ExpenseRecord ready for storage.
    """
    # Navigate to the fields map — graceful fallback to empty dict
    fields: dict[str, Any] = (
        raw_json
        .get("_raw_http", {})
        .get("inference", {})
        .get("result", {})
        .get("fields", {})
    )

    raw_confidence, overall_confidence, needs_review = _build_confidence(fields)

    return ExpenseRecord(
        user_id=user_id,
        vendor=normalize_vendor(_get_value(fields, "supplier_name")),
        date=_get_value(fields, "date"),
        time=_get_value(fields, "time"),
        total_amount=_get_value(fields, "total_amount"),
        currency=_parse_currency(fields),
        category=_get_value(fields, "purchase_category"),
        subcategory=_get_value(fields, "purchase_subcategory"),
        receipt_number=str(_get_value(fields, "receipt_number"))
            if _get_value(fields, "receipt_number") is not None
            else None,
        line_items=_parse_line_items(fields),
        taxes=_parse_taxes(fields),
        raw_confidence=raw_confidence,
        overall_confidence=overall_confidence,
        needs_review=needs_review,
        source_file=source_file,
    )
