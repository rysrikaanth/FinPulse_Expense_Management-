"""
Unit tests for the normalization pipeline.

Uses a fixture based on real Mindee V2 API output structure from the
KAISER receipt (receipt.jpg).
"""

from __future__ import annotations

import copy
import pytest

from models.expense import confidence_str_to_float
from pipeline.normalizer import normalize_mindee_json, normalize_vendor


# ── Fixture: realistic Mindee V2 JSON ────────────────────────────────────────

SAMPLE_MINDEE_JSON: dict = {
    "_raw_http": {
        "inference": {
            "result": {
                "fields": {
                    "supplier_name": {
                        "confidence": "Certain",
                        "value": "KAISER",
                    },
                    "supplier_address": {
                        "confidence": "Certain",
                        "value": "D-23, SUPERMART-2, DLF PHASE-4",
                    },
                    "date": {
                        "confidence": "Certain",
                        "value": "2025-03-22",
                    },
                    "time": {
                        "confidence": "Certain",
                        "value": "17:09",
                    },
                    "total_amount": {
                        "confidence": "Certain",
                        "value": 757.0,
                    },
                    "total_net": {
                        "confidence": "High",
                        "value": 1007.0,
                    },
                    "total_tax": {
                        "confidence": "Certain",
                        "value": 33.9,
                    },
                    "receipt_number": {
                        "confidence": "High",
                        "value": "48503",
                    },
                    "purchase_category": {
                        "confidence": "Certain",
                        "value": "food",
                    },
                    "purchase_subcategory": {
                        "confidence": "Certain",
                        "value": "restaurant",
                    },
                    "taxes": {
                        "confidence": "Medium",
                        "items": [
                            {
                                "confidence": "Medium",
                                "fields": {
                                    "rate": {
                                        "confidence": "High",
                                        "value": 2.5,
                                    },
                                    "amount": {
                                        "confidence": "Certain",
                                        "value": 16.95,
                                    },
                                },
                            },
                            {
                                "confidence": "Medium",
                                "fields": {
                                    "rate": {
                                        "confidence": "High",
                                        "value": 2.5,
                                    },
                                    "amount": {
                                        "confidence": "Certain",
                                        "value": 16.95,
                                    },
                                },
                            },
                        ],
                    },
                    "line_items": {
                        "confidence": "High",
                        "items": [
                            {
                                "confidence": "High",
                                "fields": {
                                    "description": {
                                        "confidence": "High",
                                        "value": "Soya Tandoori Chaap ( Tikka )",
                                    },
                                    "quantity": {
                                        "confidence": "Certain",
                                        "value": 2.0,
                                    },
                                    "unit_price": {
                                        "confidence": "High",
                                        "value": 329.0,
                                    },
                                    "total_price": {
                                        "confidence": "Certain",
                                        "value": 658.0,
                                    },
                                },
                            },
                            {
                                "confidence": "High",
                                "fields": {
                                    "description": {
                                        "confidence": "Certain",
                                        "value": "Special Achari Soya Chaap",
                                    },
                                    "quantity": {
                                        "confidence": "Certain",
                                        "value": 1.0,
                                    },
                                    "unit_price": {
                                        "confidence": "High",
                                        "value": 349.0,
                                    },
                                    "total_price": {
                                        "confidence": "Certain",
                                        "value": 349.0,
                                    },
                                },
                            },
                        ],
                    },
                    "locale": {
                        "confidence": "High",
                        "fields": {
                            "language": {
                                "confidence": "High",
                                "value": "en",
                            },
                            "country": {
                                "confidence": "High",
                                "value": "IN",
                            },
                            "currency": {
                                "confidence": "Certain",
                                "value": "INR",
                            },
                        },
                    },
                }
            }
        }
    }
}


# ── Tests: vendor normalization ──────────────────────────────────────────────

class TestNormalizeVendor:
    """Tests for the standalone vendor-name cleaner."""

    def test_all_caps(self):
        assert normalize_vendor("KAISER") == "Kaiser"

    def test_strip_store_number_hash(self):
        assert normalize_vendor("McDONALD'S #342") == "Mcdonald's"

    def test_strip_store_number_dash(self):
        assert normalize_vendor("SUBWAY - 1024") == "Subway"

    def test_none_input(self):
        assert normalize_vendor(None) is None

    def test_empty_string(self):
        assert normalize_vendor("") is None

    def test_already_clean(self):
        assert normalize_vendor("Starbucks") == "Starbucks"


# ── Tests: confidence helpers ────────────────────────────────────────────────

class TestConfidenceConversion:
    """Tests for the confidence string → float mapping."""

    def test_certain(self):
        assert confidence_str_to_float("Certain") == 1.0

    def test_high(self):
        assert confidence_str_to_float("High") == 0.85

    def test_medium(self):
        assert confidence_str_to_float("Medium") == 0.6

    def test_low(self):
        assert confidence_str_to_float("Low") == 0.3

    def test_none(self):
        assert confidence_str_to_float(None) == 0.0

    def test_unknown(self):
        assert confidence_str_to_float("VeryHigh") == 0.0


# ── Tests: full normalization ────────────────────────────────────────────────

class TestNormalizeMindeeJson:
    """Tests that normalize_mindee_json correctly maps the full JSON fixture."""

    @pytest.fixture()
    def record(self):
        return normalize_mindee_json(
            SAMPLE_MINDEE_JSON, user_id="test_user", source_file="receipt.jpg"
        )

    # ── Core field mapping ───────────────────────────────────────────────

    def test_user_id(self, record):
        assert record.user_id == "test_user"

    def test_receipt_id_is_uuid(self, record):
        import uuid
        uuid.UUID(record.receipt_id)  # raises if invalid

    def test_vendor_normalized(self, record):
        assert record.vendor == "Kaiser"

    def test_date(self, record):
        assert record.date == "2025-03-22"

    def test_time(self, record):
        assert record.time == "17:09"

    def test_total_amount(self, record):
        assert record.total_amount == 757.0

    def test_currency(self, record):
        assert record.currency == "INR"

    def test_category(self, record):
        assert record.category == "food"

    def test_subcategory(self, record):
        assert record.subcategory == "restaurant"

    def test_receipt_number(self, record):
        assert record.receipt_number == "48503"

    def test_source_file(self, record):
        assert record.source_file == "receipt.jpg"

    # ── Line items ───────────────────────────────────────────────────────

    def test_line_items_count(self, record):
        assert len(record.line_items) == 2

    def test_first_line_item(self, record):
        li = record.line_items[0]
        assert li.description == "Soya Tandoori Chaap ( Tikka )"
        assert li.quantity == 2.0
        assert li.unit_price == 329.0
        assert li.total_price == 658.0

    def test_second_line_item(self, record):
        li = record.line_items[1]
        assert li.description == "Special Achari Soya Chaap"
        assert li.quantity == 1.0
        assert li.total_price == 349.0

    # ── Taxes ────────────────────────────────────────────────────────────

    def test_taxes_count(self, record):
        assert len(record.taxes) == 2

    def test_tax_entry(self, record):
        t = record.taxes[0]
        assert t.rate == 2.5
        assert t.amount == 16.95

    # ── Confidence ───────────────────────────────────────────────────────

    def test_overall_confidence_positive(self, record):
        assert record.overall_confidence > 0.0

    def test_no_review_needed_when_certain(self, record):
        # Both supplier_name and total_amount are "Certain"
        assert record.needs_review is False

    def test_raw_confidence_has_keys(self, record):
        assert "supplier_name" in record.raw_confidence
        assert "total_amount" in record.raw_confidence

    # ── created_at ───────────────────────────────────────────────────────

    def test_created_at_is_iso(self, record):
        from datetime import datetime
        datetime.fromisoformat(record.created_at)  # raises if invalid


# ── Tests: edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    """Ensure graceful degradation with bad / missing data."""

    def test_empty_json(self):
        record = normalize_mindee_json({}, "u1", "empty.jpg")
        assert record.vendor is None
        assert record.total_amount is None
        assert record.line_items == []
        assert record.taxes == []
        assert record.needs_review is False  # no critical fields to flag

    def test_missing_line_items(self):
        data = copy.deepcopy(SAMPLE_MINDEE_JSON)
        del data["_raw_http"]["inference"]["result"]["fields"]["line_items"]
        record = normalize_mindee_json(data, "u2", "no_items.jpg")
        assert record.line_items == []

    def test_low_confidence_triggers_review(self):
        data = copy.deepcopy(SAMPLE_MINDEE_JSON)
        data["_raw_http"]["inference"]["result"]["fields"]["supplier_name"]["confidence"] = "Low"
        record = normalize_mindee_json(data, "u3", "low_conf.jpg")
        assert record.needs_review is True

    def test_medium_total_triggers_review(self):
        data = copy.deepcopy(SAMPLE_MINDEE_JSON)
        data["_raw_http"]["inference"]["result"]["fields"]["total_amount"]["confidence"] = "Medium"
        record = normalize_mindee_json(data, "u4", "med_total.jpg")
        assert record.needs_review is True
