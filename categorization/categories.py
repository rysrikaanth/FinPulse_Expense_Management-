"""
Expense taxonomy definition.

Provides the canonical category → subcategory mapping used across the
categorization pipeline, anomaly detection, and user-facing reports.
"""

from __future__ import annotations


# ── Taxonomy ─────────────────────────────────────────────────────────────────

TAXONOMY: dict[str, list[str]] = {
    "Food": [
        "restaurant",
        "grocery",
        "coffee",
        "delivery",
        "fast_food",
        "bakery",
        "street_food",
        "sweets",
    ],
    "Travel": [
        "flights",
        "hotels",
        "train",
        "bus",
        "travel_booking",
        "visa_fees",
    ],
    "Transport": [
        "cab",
        "auto_rickshaw",
        "fuel",
        "parking",
        "metro",
        "toll",
        "bike_rental",
    ],
    "Utilities": [
        "electricity",
        "water",
        "internet",
        "mobile_recharge",
        "gas_cylinder",
        "dth",
    ],
    "Medical": [
        "pharmacy",
        "doctor_consultation",
        "lab_test",
        "hospital",
        "dental",
        "optical",
    ],
    "Shopping": [
        "clothing",
        "electronics",
        "home_appliances",
        "furniture",
        "cosmetics",
        "accessories",
        "general_store",
    ],
    "Entertainment": [
        "movies",
        "concerts",
        "streaming",
        "gaming",
        "sports",
        "amusement_park",
    ],
    "Office": [
        "stationery",
        "printing",
        "coworking",
        "software_subscription",
        "courier",
    ],
    "Education": [
        "books",
        "courses",
        "tuition",
        "exam_fees",
        "coaching",
    ],
    "Other": [
        "miscellaneous",
        "donation",
        "gift",
        "laundry",
        "pet_care",
        "insurance",
    ],
}

# Flat ordered list of main categories (used as label space by the classifier)
MAIN_CATEGORIES: list[str] = list(TAXONOMY.keys())

# Reverse lookup: subcategory → main category
SUBCATEGORY_TO_MAIN: dict[str, str] = {
    sub: main for main, subs in TAXONOMY.items() for sub in subs
}

# All subcategories as a flat set (for validation)
ALL_SUBCATEGORIES: set[str] = set(SUBCATEGORY_TO_MAIN.keys())
