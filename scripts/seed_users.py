"""
scripts/seed_users.py
=====================
Wipes and re-seeds realistic expense data for 4 Indian corporate employees.

Employee profiles:
  Rahul Mehta    — Senior Software Engineer, Bengaluru
  Priya Sharma   — Sales Manager,            Mumbai ↔ client travel
  Arjun Kumar    — Junior Developer,          Bengaluru (modest budget)
  Sneha Rao      — Product Manager,           Bengaluru / product off-sites

Anomaly showcase (clearly labelled for demo):
  1. Rahul    : ₹92,500 personal laptop claimed as equipment (10× his norm)
  2. Rahul    : Same Swiggy bill (₹347) submitted twice same day  (duplicate)
  3. Priya    : ₹2,35,000 "client dinner" at a jewellery store    (wrong category + huge amount)
  4. Arjun    : ₹68,000 Goa flight + resort stay claimed as "training" (out-of-policy travel)
  5. Sneha    : Fuel bill at 2:47 AM on a Sunday                  (impossible timing)
  6. Sneha    : Daily food claims every single day for 31 days in Jan (pattern anomaly)

Run:
    python scripts/seed_users.py [--wipe]   (--wipe deletes existing files first)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auth.users import bootstrap
from models.expense import ExpenseRecord, LineItem, TaxEntry

random.seed(99)

_DATA = ROOT / "data"
_BUDGETS_FILE = "budgets.json"

# ── Date window: automatically anchored to current date ──────────────
ORIGINAL_END = date(2026, 3, 22)
WINDOW_END   = date.today()
DATE_SHIFT   = timedelta(days=(WINDOW_END - ORIGINAL_END).days)
WINDOW_START = date(2025, 12, 22) + DATE_SHIFT

def _days() -> list[date]:
    d, out = WINDOW_START, []
    while d <= WINDOW_END:
        out.append(d); d += timedelta(days=1)
    return out

ALL_DAYS      = _days()
WEEKDAYS      = [d for d in ALL_DAYS if d.weekday() < 5]   # Mon–Fri
WEEKENDS      = [d for d in ALL_DAYS if d.weekday() >= 5]  # Sat–Sun
MONTH_GROUPS  = {}   # {(year,month): [dates]}
for d in ALL_DAYS:
    MONTH_GROUPS.setdefault((d.year, d.month), []).append(d)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def _rid() -> str:
    return str(uuid.uuid4())

def _ts(d: date, hh: int = None, mm: int = None) -> tuple[str, str]:
    """Return (date_str, time_str) for a given date."""
    h = hh if hh is not None else random.randint(8, 21)
    m = mm if mm is not None else random.randint(0, 59)
    return d.strftime("%Y-%m-%d"), f"{h:02d}:{m:02d}"

def _gst(amount: float, rate: float = 0.18) -> list[TaxEntry]:
    """Split GST into CGST + SGST at half rate each."""
    half = round(amount * rate / 2, 2)
    return [TaxEntry(rate=rate/2, amount=half), TaxEntry(rate=rate/2, amount=half)]

def _items(desc: str, amount: float, qty: int = 1) -> list[LineItem]:
    return [LineItem(description=desc, quantity=qty,
                     unit_price=round(amount/qty, 2), total_price=round(amount, 2))]

def _reviewed(d: date, status: str, note: str = None):
    """Return review metadata if the claim is old enough to have been reviewed."""
    days_ago = (WINDOW_END - d).days
    if status == "pending":
        return None, None, None
    rev_date = d + timedelta(days=random.randint(1, 4))
    return "vikram_nair", rev_date.isoformat(), note

def _make(
    user_id: str,
    vendor: str,
    category: str,
    subcategory: str,
    amount: float,
    d: date,
    hh: int = None,
    mm: int = None,
    confidence: float = None,
    line_items: list = None,
    taxes: list = None,
    needs_review: bool = False,
    flagged: bool = False,
    flagged_reason: str = None,
    claim_status: str = None,
    reviewed_by: str = None,
    reviewed_at: str = None,
    rejection_note: str = None,
    receipt_number: str = None,
) -> ExpenseRecord:
    if d < WINDOW_START:
        d = d + DATE_SHIFT
    if reviewed_at:
        try:
            r_d = date.fromisoformat(reviewed_at)
            if r_d < WINDOW_START:
                reviewed_at = (r_d + DATE_SHIFT).isoformat()
        except Exception:
            pass

    date_s, time_s = _ts(d, hh, mm)
    days_ago = (WINDOW_END - d).days

    # Auto-assign claim_status based on age if not specified
    if claim_status is None:
        if days_ago > 50:
            claim_status = random.choices(["accepted", "rejected"], weights=[88, 12])[0]
        elif days_ago > 20:
            claim_status = random.choices(["accepted", "pending", "rejected"], weights=[65, 25, 10])[0]
        else:
            claim_status = random.choices(["pending", "accepted"], weights=[65, 35])[0]

    # Auto-fill review metadata
    if claim_status in ("accepted", "rejected") and reviewed_by is None:
        reviewed_by, reviewed_at, rejection_note = _reviewed(d, claim_status, rejection_note)

    conf = confidence if confidence is not None else round(random.uniform(0.82, 0.99), 2)

    return ExpenseRecord(
        user_id=user_id,
        receipt_id=_rid(),
        vendor=vendor,
        date=date_s,
        time=time_s,
        total_amount=round(amount, 2),
        currency="INR",
        category=category,
        subcategory=subcategory,
        receipt_number=receipt_number or f"INV-{random.randint(10000, 99999)}",
        line_items=line_items or _items(f"{vendor} — {subcategory}", amount),
        taxes=taxes or _gst(amount),
        overall_confidence=conf,
        needs_review=needs_review,
        flagged=flagged,
        flagged_reason=flagged_reason,
        source_file=f"receipt_{uuid.uuid4().hex[:8]}.jpg",
        claim_status=claim_status,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        rejection_note=rejection_note,
    )

def _pick(pool: list, n: int) -> list:
    """Pick n random items from pool."""
    return random.sample(pool, min(n, len(pool)))


# ─────────────────────────────────────────────────────────────────────────────
# ── RAHUL MEHTA  |  Senior Software Engineer  |  Bengaluru ──────────────────
# ─────────────────────────────────────────────────────────────────────────────
def make_rahul(uid: str) -> list[ExpenseRecord]:
    records: list[ExpenseRecord] = []

    # ── Swiggy / Zomato lunch, most working days ──
    for d in _pick(WEEKDAYS, 58):
        vendor  = random.choice(["Swiggy", "Zomato"])
        amount  = random.randint(180, 420)
        records.append(_make(uid, vendor, "meals", "food_delivery", amount, d,
                             hh=random.choice([13,14])))

    # ── Evening tea / snack — office days ──
    for d in _pick(WEEKDAYS, 30):
        records.append(_make(uid, "Chai Point", "meals", "tea_snacks",
                             random.randint(40, 120), d, hh=random.choice([16,17])))

    # ── Ola/Uber to office (hybrid — 3 days/week) ──
    wfh_days = [d for d in WEEKDAYS if d.weekday() < 3]   # Mon/Tue/Wed in office
    for d in _pick(wfh_days, 35):
        amt = random.randint(140, 380)
        records.append(_make(uid, random.choice(["Ola", "Uber"]),
                             "transport", "cab_local", amt, d, hh=9))
    for d in _pick(wfh_days, 35):   # evening return
        records.append(_make(uid, random.choice(["Ola", "Uber"]),
                             "transport", "cab_local", random.randint(140, 380), d, hh=19))

    # ── Monthly broadband reimbursement ──
    for m_key, m_days in MONTH_GROUPS.items():
        billing_day = [d for d in m_days if d.day == 5]
        if billing_day:
            records.append(_make(uid, "JioFiber", "internet", "broadband",
                                 1199, billing_day[0], hh=11,
                                 claim_status="accepted",
                                 line_items=_items("JioFiber 150 Mbps - Monthly Plan", 1199),
                                 taxes=_gst(1199),
                                 confidence=0.99))

    # ── Quarterly AWS certification course (Jan) ──
    jan_days = [d for d in ALL_DAYS if d.month == 1]
    if jan_days:
        records.append(_make(uid, "Udemy Business", "training", "online_course",
                             3499, jan_days[5], hh=10,
                             line_items=_items("AWS Solutions Architect – Udemy", 3499),
                             taxes=_gst(3499)))

    # ── Mechanical keyboard (peripheral — within policy) ──
    feb_days = [d for d in ALL_DAYS if d.month == 2]
    if feb_days:
        records.append(_make(uid, "Amazon Business", "equipment", "peripherals",
                             11999, feb_days[3], hh=14,
                             line_items=[
                                 LineItem(description="Keychron K2 Mechanical Keyboard",
                                          quantity=1, unit_price=8999, total_price=8999),
                                 LineItem(description="USB-C Hub 7-in-1",
                                          quantity=1, unit_price=3000, total_price=3000),
                             ],
                             taxes=_gst(11999),
                             confidence=0.97))

    # ── Bengaluru team off-site (hotel stay — within policy) ──
    mar_days = [d for d in ALL_DAYS if d.month == 3]
    if len(mar_days) >= 10:
        records.append(_make(uid, "Lemon Tree Hotel Bengaluru", "accommodation",
                             "hotel_stay", 7800, mar_days[6], hh=15,
                             line_items=_items("Deluxe Room x 2 nights", 7800, 2),
                             taxes=_gst(7800, 0.12)))

    # ═══════════════════════════════════════════════════════════════
    # ANOMALY 1 — AMOUNT SPIKE:  ₹92,500 personal laptop
    #   Rahul's normal equipment claim: ₹5,000–₹15,000
    #   This is a MacBook Pro claimed as "equipment" — 6× his monthly limit
    # ═══════════════════════════════════════════════════════════════
    anomaly_1_day = date(2026, 3, 5)
    records.append(_make(uid, "Apple India Store", "equipment", "laptop_purchase",
                         92500, anomaly_1_day, hh=16,
                         line_items=[
                             LineItem(description="Apple MacBook Pro 14\" M3 Pro 18GB",
                                      quantity=1, unit_price=92500, total_price=92500),
                         ],
                         taxes=_gst(92500),
                         confidence=0.96,
                         needs_review=True,
                         flagged=True,
                         flagged_reason="Amount ₹92,500 is 6× monthly equipment limit (₹15,000). Anomaly score: 0.94",
                         claim_status="pending"))   # still under admin review

    # ═══════════════════════════════════════════════════════════════
    # ANOMALY 2 — DUPLICATE SUBMISSION:  Same Swiggy bill ₹347 twice
    #   Submitted on 2026-02-14; exact same amount + vendor
    # ═══════════════════════════════════════════════════════════════
    dup_day = date(2026, 2, 14)
    dup_amount = 347.0
    dup_receipt = f"SWG-{random.randint(100000,999999)}"
    for i in range(2):
        records.append(_make(uid, "Swiggy", "meals", "food_delivery",
                             dup_amount, dup_day, hh=13 if i == 0 else 13, mm=22 if i == 0 else 51,
                             receipt_number=dup_receipt,
                             confidence=0.98,
                             flagged=(i == 1),
                             flagged_reason=("Duplicate: identical amount ₹347 from Swiggy "
                                             "on same date already exists (receipt SWG-same)" if i == 1 else None),
                             claim_status="accepted" if i == 0 else "rejected",
                             reviewed_by="vikram_nair" if i == 1 else None,
                             reviewed_at=date(2026,2,16).isoformat() if i == 1 else None,
                             rejection_note="Duplicate receipt — already accepted on 2026-02-14 at 13:22" if i == 1 else None))

    records.sort(key=lambda r: r.date or "")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# ── PRIYA SHARMA  |  Sales Manager  |  Mumbai + client travel ───────────────
# ─────────────────────────────────────────────────────────────────────────────
def make_priya(uid: str) -> list[ExpenseRecord]:
    records: list[ExpenseRecord] = []

    # ── Airport taxi + flights (Mumbai ↔ Hyderabad/Delhi/Chennai) ──
    trips = [
        (date(2025,12,26), "IndiGo",     "transport", "domestic_flight", 6800, 7),
        (date(2026,1,8),   "Air India",  "transport", "domestic_flight", 8200, 8),
        (date(2026,1,20),  "IndiGo",     "transport", "domestic_flight", 5600, 8),
        (date(2026,2,3),   "IndiGo",     "transport", "domestic_flight", 7400, 9),
        (date(2026,2,17),  "Air India",  "transport", "domestic_flight", 9100, 7),
        (date(2026,3,3),   "IndiGo",     "transport", "domestic_flight", 6300, 8),
        (date(2026,3,14),  "SpiceJet",   "transport", "domestic_flight", 5900, 7),
    ]
    for trip_date, vendor, cat, subcat, amt, hh in trips:
        records.append(_make(uid, vendor, cat, subcat, amt, trip_date, hh=hh))

    # ── Airport cabs (Ola / Uber) — twice per trip ──
    for trip_date, *_ in trips:
        for h in [6, 22]:  # morning pickup + late night return
            records.append(_make(uid, "Ola", "transport", "cab_local",
                                 random.randint(480, 850), trip_date, hh=h))

    # ── Business hotel stays (2-3 nights per trip) ──
    hotels = [
        (date(2025,12,26), "Novotel Hyderabad Airport",  9800*2,  3),
        (date(2026,1,8),   "The Lalit New Delhi",         13500*3, 12),
        (date(2026,1,20),  "Radisson Blu Chennai",        10200*2, 0),
        (date(2026,2,3),   "Marriott Hyderabad",          11800*2, 15),
        (date(2026,2,17),  "ITC Gardenia Bengaluru",      12400*2, 14),
        (date(2026,3,3),   "Taj Lands End Mumbai",        15600*2, 11),
        (date(2026,3,14),  "Hyatt Regency Pune",           8900*2, 10),
    ]
    for hotel_date, name, amt, hh in hotels:
        records.append(_make(uid, name, "accommodation", "hotel_stay", amt, hotel_date,
                             hh=hh,
                             line_items=_items(f"{name} - Business Suite", amt, 2),
                             taxes=_gst(amt, 0.12)))

    # ── Client business meals (restaurants, lunch + dinner) ──
    client_meal_days = _pick(WEEKDAYS, 45)
    restaurants = [
        ("Karavalli - The Gateway Hotel", 3200, 4800),
        ("Trishna Mumbai",                2800, 4200),
        ("The Table Mumbai",              3500, 5500),
        ("Peshwa Pavilion Pune",          2200, 3800),
        ("Mainland China Hyderabad",      1800, 3200),
        ("Barbecue Nation Bengaluru",     1600, 2800),
        ("Hard Rock Café Bengaluru",      1400, 2400),
    ]
    for d in client_meal_days:
        r_name, lo, hi = random.choice(restaurants)
        records.append(_make(uid, r_name, "meals", "client_entertainment",
                             random.randint(lo, hi), d,
                             hh=random.choice([13, 14, 20, 21])))

    # ── Small client gifts (within ₹5000 policy) ──
    gift_days = _pick(WEEKDAYS, 8)
    for d in gift_days:
        records.append(_make(uid, "Archies Gallery", "client_gifts", "corporate_gift",
                             random.randint(800, 3500), d, hh=17))

    # ═══════════════════════════════════════════════════════════════
    # ANOMALY 3 — WRONG CATEGORY + EXTREME AMOUNT:
    #   ₹2,35,000 at "TBZ – Tribhovandas Bhimji Zaveri" (jewellery) 
    #   submitted as "client_gifts" — 47× her monthly gift limit
    # ═══════════════════════════════════════════════════════════════
    records.append(_make(uid, "TBZ – Tribhovandas Bhimji Zaveri",
                         "client_gifts", "corporate_gift",
                         235000, date(2026, 1, 29), hh=15,
                         line_items=[
                             LineItem(description="Gold Bracelet 22K – 40g",
                                      quantity=1, unit_price=235000, total_price=235000),
                         ],
                         taxes=_gst(235000, 0.03),   # GST on gold = 3%
                         confidence=0.91,
                         needs_review=True,
                         flagged=True,
                         flagged_reason=(
                             "Amount ₹2,35,000 at jewellery store — 47× monthly gift limit (₹5,000). "
                             "Category mismatch: vendor is a jewellery shop, not a gift/hospitality vendor. "
                             "Anomaly score: 0.99"
                         ),
                         claim_status="rejected",
                         reviewed_by="vikram_nair",
                         reviewed_at=date(2026, 1, 31).isoformat(),
                         rejection_note="Rejected: Jewellery purchase is not an eligible business expense. "
                                        "Category 'client_gifts' limit is ₹5,000/month."))

    records.sort(key=lambda r: r.date or "")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# ── ARJUN KUMAR  |  Junior Developer  |  Bengaluru ──────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def make_arjun(uid: str) -> list[ExpenseRecord]:
    records: list[ExpenseRecord] = []

    # ── Daily lunch — Swiggy / office canteen (Mon–Fri, modest) ──
    for d in _pick(WEEKDAYS, 55):
        vendor = random.choice(["Swiggy", "Zomato", "Office Canteen"])
        records.append(_make(uid, vendor, "meals", "food_delivery",
                             random.randint(100, 280), d, hh=random.choice([13,14])))

    # ── Rapido / BMTC bus (commute) ──
    for d in _pick(WEEKDAYS, 50):
        records.append(_make(uid, random.choice(["Rapido", "BMTC Bengaluru"]),
                             "transport", "local_commute",
                             random.randint(18, 120), d, hh=random.choice([9, 18])))

    # ── Home broadband (BSNL) ──
    for m_key, m_days in MONTH_GROUPS.items():
        bill_day = [d for d in m_days if d.day == 10]
        if bill_day:
            records.append(_make(uid, "BSNL Broadband", "internet", "broadband",
                                 849, bill_day[0], hh=11,
                                 claim_status="accepted"))

    # ── Online course (Coursera — Java Spring Boot, Feb) ──
    feb_days = [d for d in ALL_DAYS if d.month == 2 and d.day > 1]
    if feb_days:
        records.append(_make(uid, "Coursera", "training", "online_course",
                             2499, feb_days[2], hh=10,
                             line_items=_items("Spring Boot & Microservices Full Course", 2499),
                             taxes=_gst(2499)))

    # __ Weekend coffee (personal, occasional, small) ──
    for d in _pick(WEEKENDS, 8):
        records.append(_make(uid, "Starbucks", "meals", "cafe",
                             random.randint(240, 480), d, hh=11))

    # ═══════════════════════════════════════════════════════════════
    # ANOMALY 4 — OUT-OF-POLICY TRAVEL:
    #   ₹68,000 Goa trip (flight + resort) claimed as "training"
    #   Junior dev policy: max ₹3,000/month training, ₹3,000/month transport
    #   No prior travel history; total is 22× his monthly transport limit
    # ═══════════════════════════════════════════════════════════════
    goa_date = date(2026, 2, 22)   # Saturday
    records.append(_make(uid, "IndiGo Airlines", "transport", "domestic_flight",
                         8400, goa_date, hh=7,
                         line_items=_items("BLR-GOA-BLR — Round Trip", 8400),
                         taxes=_gst(8400),
                         confidence=0.95,
                         needs_review=True,
                         flagged=True,
                         flagged_reason=(
                             "Domestic flight ₹8,400 — no outstation travel policy for Junior Developer. "
                             "First-ever flight claim; no travel pre-approval found. Anomaly score: 0.88"
                         ),
                         claim_status="pending"))

    records.append(_make(uid, "W Goa Resort & Spa", "accommodation", "hotel_stay",
                         59600, date(2026, 2, 22), hh=15,
                         line_items=_items("Ocean Deluxe Room x 4 nights @ ₹14,900/night", 59600, 4),
                         taxes=_gst(59600, 0.18),   # luxury hotel = 18% GST
                         confidence=0.93,
                         needs_review=True,
                         flagged=True,
                         flagged_reason=(
                             "5-star resort stay ₹59,600 — no accommodation policy for Junior Developer. "
                             "Submitted under category 'training' but vendor is a luxury resort. "
                             "Anomaly score: 0.97"
                         ),
                         claim_status="rejected",
                         reviewed_by="vikram_nair",
                         reviewed_at=date(2026, 2, 25).isoformat(),
                         rejection_note="Rejected: No travel entitlement for Grade L2. "
                                        "W Goa is a leisure resort, not a training venue. "
                                        "Please raise a personal leave request instead."))

    records.sort(key=lambda r: r.date or "")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# ── SNEHA RAO  |  Product Manager  |  Bengaluru + off-sites ─────────────────
# ─────────────────────────────────────────────────────────────────────────────
def make_sneha(uid: str) -> list[ExpenseRecord]:
    records: list[ExpenseRecord] = []

    # ── Team lunches (PM hosts weekly team lunch — Fridays) ──
    fridays = [d for d in WEEKDAYS if d.weekday() == 4]
    for d in fridays:
        amt = random.randint(1800, 5500)  # for a team of ~6
        records.append(_make(uid, random.choice([
            "Mainland China Bengaluru", "Punjab Grill Bengaluru",
            "Social Koramangala", "Truffles Bengaluru", "The Black Pearl Indiranagar"
        ]), "meals", "team_lunch", amt, d, hh=13))

    # ── Solo working lunch (Mon–Thu) ──
    mon_thu = [d for d in WEEKDAYS if d.weekday() < 4]
    for d in _pick(mon_thu, 40):
        records.append(_make(uid, random.choice(["Swiggy", "EatFit", "Faasos"]),
                             "meals", "food_delivery", random.randint(150, 380), d, hh=13))

    # ── Ola to office (hybrid, Mon/Wed/Fri) ──
    mwf = [d for d in WEEKDAYS if d.weekday() in (0, 2, 4)]
    for d in _pick(mwf, 35):
        records.append(_make(uid, "Ola", "transport", "cab_local",
                             random.randint(200, 450), d, hh=9))
    for d in _pick(mwf, 30):
        records.append(_make(uid, "Uber", "transport", "cab_local",
                             random.randint(200, 480), d, hh=19))

    # ── SaaS tool subscriptions (monthly) ──
    tools = [
        ("Figma Professional", 3800, 5),
        ("Miro Team Plan",     2200, 7),
        ("Jira Atlassian",     1600, 3),
        ("Notion Plus",         960, 8),
    ]
    for tool_name, price, bill_day in tools:
        for m_key, m_days in MONTH_GROUPS.items():
            bd = [d for d in m_days if d.day == bill_day]
            if bd:
                records.append(_make(uid, tool_name, "software", "saas_subscription",
                                     price, bd[0], hh=10,
                                     claim_status="accepted",
                                     confidence=0.99))

    # ── ProductCon Bengaluru (conference, Jan) ──
    conf_date = date(2026, 1, 17)
    records.append(_make(uid, "ProductCon Bengaluru 2026", "conference", "industry_event",
                         12500, conf_date, hh=9,
                         line_items=[
                             LineItem(description="ProductCon 2026 — 2-day Pass", quantity=1,
                                      unit_price=10000, total_price=10000),
                             LineItem(description="Workshop: AI for PMs", quantity=1,
                                      unit_price=2500, total_price=2500),
                         ],
                         taxes=_gst(12500),
                         claim_status="accepted"))

    # ── Product off-site — Coorg (2 nights, team of 4) ──
    offsite = date(2026, 2, 7)
    records.append(_make(uid, "Orange County Resort Coorg", "accommodation",
                         "team_offsite", 28800, offsite, hh=14,
                         line_items=_items("Premium Villa x 2 nights (4 pax)", 28800, 2),
                         taxes=_gst(28800, 0.12)))

    # ═══════════════════════════════════════════════════════════════
    # ANOMALY 5 — IMPOSSIBLE TIMING:
    #   Fuel bill at 2:47 AM on a Sunday — company car is in the workshop
    #   Sneha's normal fuel claims are 0 (she uses cabs); first-ever fuel claim
    # ═══════════════════════════════════════════════════════════════
    late_night = date(2026, 1, 26)   # Sunday (Republic Day)
    records.append(_make(uid, "BPCL Filling Station, Mysore Road", "transport",
                         "fuel", 4200, late_night, hh=2, mm=47,
                         line_items=_items("BPCL Petrol — 25.91 L @ ₹107.52/L", 4200),
                         taxes=[],   # fuel tax is embedded (no separate GST line)
                         confidence=0.88,
                         needs_review=True,
                         flagged=True,
                         flagged_reason=(
                             "Fuel expense at 02:47 AM on Sunday — highly unusual timing. "
                             "No prior fuel claims in 3-month history (PM uses company cab policy). "
                             "Anomaly score: 0.91"
                         ),
                         claim_status="pending"))

    # ═══════════════════════════════════════════════════════════════
    # ANOMALY 6 — FREQUENCY PATTERN:
    #   Sneha submits a food delivery claim every single day in January (31 days)
    #   Normal pattern: ~5 claims/week. Anomaly: 31 consecutive daily claims
    # ═══════════════════════════════════════════════════════════════
    months_keys = sorted(MONTH_GROUPS.keys())
    target_month_key = months_keys[-2] if len(months_keys) >= 2 else months_keys[0]
    jan_all = MONTH_GROUPS[target_month_key]
    # First add normal workday claims (already partially added above)
    # Now add weekend + non-typical days to make it every single day
    jan_weekend = [d for d in jan_all if d.weekday() >= 5]
    for d in jan_weekend:
        records.append(_make(uid, "Swiggy", "meals", "food_delivery",
                             random.randint(200, 450), d, hh=random.choice([12, 13, 19, 20]),
                             flagged=True,
                             flagged_reason=(
                                 "Part of a 31-consecutive-day food claim streak in January. "
                                 "Normal weekly frequency is 4–5 claims. "
                                 "Pattern anomaly detected — possible personal food bills claimed. "
                                 "Anomaly score: 0.83"
                             ),
                             confidence=0.94,
                             claim_status="pending",
                             needs_review=True))

    records.sort(key=lambda r: r.date or "")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET WRITER
# ─────────────────────────────────────────────────────────────────────────────

BUDGETS = {
    "user_rahul": {
        "meals":         8000,
        "transport":    10000,
        "accommodation":30000,
        "equipment":    15000,
        "training":      5000,
        "internet":      2000,
    },
    "user_priya": {
        "meals":         20000,
        "transport":     40000,
        "accommodation": 60000,
        "client_gifts":   5000,
        "conference":    10000,
    },
    "user_arjun": {
        "meals":         4000,
        "transport":     3000,
        "internet":      1000,
        "training":      3000,
    },
    "user_sneha": {
        "meals":         12000,
        "transport":     15000,
        "accommodation": 25000,
        "software":      10000,
        "conference":    15000,
        "training":       5000,
    },
}

def write_budgets() -> None:
    for uid, cats in BUDGETS.items():
        user_dir = _DATA / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        budget_path = user_dir / _BUDGETS_FILE
        with open(budget_path, "w", encoding="utf-8") as f:
            json.dump(cats, f, indent=2)
        print(f"   💰 budgets saved → {budget_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY LOG WRITER (pre-populate so the dashboard has data immediately)
# ─────────────────────────────────────────────────────────────────────────────

def write_anomaly_logs(records_by_uid: dict[str, list[ExpenseRecord]]) -> None:
    for uid, records in records_by_uid.items():
        flagged = [r for r in records if r.flagged]
        if not flagged:
            continue
        user_dir = _DATA / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        log_path = user_dir / "anomaly_log.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for r in flagged:
                entry = {
                    "receipt_id":    r.receipt_id,
                    "vendor":        r.vendor,
                    "date":          r.date,
                    "amount":        r.total_amount,
                    "category":      r.category,
                    "anomaly_score": float(r.flagged_reason.split("Anomaly score: ")[-1].rstrip(")").strip())
                                     if r.flagged_reason and "Anomaly score:" in r.flagged_reason
                                     else round(random.uniform(0.78, 0.95), 2),
                    "is_anomaly":    True,
                    "explanation":   r.flagged_reason or "Flagged for review",
                    "signals": [
                        {"feature": "amount_zscore",     "contribution": 0.8, "description": "Amount is unusually high"},
                        {"feature": "vendor_frequency",  "contribution": 0.3, "description": "New or rare vendor"},
                        {"feature": "category_budget_ratio","contribution": 0.6, "description": "Exceeds category budget"},
                    ],
                    "logged_at":     datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"   🔴 anomaly log ({len(flagged)} entries) → {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(wipe: bool = False) -> None:
    print("\n" + "═" * 65)
    print("  EXPENSE INTELLIGENCE — REALISTIC DEMO SEED")
    print("═" * 65)

    # 1. Bootstrap auth (force-update to get positions + limits)
    bootstrap(force=True)
    print("✅ Auth users bootstrapped with positions + monthly limits")

    # 2. Optionally wipe old data
    USERS = ["user_rahul", "user_priya", "user_arjun", "user_sneha"]
    if wipe:
        for uid in USERS:
            p = _DATA / uid
            if p.exists():
                shutil.rmtree(p)
                print(f"🗑  Wiped {p}")

    # 3. Generate + save records
    generators = {
        "user_rahul": make_rahul,
        "user_priya": make_priya,
        "user_arjun": make_arjun,
        "user_sneha": make_sneha,
    }
    all_records: dict[str, list[ExpenseRecord]] = {}

    for uid, gen_fn in generators.items():
        user_dir = _DATA / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = user_dir / "expenses.jsonl"

        records = gen_fn(uid)
        all_records[uid] = records

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

        flagged_n  = sum(1 for r in records if r.flagged)
        accepted_n = sum(1 for r in records if r.claim_status == "accepted")
        pending_n  = sum(1 for r in records if r.claim_status == "pending")
        rejected_n = sum(1 for r in records if r.claim_status == "rejected")
        total_amt  = sum((r.total_amount or 0) for r in records)

        print(f"\n✅ {uid:15s}  {len(records):3d} records  ₹{total_amt:>10,.0f} total")
        print(f"   ✓ accepted={accepted_n}  ⏳ pending={pending_n}  ✗ rejected={rejected_n}  🔴 flagged={flagged_n}")

    print()

    # 4. Write budgets
    write_budgets()

    # 5. Write anomaly logs
    write_anomaly_logs(all_records)

    # 6. Print anomaly summary for demo reference
    print("\n" + "─" * 65)
    print("  ANOMALY SHOWCASE SUMMARY (main project scope)")
    print("─" * 65)
    anomalies = [
        ("AMOUNT SPIKE",  "user_rahul", "₹92,500 MacBook Pro as 'equipment' — 6× monthly limit"),
        ("DUPLICATE",     "user_rahul", "₹347 Swiggy bill submitted twice same day"),
        ("WRONG CAT+AMT", "user_priya", "₹2,35,000 jewellery store as 'client gift' — 47× limit"),
        ("OUT-OF-POLICY", "user_arjun", "₹68,000 Goa trip claimed as 'training' by junior dev"),
        ("TIME ANOMALY",  "user_sneha", "Fuel bill at 02:47 AM on Sunday — impossible timing"),
        ("FREQ ANOMALY",  "user_sneha", "31 consecutive daily food claims in January"),
    ]
    for atype, user, desc in anomalies:
        print(f"  [{atype:14s}] {user:12s} — {desc}")

    print("\n🎉 Seed complete!\n")
    print("  Credentials:")
    print("  ┌──────────────┬──────────────┬────────────────────────────────┐")
    print("  │ Username     │ Password     │ Role / Position                │")
    print("  ├──────────────┼──────────────┼────────────────────────────────┤")
    print("  │ admin        │ admin@123    │ Admin — Finance Controller     │")
    print("  │ rahul_m      │ rahul@2024   │ Senior Software Engineer       │")
    print("  │ priya_s      │ priya@2024   │ Sales Manager                  │")
    print("  │ arjun_k      │ arjun@2024   │ Junior Software Developer      │")
    print("  │ sneha_r      │ sneha@2024   │ Product Manager                │")
    print("  └──────────────┴──────────────┴────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wipe", action="store_true",
                        help="Delete existing user data before seeding")
    args = parser.parse_args()
    main(wipe=args.wipe)
