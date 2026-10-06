"""
LightGBM expense classifier — improved v2.

Improvements over v1:
  - Balanced seed data: 50 examples per class (500 total, up from 192)
  - Extra scalar features: amount_bucket, vendor_word_count, amount_range (3 scalars)
  - Total feature vector: 387 dims (384 SBERT + 3 scalars)
  - Tuned LightGBM hyperparameters for small-data regime
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from categorization.categories import MAIN_CATEGORIES, TAXONOMY
from categorization.embedder import embed_text_array
from models.expense import ExpenseRecord

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "categorizer.joblib"


# ── CategoryResult ────────────────────────────────────────────────────────────

@dataclass
class CategoryResult:
    """Output of the categorization pipeline."""
    main_category: str
    subcategory: str
    confidence: float
    method: str  # "ml" or "fallback"


# ── Seed data — 50 per class, India-centric ───────────────────────────────────

def _generate_seed_data() -> list[dict]:
    rows: list[dict] = []

    def _a(text: str, cat: str, sub: str, amount: float):
        rows.append({"text": text, "category": cat, "subcategory": sub, "amount": amount})

    # ── Food (50) ──────────────────────────────────────────────────────────
    _a("Swiggy butter chicken naan garlic bread", "Food", "delivery", 450)
    _a("Zomato Biryani Paradise chicken biryani raita", "Food", "delivery", 380)
    _a("Swiggy McDonald's McChicken Coke fries", "Food", "delivery", 299)
    _a("Domino's Pizza margarita garlic bread", "Food", "delivery", 550)
    _a("Uber Eats paneer tikka wrap", "Food", "delivery", 320)
    _a("Box8 desi meal thali rice dal", "Food", "delivery", 199)
    _a("Swiggy shawarma roll kebab wrap", "Food", "delivery", 280)
    _a("Zomato Fasoos burrito bowl", "Food", "delivery", 350)
    _a("McDonald's McAloo Tikki burger meal", "Food", "fast_food", 189)
    _a("KFC chicken bucket zinger tower", "Food", "fast_food", 599)
    _a("Burger King Whopper onion rings", "Food", "fast_food", 349)
    _a("Subway footlong veggie delight", "Food", "fast_food", 299)
    _a("Pizza Hut pan pizza cheesy bites", "Food", "fast_food", 449)
    _a("Wendy's burger fries drink combo", "Food", "fast_food", 399)
    _a("Haldiram's namkeen sweets combo", "Food", "sweets", 350)
    _a("Bikanervala rasgulla gulab jamun mithai", "Food", "sweets", 220)
    _a("Anand Sweets kaju katli barfi peda", "Food", "sweets", 400)
    _a("Kaiser Soya Tandoori Chaap Tikka Special Achari Soya", "Food", "restaurant", 757)
    _a("Saravana Bhavan dosa idli vada sambar", "Food", "restaurant", 320)
    _a("Punjab Grill dal makhani paneer butter naan", "Food", "restaurant", 1200)
    _a("Mainland China hakka noodles manchurian soup dimsums", "Food", "restaurant", 1500)
    _a("Barbeque Nation buffet unlimited kebabs starters", "Food", "restaurant", 1800)
    _a("Paradise Biryani dum biryani mirchi ka salan", "Food", "restaurant", 650)
    _a("Copper Chimney mutton rogan josh biryani", "Food", "restaurant", 900)
    _a("Social restaurant craft beer nachos pizza", "Food", "restaurant", 1100)
    _a("Cafe Coffee Day cappuccino sandwich brownie", "Food", "coffee", 280)
    _a("Starbucks latte frappuccino blueberry muffin", "Food", "coffee", 550)
    _a("Third Wave Coffee flat white croissant", "Food", "coffee", 420)
    _a("Chaayos masala chai bun maska", "Food", "coffee", 180)
    _a("Blue Tokai pour over coffee cookies", "Food", "coffee", 350)
    _a("Barista espresso cappuccino pastry", "Food", "coffee", 300)
    _a("BigBasket vegetables fruits milk bread eggs curd", "Food", "grocery", 850)
    _a("Blinkit atta rice dal cooking oil sugar", "Food", "grocery", 620)
    _a("DMart grocery rice wheat flour dal sugar", "Food", "grocery", 1200)
    _a("Reliance Fresh fruits vegetables paneer curd butter", "Food", "grocery", 480)
    _a("Nature's Basket organic honey oats granola muesli", "Food", "grocery", 750)
    _a("More Supermarket snacks biscuits soap detergent", "Food", "grocery", 550)
    _a("JioMart groceries household supplies monthly", "Food", "grocery", 900)
    _a("Zepto instant delivery groceries milk bread", "Food", "grocery", 320)
    _a("Monginis cake pastry birthday cupcakes", "Food", "bakery", 400)
    _a("Theobroma chocolate truffle brownie pastry", "Food", "bakery", 650)
    _a("Karachi Bakery fruit biscuits dry cake rusk", "Food", "bakery", 300)
    _a("French Loaf bakery sourdough baguette", "Food", "bakery", 280)
    _a("Chaat wallah pani puri bhel sev puri aloo tikki", "Food", "street_food", 80)
    _a("Vada pav stall vada pav cutting chai", "Food", "street_food", 50)
    _a("Golgappa vendor soya chaap momos", "Food", "street_food", 120)
    _a("Rolls Corner kathi roll egg roll", "Food", "street_food", 90)
    _a("Biryani by Kilo dum biryani delivery", "Food", "delivery", 420)
    _a("Dunkin Donuts donut coffee coolatta", "Food", "coffee", 250)
    _a("Baskin Robbins ice cream scoop cone", "Food", "sweets", 180)

    # ── Travel (50) ────────────────────────────────────────────────────────
    _a("IndiGo flight ticket Delhi to Mumbai economy", "Travel", "flights", 4500)
    _a("Air India flight Bangalore to Chennai business class", "Travel", "flights", 12000)
    _a("SpiceJet booking Kolkata to Goa economy seat", "Travel", "flights", 5200)
    _a("Vistara flight Delhi to Hyderabad economy cabin", "Travel", "flights", 6500)
    _a("GoAir flight Pune to Jaipur low cost", "Travel", "flights", 3200)
    _a("AirAsia flight Mumbai to Bangkok international", "Travel", "flights", 9800)
    _a("Air India Express Kochi to Dubai economy", "Travel", "flights", 14000)
    _a("IndiGo Bengaluru Chennai morning early bird", "Travel", "flights", 3800)
    _a("MakeMyTrip flight hotel booking international tour", "Travel", "travel_booking", 15000)
    _a("Goibibo hotel flight combo Goa beach package", "Travel", "travel_booking", 12000)
    _a("Yatra.com holiday package Kerala backwaters", "Travel", "travel_booking", 18000)
    _a("Cleartrip flight hotel Rajasthan heritage tour", "Travel", "travel_booking", 9500)
    _a("Thomas Cook holiday package Europe tour", "Travel", "travel_booking", 75000)
    _a("SOTC international holiday honeymoon Maldives", "Travel", "travel_booking", 95000)
    _a("EaseMyTrip domestic flight booking round trip", "Travel", "travel_booking", 7500)
    _a("OYO Rooms hotel stay Jaipur deluxe room", "Travel", "hotels", 1800)
    _a("Taj Hotel room service accommodation premium", "Travel", "hotels", 8500)
    _a("Treebo hotel Bangalore business stay", "Travel", "hotels", 2200)
    _a("Lemon Tree Hotels room booking Hyderabad", "Travel", "hotels", 3500)
    _a("Marriott hotel stay Delhi NCR executive suite", "Travel", "hotels", 12000)
    _a("FabHotels budget stay Jaipur pink city", "Travel", "hotels", 1200)
    _a("IRCTC railway ticket Rajdhani Express 2AC berth", "Travel", "train", 1500)
    _a("IRCTC Shatabdi Express ticket booking CC chair", "Travel", "train", 1200)
    _a("Indian Railways Duronto sleeper class berth", "Travel", "train", 850)
    _a("IRCTC Vande Bharat Express executive chair car", "Travel", "train", 1800)
    _a("IRCTC Humsafar Express 3AC long journey", "Travel", "train", 1100)
    _a("IRCTC Tejas Express Mumbai Goa premium train", "Travel", "train", 2200)
    _a("Indian Railways tatkal booking premium charge", "Travel", "train", 1900)
    _a("RedBus Delhi to Jaipur Volvo AC sleeper", "Travel", "bus", 800)
    _a("AbhiBus Hyderabad to Bangalore sleeper night", "Travel", "bus", 1100)
    _a("KSRTC bus ticket Mysore to Ooty hill station", "Travel", "bus", 450)
    _a("Greenline travels luxury bus Mumbai Pune", "Travel", "bus", 600)
    _a("Neeta Tours national highway bus booking", "Travel", "bus", 700)
    _a("VFS Global visa application fee UK Schengen", "Travel", "visa_fees", 12000)
    _a("BLS International visa fee USA appointment", "Travel", "visa_fees", 15000)
    _a("Agoda hotel booking Bali international", "Travel", "hotels", 7500)
    _a("Booking.com hotel stay Amsterdam Europe", "Travel", "hotels", 9000)
    _a("Airbnb stay Goa villa beachfront", "Travel", "hotels", 5500)
    _a("Zostel hostel dormitory Manali backpacker", "Travel", "hotels", 600)
    _a("Evolve Back resort Coorg luxury stay", "Travel", "hotels", 18000)
    _a("SpiceJet check baggage fee extra luggage", "Travel", "flights", 1000)
    _a("IndiGo seat selection upgrade premium", "Travel", "flights", 500)
    _a("IRCTC platform ticket cancellation refund", "Travel", "train", 200)
    _a("Cox and Kings group tour package Rajasthan", "Travel", "travel_booking", 22000)
    _a("Kesari Tours Singapore Malaysia package", "Travel", "travel_booking", 85000)
    _a("Club Mahindra resort membership holiday", "Travel", "hotels", 25000)
    _a("Sterling resorts hill station stay weekend", "Travel", "hotels", 7000)
    _a("OYO Townhouse stay Bangalore work trip", "Travel", "hotels", 2800)
    _a("Treebo Trend hotel stay Pune business", "Travel", "hotels", 1900)
    _a("IRCTC e-catering food order train meal", "Travel", "train", 350)

    # ── Transport (50) ─────────────────────────────────────────────────────
    _a("Ola Mini cab ride Koramangala to MG Road", "Transport", "cab", 180)
    _a("Uber Go airport drop Indira Gandhi Terminal 3", "Transport", "cab", 350)
    _a("Ola Prime sedan office commute Gurgaon", "Transport", "cab", 250)
    _a("Uber Premier evening ride South Delhi", "Transport", "cab", 420)
    _a("Rapido bike taxi ride Andheri to Bandra", "Transport", "cab", 80)
    _a("Ola Bike quick ride short distance", "Transport", "cab", 50)
    _a("Uber Pool shared cab commute office", "Transport", "cab", 120)
    _a("Meru cab radio taxi airport pickup", "Transport", "cab", 450)
    _a("InDriver cab booking city ride", "Transport", "cab", 200)
    _a("Bluesmart EV cab electric ride Noida", "Transport", "cab", 300)
    _a("Ola Auto rickshaw ride Bangalore", "Transport", "auto_rickshaw", 75)
    _a("Auto rickshaw meter fare Chennai Marina", "Transport", "auto_rickshaw", 60)
    _a("Uber Auto Hyderabad short ride", "Transport", "auto_rickshaw", 45)
    _a("Auto stand local fare Pune", "Transport", "auto_rickshaw", 55)
    _a("Jugnoo auto e-rickshaw Chandigarh ride", "Transport", "auto_rickshaw", 40)
    _a("Indian Oil petrol diesel fuel car filling", "Transport", "fuel", 2500)
    _a("HP Petrol Pump fuel CNG refill car", "Transport", "fuel", 1800)
    _a("Bharat Petroleum diesel car full tank refuel", "Transport", "fuel", 3200)
    _a("Shell petrol station premium fuel bike", "Transport", "fuel", 900)
    _a("HPCL petrol pump fuel fill two-wheeler", "Transport", "fuel", 600)
    _a("DMRC Delhi Metro smart card recharge", "Transport", "metro", 500)
    _a("BMRCL Namma Metro Bangalore token trip", "Transport", "metro", 60)
    _a("Mumbai Metro one day pass monthly commute", "Transport", "metro", 1200)
    _a("Hyderabad Metro MMTS pass monthly", "Transport", "metro", 750)
    _a("Chennai Metro smart card recharge commute", "Transport", "metro", 400)
    _a("NHAI FASTag highway toll charge", "Transport", "toll", 250)
    _a("FASTag toll recharge highway expressway", "Transport", "toll", 500)
    _a("Mumbai Pune expressway toll FASTag", "Transport", "toll", 300)
    _a("NH48 Delhi Jaipur toll booth charge", "Transport", "toll", 180)
    _a("ParkPlus parking mall basement hourly", "Transport", "parking", 100)
    _a("Get My Parking airport paid parking daily", "Transport", "parking", 200)
    _a("Smart park app parking city center", "Transport", "parking", 80)
    _a("Yulu bike rental ride campus hourly", "Transport", "bike_rental", 30)
    _a("Bounce scooter rental hourly Bangalore", "Transport", "bike_rental", 80)
    _a("Vogo scooter two-wheeler rental monthly", "Transport", "bike_rental", 1500)
    _a("MMTS suburban train ticket Hyderabad local", "Transport", "metro", 25)
    _a("Ola ride surge pricing peak hour", "Transport", "cab", 380)
    _a("Uber convenience fee booking charge", "Transport", "cab", 15)
    _a("Scooter rental Goa beach trip daily", "Transport", "bike_rental", 400)
    _a("Royal Enfield bike rental Manali trip", "Transport", "bike_rental", 1200)
    _a("Ola Rentals 8hr outstation hire", "Transport", "cab", 1800)
    _a("Uber Hire 4hr city package sedan", "Transport", "cab", 1200)
    _a("CNG filling station NGT3 vehicle refuel", "Transport", "fuel", 700)
    _a("IOCL fuel pump petrol two-wheeler fill", "Transport", "fuel", 400)
    _a("BluSmart EV fleet subscription monthly", "Transport", "cab", 3000)
    _a("Rapido Captain subscription delivery bike", "Transport", "bike_rental", 500)
    _a("Suburban railway Mumbai CST Dadar local", "Transport", "metro", 30)
    _a("Western Railway monthly pass commuter", "Transport", "metro", 800)
    _a("Parking meter city municipal corporation", "Transport", "parking", 50)
    _a("Airport cab prepaid taxi booth terminal", "Transport", "cab", 600)

    # ── Utilities (50) ─────────────────────────────────────────────────────
    _a("BSES electricity bill payment Delhi monthly", "Utilities", "electricity", 2800)
    _a("Tata Power electricity bill Mumbai quarterly", "Utilities", "electricity", 3500)
    _a("BESCOM electricity Bangalore monthly bill", "Utilities", "electricity", 1800)
    _a("UP Power Corporation electricity bill", "Utilities", "electricity", 1200)
    _a("TORRENT Power electricity Ahmedabad bill", "Utilities", "electricity", 2200)
    _a("Adani Electricity Mumbai bill payment", "Utilities", "electricity", 4000)
    _a("MSEDCL Maharashtra electricity state board", "Utilities", "electricity", 1600)
    _a("TNEB Tamil Nadu electricity board bill", "Utilities", "electricity", 900)
    _a("Delhi Jal Board water bill quarterly", "Utilities", "water", 600)
    _a("BWSSB water bill Bangalore monthly piped", "Utilities", "water", 250)
    _a("Chennai Metrowater board bill payment", "Utilities", "water", 350)
    _a("Brihanmumbai Municipal water bill payment", "Utilities", "water", 300)
    _a("Airtel Fiber broadband 200mbps monthly plan", "Utilities", "internet", 999)
    _a("JioFiber internet 100mbps plan monthly recharge", "Utilities", "internet", 899)
    _a("ACT Fibernet 150mbps broadband bill payment", "Utilities", "internet", 1100)
    _a("BSNL broadband landline bill payment", "Utilities", "internet", 749)
    _a("Hathway cable internet monthly plan", "Utilities", "internet", 650)
    _a("You Broadband WiFi plan monthly subscription", "Utilities", "internet", 799)
    _a("Airtel Xstream prepaid mobile recharge 84 days", "Utilities", "mobile_recharge", 599)
    _a("Jio prepaid recharge unlimited data calling", "Utilities", "mobile_recharge", 449)
    _a("Vi Vodafone Idea monthly recharge plan", "Utilities", "mobile_recharge", 399)
    _a("BSNL postpaid mobile bill payment", "Utilities", "mobile_recharge", 299)
    _a("Airtel postpaid bill corporate plan", "Utilities", "mobile_recharge", 799)
    _a("Jio international roaming pack", "Utilities", "mobile_recharge", 2999)
    _a("Indane LPG gas cylinder booking delivery", "Utilities", "gas_cylinder", 1100)
    _a("Bharat Gas LPG cylinder refill payment", "Utilities", "gas_cylinder", 1050)
    _a("HP Gas booking subsidy cylinder", "Utilities", "gas_cylinder", 1000)
    _a("Piped gas IGL bill Delhi monthly", "Utilities", "gas_cylinder", 850)
    _a("MGL Mahanagar Gas PNG bill Mumbai", "Utilities", "gas_cylinder", 750)
    _a("Tata Sky DTH annual premium pack recharge", "Utilities", "dth", 3600)
    _a("Airtel DTH monthly plan HD sports pack", "Utilities", "dth", 450)
    _a("Dish TV HD recharge cricket platinum", "Utilities", "dth", 350)
    _a("Sun Direct DTH Tamil channels monthly", "Utilities", "dth", 280)
    _a("Videocon D2H recharge monthly pack", "Utilities", "dth", 300)
    _a("CESC Kolkata electricity bill payment", "Utilities", "electricity", 1400)
    _a("Karnataka electricity regulatory KEB bill", "Utilities", "electricity", 1100)
    _a("DVC Damodar Valley electricity bill", "Utilities", "electricity", 800)
    _a("Airtel wifi calling monthly recharge pack", "Utilities", "mobile_recharge", 349)
    _a("Google Fi international data SIM plan", "Utilities", "mobile_recharge", 4000)
    _a("ACT broadband annual upfront payment", "Utilities", "internet", 9000)
    _a("Spectranet fiber internet monthly bill", "Utilities", "internet", 850)
    _a("Tikona broadband wireless plan monthly", "Utilities", "internet", 599)
    _a("Municipal water tanker order payment", "Utilities", "water", 500)
    _a("Rainwater harvesting system maintenance", "Utilities", "water", 200)
    _a("Society maintenance water charge monthly", "Utilities", "water", 400)
    _a("BSNL landline phone bill monthly", "Utilities", "mobile_recharge", 250)
    _a("Tata Docomo postpaid mobile bill", "Utilities", "mobile_recharge", 499)
    _a("Jio Fiber gigabit plan premium monthly", "Utilities", "internet", 1499)
    _a("Electricity generator diesel fuel DG set", "Utilities", "electricity", 5000)
    _a("Smart meter prepaid electricity recharge token", "Utilities", "electricity", 2000)

    # ── Medical (50) ───────────────────────────────────────────────────────
    _a("Apollo Pharmacy medicines paracetamol vitamins", "Medical", "pharmacy", 450)
    _a("MedPlus pharmacy prescription drugs tablets capsules", "Medical", "pharmacy", 320)
    _a("1mg medicines online order antibiotics azithromycin", "Medical", "pharmacy", 580)
    _a("PharmEasy health order supplements protein vitamin D", "Medical", "pharmacy", 750)
    _a("Netmeds online pharmacy vitamins minerals", "Medical", "pharmacy", 420)
    _a("Tata 1mg prescription upload medicine order", "Medical", "pharmacy", 660)
    _a("Wellness Forever pharmacy OTC drugs", "Medical", "pharmacy", 380)
    _a("Guardian pharmacy health supplements", "Medical", "pharmacy", 290)
    _a("Practo doctor consultation online ENT specialist", "Medical", "doctor_consultation", 800)
    _a("Apollo Clinic general physician checkup OPD", "Medical", "doctor_consultation", 600)
    _a("Dermatologist skin clinic acne consultation", "Medical", "doctor_consultation", 1000)
    _a("Psychiatrist mental health consultation session", "Medical", "doctor_consultation", 1500)
    _a("Pediatrician child doctor appointment fees", "Medical", "doctor_consultation", 700)
    _a("Cardiologist heart specialist consultation ECG", "Medical", "doctor_consultation", 1200)
    _a("Orthopedic surgeon consultation bone fracture", "Medical", "doctor_consultation", 900)
    _a("Thyrocare lab test full body checkup blood panel", "Medical", "lab_test", 1500)
    _a("SRL Diagnostics blood test CBC lipid profile", "Medical", "lab_test", 800)
    _a("Dr Lal PathLabs thyroid panel TSH T3 T4", "Medical", "lab_test", 650)
    _a("Metropolis Labs HbA1c diabetes glucose test", "Medical", "lab_test", 500)
    _a("Quest Diagnostics urine culture sensitivity test", "Medical", "lab_test", 700)
    _a("Redcliffe Labs home blood collection", "Medical", "lab_test", 400)
    _a("Fortis Hospital OPD charges admission deposit", "Medical", "hospital", 5000)
    _a("Max Hospital emergency room treatment surgery", "Medical", "hospital", 8000)
    _a("AIIMS Delhi patient charges admission", "Medical", "hospital", 3000)
    _a("Manipal Hospital ward charges nursing", "Medical", "hospital", 6000)
    _a("Medanta ICU charges critical care", "Medical", "hospital", 25000)
    _a("Lenskart spectacles contact lenses prescription frame", "Medical", "optical", 2500)
    _a("Titan Eye Plus glasses coating progressive lens", "Medical", "optical", 3200)
    _a("Vision Express eye exam frame lens fitting", "Medical", "optical", 2800)
    _a("GKB Opticals sunglasses lens replacement", "Medical", "optical", 1800)
    _a("Dental clinic root canal cleaning braces orthodontic", "Medical", "dental", 3500)
    _a("Clove Dental teeth whitening scaling checkup", "Medical", "dental", 1800)
    _a("Sabka Dentist filling extraction treatment", "Medical", "dental", 1200)
    _a("Dant Sansaar braces aligner dental treatment", "Medical", "dental", 8000)
    _a("Cipla medicine inhaler asthma prescription", "Medical", "pharmacy", 500)
    _a("Dr Reddy's medication chronic disease monthly", "Medical", "pharmacy", 1200)
    _a("Sun Pharma prescription drugs derma antibiotic", "Medical", "pharmacy", 800)
    _a("Himalaya healthcare ayurvedic supplement", "Medical", "pharmacy", 350)
    _a("Dabur chyawanprash health tonic", "Medical", "pharmacy", 280)
    _a("Physiotherapy clinic session back pain", "Medical", "doctor_consultation", 1000)
    _a("Homeopathy doctor consultation medicines", "Medical", "doctor_consultation", 600)
    _a("Ayurveda consultation panchakarma treatment", "Medical", "doctor_consultation", 2000)
    _a("COVID PCR test RTPCR certificate", "Medical", "lab_test", 800)
    _a("X-ray radiology chest imaging", "Medical", "lab_test", 600)
    _a("MRI scan brain spine joint radiology", "Medical", "lab_test", 8000)
    _a("Ultrasound sonography abdomen scan", "Medical", "lab_test", 1500)
    _a("ECG EEG cardiac test report", "Medical", "lab_test", 400)
    _a("Vaccination flu shot immunization clinic", "Medical", "doctor_consultation", 900)
    _a("HealthifyMe diet plan annual subscription", "Medical", "doctor_consultation", 3000)
    _a("Ambulance service emergency transport", "Medical", "hospital", 2000)

    # ── Shopping (50) ──────────────────────────────────────────────────────
    _a("Amazon fashion kurta jeans t-shirt delivery", "Shopping", "clothing", 1200)
    _a("Myntra ethnic wear saree salwar suit", "Shopping", "clothing", 2500)
    _a("Zara shirt trousers formal casual wear", "Shopping", "clothing", 3500)
    _a("H&M dress jacket winter clothing", "Shopping", "clothing", 2800)
    _a("Fabindia cotton kurta ethnic indian wear", "Shopping", "clothing", 1800)
    _a("Biba women's ethnic salwar dupatta", "Shopping", "clothing", 2200)
    _a("W for Woman clothing western tops", "Shopping", "clothing", 1500)
    _a("Raymond suit blazer formal fitting", "Shopping", "clothing", 8000)
    _a("Levi's jeans denim jacket purchase", "Shopping", "clothing", 4500)
    _a("Flipkart smartphone OnePlus Samsung mobile", "Shopping", "electronics", 15000)
    _a("Amazon electronics headphones bluetooth JBL", "Shopping", "electronics", 3000)
    _a("Croma laptop charger accessories SSD", "Shopping", "electronics", 8000)
    _a("Reliance Digital LED TV smart television 55inch", "Shopping", "electronics", 45000)
    _a("Vijay Sales washing machine front load", "Shopping", "home_appliances", 18000)
    _a("Amazon home mixer grinder Prestige air purifier", "Shopping", "home_appliances", 4500)
    _a("Bajaj fan water heater geyser appliance", "Shopping", "home_appliances", 3500)
    _a("Crompton air cooler desert fan purchase", "Shopping", "home_appliances", 6000)
    _a("IKEA furniture study desk chair bookshelf", "Shopping", "furniture", 12000)
    _a("Pepperfry sofa bed dining table set", "Shopping", "furniture", 20000)
    _a("Urban Ladder office chair ergonomic desk", "Shopping", "furniture", 8500)
    _a("Godrej interio wardrobe almirah steel cupboard", "Shopping", "furniture", 15000)
    _a("Nykaa cosmetics lakme makeup lipstick foundation", "Shopping", "cosmetics", 1500)
    _a("Bath Body Works lotion shower gel candle set", "Shopping", "cosmetics", 2200)
    _a("Forest Essentials skincare face wash serum", "Shopping", "cosmetics", 3500)
    _a("Biotique herbal face wash moisturizer cream", "Shopping", "cosmetics", 600)
    _a("Tanishq gold jewellery pendant earring chain", "Shopping", "accessories", 8000)
    _a("Titan analog watch leather strap chronograph", "Shopping", "accessories", 5500)
    _a("Fastrack wristwatch strap casual sports", "Shopping", "accessories", 1800)
    _a("Baggit handbag leather purse women", "Shopping", "accessories", 2500)
    _a("General kirana store household items bucket mop", "Shopping", "general_store", 350)
    _a("Kirana shop essentials soap shampoo toothpaste", "Shopping", "general_store", 280)
    _a("Spencer's retail snacks beverages groceries combo", "Shopping", "general_store", 600)
    _a("V-Mart affordable family garments clothing", "Shopping", "clothing", 1200)
    _a("Shoppers Stop lifestyle department store", "Shopping", "clothing", 3200)
    _a("Max Fashion clothing kids adults family", "Shopping", "clothing", 1800)
    _a("Samsung Galaxy mobile phone purchase EMI", "Shopping", "electronics", 25000)
    _a("Apple iPhone purchase EMI upgrade", "Shopping", "electronics", 70000)
    _a("Mi Xiaomi tablet accessories earbuds", "Shopping", "electronics", 5000)
    _a("boAt earphones Airdopes wireless bluetooth", "Shopping", "electronics", 1500)
    _a("OnePlus smartwatch fitness band health", "Shopping", "electronics", 6000)
    _a("Noise colorfit fitness band tracker", "Shopping", "electronics", 2000)
    _a("Wildcraft backpack trekking gear", "Shopping", "accessories", 3000)
    _a("American Tourister trolley bag suitcase", "Shopping", "accessories", 5000)
    _a("VIP luggage travel bag briefcase", "Shopping", "accessories", 4000)
    _a("Cavin Kare Fair Lovely personal care", "Shopping", "cosmetics", 400)
    _a("Himalaya face wash neem cream herbal", "Shopping", "cosmetics", 250)
    _a("VLCC salon service beauty facial", "Shopping", "cosmetics", 1800)
    _a("Nike Adidas sports shoes running", "Shopping", "clothing", 5000)
    _a("Bata shoes formal casual leather", "Shopping", "clothing", 2500)
    _a("Puma track pants jacket sportswear", "Shopping", "clothing", 3000)

    # ── Entertainment (50) ─────────────────────────────────────────────────
    _a("BookMyShow movie ticket PVR IMAX popcorn nachos", "Entertainment", "movies", 750)
    _a("INOX cinema hall movie screening combo snacks", "Entertainment", "movies", 600)
    _a("PVR Cinemas premiere show gold class", "Entertainment", "movies", 1200)
    _a("Cinepolis 4DX movie experience seats", "Entertainment", "movies", 900)
    _a("Netflix monthly subscription premium plan", "Entertainment", "streaming", 649)
    _a("Amazon Prime annual subscription video music", "Entertainment", "streaming", 1499)
    _a("Disney+ Hotstar premium cricket IPL plan", "Entertainment", "streaming", 899)
    _a("Spotify premium music ad-free subscription", "Entertainment", "streaming", 119)
    _a("Sony LIV premium annual sports plan", "Entertainment", "streaming", 999)
    _a("ZEE5 premium subscription annual plan", "Entertainment", "streaming", 599)
    _a("MX Player Gold subscription movies shows", "Entertainment", "streaming", 299)
    _a("YouTube Premium ad-free music background", "Entertainment", "streaming", 129)
    _a("BookMyShow concert live music Arijit show", "Entertainment", "concerts", 2500)
    _a("Zomato Live comedy show standup Zakir Khan", "Entertainment", "concerts", 1500)
    _a("NH7 Weekender music festival tickets", "Entertainment", "concerts", 3500)
    _a("Sunday Soul Sante art music festival Bangalore", "Entertainment", "concerts", 500)
    _a("Sunburn EDM festival Goa electronic music", "Entertainment", "concerts", 5000)
    _a("Steam game purchase Counter-Strike gaming", "Entertainment", "gaming", 1200)
    _a("PlayStation PS5 game subscription PS Plus", "Entertainment", "gaming", 3000)
    _a("Xbox Game Pass ultimate monthly gold", "Entertainment", "gaming", 699)
    _a("Dream11 fantasy cricket IPL season contest", "Entertainment", "sports", 500)
    _a("Decathlon badminton racket sports gear kit", "Entertainment", "sports", 2000)
    _a("Wonderla Bangalore amusement park ticket rides", "Entertainment", "amusement_park", 1200)
    _a("Imagicaa theme park Maharashtra fun entry", "Entertainment", "amusement_park", 1500)
    _a("EsselWorld water park family entry", "Entertainment", "amusement_park", 2000)
    _a("MyFitnessPal Premium annual fitness plan", "Entertainment", "sports", 3000)
    _a("Cult.fit gym membership monthly sports", "Entertainment", "sports", 2500)
    _a("Anytime Fitness gym annual membership", "Entertainment", "sports", 12000)
    _a("Hotstar sports cricket ODI IPL broadcast", "Entertainment", "streaming", 499)
    _a("Lionsgate Play movies web series subscription", "Entertainment", "streaming", 199)
    _a("Apple TV+ streaming service monthly", "Entertainment", "streaming", 99)
    _a("Audible audiobook subscription premium", "Entertainment", "streaming", 599)
    _a("Gaana music streaming subscription premium", "Entertainment", "streaming", 168)
    _a("JioSaavn music app subscription annual", "Entertainment", "streaming", 299)
    _a("BookMyShow sports IPL match ticket", "Entertainment", "sports", 3000)
    _a("Paytm Insider standup comedy event", "Entertainment", "concerts", 800)
    _a("IMAX VR experience gaming centre", "Entertainment", "gaming", 600)
    _a("Gaming cafe PC console hourly session", "Entertainment", "gaming", 200)
    _a("Pool billiards gaming center session", "Entertainment", "gaming", 150)
    _a("Laser tag arena family gaming center", "Entertainment", "amusement_park", 400)
    _a("Snow World Hyderabad entry ice park", "Entertainment", "amusement_park", 800)
    _a("VR Zone virtual reality gaming experience", "Entertainment", "gaming", 500)
    _a("Smaaash entertainment sports gaming zone", "Entertainment", "amusement_park", 700)
    _a("Fun Republic mall play zone kids", "Entertainment", "amusement_park", 350)
    _a("GoldAdlabs multiplex movie seat booking", "Entertainment", "movies", 450)
    _a("Miraj Cinemas regional language show", "Entertainment", "movies", 300)
    _a("Cinepolis BIGF Bollywood premiere show", "Entertainment", "movies", 700)
    _a("BookMyShow play drama theatre performance", "Entertainment", "concerts", 600)
    _a("Pro Kabaddi PKL match ticket arena", "Entertainment", "sports", 1500)
    _a("ISL football match ticket stadium", "Entertainment", "sports", 1200)

    # ── Office (50) ────────────────────────────────────────────────────────
    _a("Stationery mart notebooks pens pencils rulers", "Office", "stationery", 250)
    _a("Amazon office supplies sticky notes whiteboard marker", "Office", "stationery", 400)
    _a("Classmate notebooks spiral registers files folders", "Office", "stationery", 180)
    _a("Staples India office stationery bulk order", "Office", "stationery", 1200)
    _a("DOMS Navneet stationery pen set art", "Office", "stationery", 350)
    _a("Faber-Castell color pencils highlighters set", "Office", "stationery", 450)
    _a("Cello pens ballpoint gel ink set", "Office", "stationery", 120)
    _a("Print shop photocopying binding spiral A4 sheets", "Office", "printing", 150)
    _a("Printo digital printing business cards brochure", "Office", "printing", 600)
    _a("HP printer ink cartridge replacement", "Office", "printing", 800)
    _a("Canon toner cartridge printer refill", "Office", "printing", 1200)
    _a("Epson printer paper ream A4 80gsm", "Office", "printing", 500)
    _a("WeWork coworking space monthly hot desk", "Office", "coworking", 8000)
    _a("91springboard shared office cabin meeting", "Office", "coworking", 6000)
    _a("Innov8 coworking day pass flex desk", "Office", "coworking", 500)
    _a("DBS coworking Bengaluru monthly plan", "Office", "coworking", 7000)
    _a("BHIVE coworking Bangalore monthly dedicated", "Office", "coworking", 9000)
    _a("Notion team plan annual SaaS subscription", "Office", "software_subscription", 4800)
    _a("Zoho CRM business annual subscription", "Office", "software_subscription", 15000)
    _a("Slack Pro workspace annual plan messaging", "Office", "software_subscription", 3600)
    _a("GitHub Pro developer annual subscription", "Office", "software_subscription", 800)
    _a("Salesforce CRM monthly enterprise plan", "Office", "software_subscription", 12000)
    _a("Microsoft 365 Office annual subscription", "Office", "software_subscription", 6000)
    _a("Google Workspace Business Plus monthly", "Office", "software_subscription", 2400)
    _a("Jira Atlassian project management monthly", "Office", "software_subscription", 1800)
    _a("Canva Pro annual design subscription", "Office", "software_subscription", 3999)
    _a("Adobe Creative Cloud monthly plan", "Office", "software_subscription", 4230)
    _a("Blue Dart courier parcel express delivery", "Office", "courier", 250)
    _a("DTDC courier speed post document parcel", "Office", "courier", 180)
    _a("Delhivery shipment dispatch last mile", "Office", "courier", 120)
    _a("FedEx international courier DHL delivery", "Office", "courier", 1500)
    _a("India Post registered parcel speed post", "Office", "courier", 80)
    _a("Shadowfax last mile delivery shipment", "Office", "courier", 90)
    _a("AWS Amazon cloud hosting monthly bill", "Office", "software_subscription", 8000)
    _a("Heroku platform cloud deploy monthly", "Office", "software_subscription", 2000)
    _a("DigitalOcean server droplet monthly hosting", "Office", "software_subscription", 1500)
    _a("Namecheap domain registration annual", "Office", "software_subscription", 800)
    _a("GoDaddy domain hosting web renewal", "Office", "software_subscription", 1200)
    _a("Zoom Pro video call plan annual", "Office", "software_subscription", 10000)
    _a("Trello Gold project management board", "Office", "software_subscription", 500)
    _a("Dropbox Business storage team plan", "Office", "software_subscription", 4200)
    _a("Freshdesk customer support subscription", "Office", "software_subscription", 3000)
    _a("Postman API tool team plan annual", "Office", "software_subscription", 2500)
    _a("Figma professional design subscription", "Office", "software_subscription", 4500)
    _a("Office stationery printer paper stapler", "Office", "stationery", 600)
    _a("Whiteboard markers dry erase office", "Office", "stationery", 200)
    _a("Visiting card printing bulk 500 pieces", "Office", "printing", 350)
    _a("Lamination binding presentation folder", "Office", "printing", 200)
    _a("OkCredit billing POS receipt paper", "Office", "stationery", 100)
    _a("Registrar of Companies filing fee MCA", "Office", "software_subscription", 3000)

    # ── Education (50) ─────────────────────────────────────────────────────
    _a("Amazon books programming textbook Python Java", "Education", "books", 600)
    _a("Flipkart study material UPSC IAS preparation", "Education", "books", 800)
    _a("Crossword bookstore fiction autobiography history", "Education", "books", 450)
    _a("Kitabay used books second-hand textbooks school", "Education", "books", 300)
    _a("Oxford Bookstore academic reference books", "Education", "books", 700)
    _a("Landmark bookstore novels comics children", "Education", "books", 500)
    _a("Penguin Random House ebook Kindle purchase", "Education", "books", 350)
    _a("Udemy online course web development React", "Education", "courses", 450)
    _a("Coursera specialization data science ML certificate", "Education", "courses", 3500)
    _a("Unacademy Plus UPSC SSC competitive exam", "Education", "courses", 5000)
    _a("BYJU'S learning platform subscription class 10 CBSE", "Education", "courses", 12000)
    _a("Great Learning AI ML data course certificate", "Education", "courses", 8000)
    _a("upskill edtech bootcamp coding live class", "Education", "courses", 15000)
    _a("Simplilearn PMP certification training online", "Education", "courses", 20000)
    _a("Pluralsight IT skill learning subscription", "Education", "courses", 3000)
    _a("LinkedIn Learning professional skill annual", "Education", "courses", 2500)
    _a("Khan Academy donation premium subscription", "Education", "courses", 500)
    _a("upGrad MBA online degree program fees", "Education", "tuition", 250000)
    _a("School annual tuition fees Class 9 CBSE", "Education", "tuition", 15000)
    _a("College semester fees B.Tech engineering", "Education", "tuition", 50000)
    _a("Amity University semester fee quarterly", "Education", "tuition", 35000)
    _a("DPS school term fee transport hostel", "Education", "tuition", 25000)
    _a("IIT Madras online BSc degree semester", "Education", "tuition", 20000)
    _a("NTA NEET exam registration application fee", "Education", "exam_fees", 1500)
    _a("GATE IIT exam registration department fee", "Education", "exam_fees", 1800)
    _a("CAT MBA entrance exam application", "Education", "exam_fees", 2200)
    _a("UPSC civil services Prelims registration", "Education", "exam_fees", 100)
    _a("JEE Main NIT exam application online", "Education", "exam_fees", 650)
    _a("CLAT law entrance exam registration", "Education", "exam_fees", 4000)
    _a("IELTS British Council exam test fee", "Education", "exam_fees", 16500)
    _a("TOEFL ETS exam registration center", "Education", "exam_fees", 14000)
    _a("GRE exam Prometric registration fee", "Education", "exam_fees", 18000)
    _a("GMAT exam center MBA graduate admission", "Education", "exam_fees", 22500)
    _a("Allen coaching IIT JEE batch monthly", "Education", "coaching", 35000)
    _a("FIITJEE coaching physics chemistry maths", "Education", "coaching", 8000)
    _a("Aakash Institute NEET medical biology", "Education", "coaching", 15000)
    _a("Vedantu online tutor class 10 CBSE", "Education", "coaching", 3000)
    _a("Toppr adaptive learning subscription", "Education", "coaching", 2500)
    _a("Doubtnut JEE NEET doubt solving premium", "Education", "coaching", 1500)
    _a("ClassPlus tutor subscription monthly batch", "Education", "coaching", 1000)
    _a("Resonance Kota coaching institute fee", "Education", "coaching", 40000)
    _a("Sri Chaitanya coaching AP Telangana", "Education", "coaching", 12000)
    _a("Narayana coaching group study material", "Education", "coaching", 9000)
    _a("Career Launcher CAT MBA coaching class", "Education", "coaching", 18000)
    _a("T.I.M.E. Institute MBA GRE coaching", "Education", "coaching", 15000)
    _a("Motion Education JEE coaching batch", "Education", "coaching", 30000)
    _a("Board exam model question paper set", "Education", "books", 200)
    _a("NCERT textbook set Class 11 12 science", "Education", "books", 400)
    _a("Arihant publication competition books set", "Education", "books", 900)
    _a("Oswaal sample paper CBSE board prep", "Education", "books", 350)

    # ── Other (50) ─────────────────────────────────────────────────────────
    _a("Temple hundi donation charitable offering", "Other", "donation", 500)
    _a("NGO charity Goonj CRY fund online", "Other", "donation", 1000)
    _a("Akshaya Patra Foundation donation meal", "Other", "donation", 2500)
    _a("Smile Foundation donate child education", "Other", "donation", 800)
    _a("PM CARES Fund online contribution", "Other", "donation", 5000)
    _a("SBI Yono charity donation transfer", "Other", "donation", 1500)
    _a("Amazon gift card birthday present", "Other", "gift", 2000)
    _a("Archies greeting card gift hamper flowers", "Other", "gift", 600)
    _a("Flipkart gift voucher special occasion", "Other", "gift", 3000)
    _a("Ferns N Petals flower bouquet gift chocolate", "Other", "gift", 1200)
    _a("Cadbury chocolate gift box Diwali hamper", "Other", "gift", 800)
    _a("Prestige dry cleaning suit jacket laundry", "Other", "laundry", 350)
    _a("Urban Company laundry pickup home delivery", "Other", "laundry", 500)
    _a("Washmart laundry subscription monthly plan", "Other", "laundry", 1200)
    _a("Uclean laundry kg-wise wash fold service", "Other", "laundry", 400)
    _a("Dhobiking online laundry order steam iron", "Other", "laundry", 300)
    _a("PetStar dog food Royal Canin cat litter", "Other", "pet_care", 1200)
    _a("Supertails pet grooming nail trim bath", "Other", "pet_care", 800)
    _a("Heads Up For Tails pet accessories toys", "Other", "pet_care", 1500)
    _a("Mr n Mrs Pet grooming home visit", "Other", "pet_care", 700)
    _a("PetKonnect veterinary dog vaccination", "Other", "pet_care", 600)
    _a("LIC life insurance premium annual payment", "Other", "insurance", 15000)
    _a("ICICI Lombard health insurance cashless renewal", "Other", "insurance", 8000)
    _a("Star Health family floater premium annual", "Other", "insurance", 12000)
    _a("Bajaj Allianz vehicle two-wheeler insurance", "Other", "insurance", 3000)
    _a("TATA AIG car comprehensive insurance renewal", "Other", "insurance", 7000)
    _a("Max Life term insurance annual premium", "Other", "insurance", 20000)
    _a("HDFC Life policy renewal installment", "Other", "insurance", 25000)
    _a("National Insurance mediclaim renewal", "Other", "insurance", 5000)
    _a("New India Assurance property insurance", "Other", "insurance", 4000)
    _a("Unknown vendor miscellaneous purchase", "Other", "miscellaneous", 200)
    _a("ATM cash withdrawal bank", "Other", "miscellaneous", 5000)
    _a("Random receipt unrecognized vendor", "Other", "miscellaneous", 150)
    _a("Sundry expenses petty cash office", "Other", "miscellaneous", 500)
    _a("Bank charges maintenance fee penalty", "Other", "miscellaneous", 100)
    _a("Notary legal document stamp paper", "Other", "miscellaneous", 1000)
    _a("Society maintenance apartment monthly charges", "Other", "miscellaneous", 3000)
    _a("Club membership residential recreational", "Other", "miscellaneous", 5000)
    _a("Religious pooja samagri material ritual", "Other", "donation", 800)
    _a("Political party fund donation receipt", "Other", "donation", 2000)
    _a("School alumni donation annual fund", "Other", "donation", 1000)
    _a("Wedding anniversary gift flowers ring", "Other", "gift", 5000)
    _a("Housewarming gift home decor present", "Other", "gift", 2500)
    _a("Laundromat coin-operated self-service wash", "Other", "laundry", 200)
    _a("Dry clean leather jacket silk saree", "Other", "laundry", 600)
    _a("Pet food Pedigree Drools dog kibble bag", "Other", "pet_care", 900)
    _a("Animal shelter adoption fee pet rescue", "Other", "pet_care", 1000)
    _a("General office miscellaneous petty expense", "Other", "miscellaneous", 300)
    _a("Household repair plumber electrician call", "Other", "miscellaneous", 800)
    _a("Courier charges document personal parcel", "Other", "miscellaneous", 120)

    return rows


# ── Feature engineering ───────────────────────────────────────────────────────

def _amount_bucket(amount: float | None) -> float:
    """log2(amount + 1) — ~14 natural bins for ₹1–₹50,000."""
    if amount is None or amount <= 0:
        return 0.0
    return math.log2(amount + 1)


def _amount_range_bin(amount: float | None) -> int:
    """Coarse price-range bucket (0–5) to help separate categories.

    0: ≤₹100   1: ≤₹500   2: ≤₹2 000   3: ≤₹10 000   4: ≤₹50 000   5: >₹50 000
    """
    if amount is None or amount <= 0:
        return 0
    if amount <= 100:    return 0
    if amount <= 500:    return 1
    if amount <= 2000:   return 2
    if amount <= 10000:  return 3
    if amount <= 50000:  return 4
    return 5


def _vendor_word_count(vendor: str | None) -> int:
    """Whitespace-delimited token count in vendor name."""
    if not vendor:
        return 0
    return len(vendor.strip().split())


def _build_feature_vector(
    embedding: np.ndarray, amount: float | None, vendor: str | None
) -> np.ndarray:
    """SBERT (384) + amount_bucket + amount_range_bin + vendor_word_count = 387 dims."""
    scalars = np.array(
        [_amount_bucket(amount), float(_amount_range_bin(amount)), float(_vendor_word_count(vendor))],
        dtype=np.float32,
    )
    return np.concatenate([embedding, scalars])


# ── Classifier wrapper ────────────────────────────────────────────────────────

class ExpenseCategorizer:
    """LightGBM-backed expense categorizer with SBERT + scalar features."""

    def __init__(self):
        self.model: Optional[LGBMClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._is_trained = False

    def train(self, seed_data: list[dict] | None = None) -> "ExpenseCategorizer":
        """Train on seed_data (or built-in dataset if None).

        Args:
            seed_data: List of dicts with keys text, category, subcategory, amount.
        """
        if seed_data is None:
            seed_data = _generate_seed_data()

        texts   = [r["text"]     for r in seed_data]
        labels  = [r["category"] for r in seed_data]
        amounts = [r["amount"]   for r in seed_data]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(MAIN_CATEGORIES)
        y = self.label_encoder.transform(labels)

        X = np.array(
            [_build_feature_vector(embed_text_array(t), a, t.split()[0])
             for t, a in zip(texts, amounts)],
            dtype=np.float32,
        )

        self.model = LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=20,
            min_child_samples=3,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)
        self._is_trained = True
        return self

    def categorize_expense(self, expense: ExpenseRecord, user_id: str) -> CategoryResult:
        """Categorize an ExpenseRecord; falls back to Mindee if confidence < 0.6.

        Args:
            expense: Validated ExpenseRecord.
            user_id: Owner user ID.

        Returns:
            CategoryResult with main_category, subcategory, confidence, method.
        """
        if not self._is_trained:
            raise RuntimeError("Categorizer not trained. Call .train() first.")

        parts = [expense.vendor] if expense.vendor else []
        parts += [li.description for li in expense.line_items if li.description]
        combined = " ".join(parts).strip() or "unknown"

        vec = _build_feature_vector(
            embed_text_array(combined), expense.total_amount, expense.vendor
        )

        probs      = self.model.predict_proba(vec.reshape(1, -1))[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_cat   = self.label_encoder.inverse_transform([pred_idx])[0]
        subcategory = self._infer_subcategory(pred_cat, expense)

        if confidence < 0.6 and expense.category:
            return CategoryResult(
                main_category=self._normalize_mindee_category(expense.category),
                subcategory=expense.subcategory or "miscellaneous",
                confidence=round(confidence, 4),
                method="fallback",
            )

        return CategoryResult(
            main_category=pred_cat,
            subcategory=subcategory,
            confidence=round(confidence, 4),
            method="ml",
        )

    def save_model(self, path: str | Path | None = None) -> Path:
        """Persist model + label_encoder via joblib.

        Args:
            path: Destination .joblib file path.
        """
        path = Path(path) if path else _DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "label_encoder": self.label_encoder}, path)
        return path

    def load_model(self, path: str | Path | None = None) -> "ExpenseCategorizer":
        """Load a saved model from disk.

        Args:
            path: Path to the .joblib file.
        """
        path = Path(path) if path else _DEFAULT_MODEL_PATH
        data = joblib.load(path)
        self.model          = data["model"]
        self.label_encoder  = data["label_encoder"]
        self._is_trained    = True
        return self

    @staticmethod
    def _normalize_mindee_category(raw: str) -> str:
        title = raw.strip().title()
        return title if title in MAIN_CATEGORIES else "Other"

    @staticmethod
    def _infer_subcategory(main_cat: str, expense: ExpenseRecord) -> str:
        valid = TAXONOMY.get(main_cat, ["miscellaneous"])
        if expense.subcategory and expense.subcategory in valid:
            return expense.subcategory
        return valid[0]

    def get_seed_data(self) -> list[dict]:
        """Return the built-in seed dataset."""
        return _generate_seed_data()

    def predict_raw(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (label_array, probability_matrix) for a feature matrix.

        Args:
            X: Feature matrix of shape (n, 387).
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained.")
        probs = self.model.predict_proba(X)
        preds = self.label_encoder.inverse_transform(np.argmax(probs, axis=1))
        return preds, probs
