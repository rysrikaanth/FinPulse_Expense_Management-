# Expense Intelligence System — Project Context

> **Last updated:** 2026-03-22
> **Python:** 3.13 · **Runtime:** venv at `./venv/`
> **API key:** Mindee V2 (`MINDEE_API_KEY` in `.env`)

---

## 1. What This Project Is

A **multi-user expense intelligence system** that processes receipt images into structured, categorized, and deduplicated expense records. It is **user-centric** — every record, index, and history file is scoped to a `user_id`.

### High-level pipeline

```
Receipt Image
    │
    ▼
┌─────────────────────┐
│  OCR (Mindee V2)    │  ← ocr_extractor.py
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Normalizer         │  ← pipeline/normalizer.py
│  Raw JSON → Record  │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Deduplication      │  ← deduplication/resolver.py
│  Exact + Fuzzy      │     (reject / flag / accept)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Storage (JSONL)    │  ← storage/expense_store.py
│  data/{user}/       │     expenses.jsonl
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Categorization     │  ← categorization/classifier.py
│  SBERT + LightGBM   │     (ML or Mindee fallback)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  User History       │  ← categorization/user_history.py
│  Per-user analytics │     data/{user}/category_history.jsonl
└─────────────────────┘
```

The main CLI entry point is **`pipeline/ingest.py`**:
```bash
python pipeline/ingest.py --image receipt.jpg --user_id user_001
```

---

## 2. Complete File Tree

```
minor_project/
├── .env                          # MINDEE_API_KEY=<key>
├── .gitignore
├── requirements.txt              # All Python dependencies
├── receipt.jpg                   # Sample receipt for testing
├── mindee_output.json            # Cached Mindee V2 API response (sample)
│
├── ocr_extractor.py              # Mindee V2 ClientV2 wrapper
│
├── models/
│   ├── __init__.py
│   └── expense.py                # Pydantic v2 ExpenseRecord model
│
├── pipeline/
│   ├── __init__.py
│   ├── normalizer.py             # Mindee JSON → ExpenseRecord mapping
│   └── ingest.py                 # CLI entry point (OCR → normalize → dedup → store)
│
├── storage/
│   ├── __init__.py
│   └── expense_store.py          # Per-user JSONL flat-file storage
│
├── categorization/
│   ├── __init__.py
│   ├── categories.py             # Taxonomy: 10 main categories, ~60 subcategories
│   ├── embedder.py               # SBERT wrapper (all-MiniLM-L6-v2, 384-dim)
│   ├── classifier.py             # LightGBM classifier (500 seed examples, 387-dim features)
│   ├── user_history.py           # Per-user category tracking (JSONL)
│   └── evaluate.py               # 5-fold CV evaluation script
│
├── deduplication/
│   ├── __init__.py
│   ├── fingerprinter.py          # SHA-256 exact key + SBERT fuzzy embedding
│   ├── detector.py               # Faiss flat L2 index + date/amount guards
│   ├── resolver.py               # Action decisions: reject / flag / accept
│   └── evaluate.py               # Precision@K evaluation (synthetic pairs)
│
├── tests/
│   ├── __init__.py
│   └── test_normalizer.py        # 36 unit tests for the normalizer
│
├── data/                         # Runtime data (gitignored ideally)
│   └── {user_id}/
│       ├── expenses.jsonl        # Stored expense records
│       ├── category_history.jsonl# Category tracking events
│       ├── faiss.index           # Persisted Faiss L2 index
│       └── index_map.json        # receipt_id list + exact-key set for index
│
└── venv/                         # Python 3.13 virtual environment
```

---

## 3. Core Data Model

**`models/expense.py`** — Pydantic v2 `ExpenseRecord`:

| Field | Type | Description |
|---|---|---|
| `user_id` | `str` | Owner user (scoping key) |
| `receipt_id` | `str` | UUID v4 (auto-generated) |
| `vendor` | `str?` | Normalized vendor name (title case, store # stripped) |
| `date` | `str?` | ISO `YYYY-MM-DD` |
| `time` | `str?` | `HH:MM` |
| `total_amount` | `float?` | Total transaction amount |
| `currency` | `str?` | e.g. `"INR"` |
| `category` | `str?` | From Mindee or ML classifier |
| `subcategory` | `str?` | From Mindee or ML classifier |
| `receipt_number` | `str?` | Receipt/invoice number |
| `line_items` | `list[LineItem]` | `{description, quantity, unit_price, total_price}` |
| `taxes` | `list[TaxEntry]` | `{rate, amount}` |
| `raw_confidence` | `dict[str, str]` | Per-field Mindee confidence labels |
| `overall_confidence` | `float` | Averaged numeric confidence (0–1) |
| `needs_review` | `bool` | True if any critical field < 0.85 confidence |
| `flagged` | `bool` | True if dedup resolver flagged as potential duplicate |
| `flagged_reason` | `str?` | Explanation for the flag |
| `source_file` | `str?` | Original image filename |
| `created_at` | `str` | ISO timestamp (UTC) |

Sub-models: `LineItem(description, quantity, unit_price, total_price)`, `TaxEntry(rate, amount)`.

---

## 4. Module Details

### 4.1 OCR Layer — `ocr_extractor.py`

- Uses **Mindee V2 API** (`ClientV2`) with `enqueue_and_get_result` (async flow)
- Model ID: `a3536038-e793-4f2f-a2a0-4cac5e236927`
- Returns full JSON via `response.__dict__` — nested at `_raw_http.inference.result.fields`
- Config flags: `rag=True`, `raw_text=True`, `polygon=True`, `confidence=True`

### 4.2 Normalization — `pipeline/normalizer.py`

- Navigates `_raw_http.inference.result.fields` to extract all field values
- **Vendor cleaning**: strips trailing `#123` / `- 456`, title-cases, fixes apostrophes
- Extracts `line_items` from `.items[].fields` with `description/quantity/unit_price/total_price`
- Extracts `taxes` similarly
- Currency comes from `locale.fields.currency.value`
- **Confidence calculation**: averages confidence across 6 tracked fields; flags `needs_review` if supplier_name or total_amount confidence < 0.85
- Tested with **36 unit tests** in `tests/test_normalizer.py`

### 4.3 Storage — `storage/expense_store.py`

- **JSONL flat files** at `data/{user_id}/expenses.jsonl`
- One JSON object per line (append-only via `save_expense()`)
- `load_expenses(user_id)` returns `list[ExpenseRecord]`
- `get_expense_by_id(user_id, receipt_id)` for single lookups

### 4.4 Categorization — `categorization/`

**Taxonomy** (`categories.py`):
- 10 main categories: Food, Travel, Transport, Utilities, Medical, Shopping, Entertainment, Office, Education, Other
- ~60 subcategories total (India-centric: Swiggy, IRCTC, Ola, BigBasket, etc.)
- `MAIN_CATEGORIES`, `TAXONOMY`, `SUBCATEGORY_TO_MAIN`, `ALL_SUBCATEGORIES` exports

**Embedder** (`embedder.py`):
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim output)
- Lazy-loaded on first call, LRU-cached embeddings
- `embed_text_array(text) → np.ndarray` — the core function used across modules
- `embed_expense(expense) → np.ndarray` — combines vendor + line-item descriptions

**Classifier** (`classifier.py`):
- **LightGBM** trained on **500 synthetic India-centric examples** (50 per class, balanced)
- Feature vector: 384 SBERT + `amount_bucket` (log₂) + `amount_range_bin` (0–5) + `vendor_word_count` = **387 dims**
- Hyperparams: 400 estimators, max_depth=6, lr=0.05, 20 leaves, L1/L2 reg, balanced weights
- `categorize_expense(expense, user_id) → CategoryResult(main_category, subcategory, confidence, method)`
- **Fallback**: if ML confidence < 0.6, returns Mindee's raw category instead (`method="fallback"`)
- Model persisted at `models/categorizer.joblib` via `save_model()` / `load_model()`
- Latest evaluation: **72.6% ± 4.0% mean accuracy (5-fold CV)**

**User History** (`user_history.py`):
- Appends categorization events to `data/{user_id}/category_history.jsonl`
- `update_history(user_id, expense_id, category_result, vendor, amount)`
- `get_user_category_distribution(user_id) → {category: count}`
- `get_user_top_vendors(user_id, category?, limit=10) → [{vendor, count, total_spent}]`

**Evaluation** (`evaluate.py`):
- 5-fold stratified cross-validation + 80/20 held-out per-category report
- Prints classification report + confusion matrix
- Retrains on full data and saves model after evaluation

### 4.5 Deduplication — `deduplication/`

**Fingerprinter** (`fingerprinter.py`):
- `generate_exact_key(expense) → str`: SHA-256 of `"user_id|vendor_lower|date|amount"` — catches identical re-uploads
- `generate_fuzzy_embedding(expense) → np.ndarray`: SBERT embed of vendor + **alphabetically-sorted** line-item descriptions — catches near-duplicates regardless of item order

**Detector** (`detector.py`):
- `DuplicateDetector` class with per-user **Faiss IndexFlatL2** index
- Cached in memory, persisted to `data/{user_id}/faiss.index` + `index_map.json`
- `find_duplicates(expense, user_id, threshold=0.12) → list[DuplicateMatch]`
  - Step 1: Exact key lookup (score = 1.0)
  - Step 2: Faiss top-5 nearest neighbors, filtered by:
    - L2 distance < threshold
    - Date within ±3 days
    - Amount within ±5%
  - Similarity = `1 - dist/2` (for L2-normalized vectors)
- `add_to_index(user_id, expense)` — called after save to keep index current
- Module-level singleton `detector` for reuse

**Resolver** (`resolver.py`):
- `resolve_duplicates(expense, user_id) → DuplicateResolution`
- Decision rules:
  - Exact match → `action="reject"` (do not save)
  - Fuzzy match with similarity ≥ 0.85 → `action="flag"` (save with `flagged=True`)
  - Otherwise → `action="accept"`
- Returns `DuplicateResolution(is_duplicate, duplicate_of, action, reason)`

**Evaluation** (`evaluate.py`):
- 20     corpus + 20 queries (5 exact, 10 near variants, 5 non-dupes)
- Result: **Precision@1 = 100%, Precision@3 = 100%, Precision@5 = 100%**

### 4.6 Ingestion Pipeline — `pipeline/ingest.py`

The integrated flow:
1. Mindee V2 OCR on receipt image
2. Normalize raw JSON → `ExpenseRecord`
3. **Dedup check** via `resolve_duplicates()` — rejects exact dupes, flags fuzzy matches
4. Save to JSONL (unless rejected)
5. Update Faiss index with the new record
6. Print summary with status (✅ OK / 🔶 FLAGGED / 🚫 REJECTED)

```bash
# First upload — accepted
python pipeline/ingest.py --image receipt.jpg --user_id user_001

# Second upload of same receipt — rejected
python pipeline/ingest.py --image receipt.jpg --user_id user_001
```

---

## 5. Dependencies

```
mindee               # Mindee V2 API client
python-dotenv        # .env file loader
pydantic>=2.0        # Data validation & serialization
pytest               # Unit testing
sentence-transformers # SBERT embeddings (all-MiniLM-L6-v2)
lightgbm             # Gradient boosted classifier
scikit-learn         # LabelEncoder, metrics, train_test_split, CV
numpy                # Array operations
joblib               # Model persistence
faiss-cpu            # Approximate nearest neighbor search
```

Install: `pip install -r requirements.txt`

---

## 6. How to Run

```bash
# Activate venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux/Mac

# Ingest a receipt
python pipeline/ingest.py --image receipt.jpg --user_id user_001

# Run normalizer tests (36 tests)
python -m pytest tests/test_normalizer.py -v

# Evaluate categorizer (5-fold CV + confusion matrix)
python categorization/evaluate.py

# Evaluate duplicate detector (Precision@K)
python deduplication/evaluate.py
```

---

## 7. Data Flow & Storage Layout

All runtime data lives under `data/{user_id}/`:

| File | Written By | Format |
|---|---|---|
| `expenses.jsonl` | `storage/expense_store.py` | One JSON line per ExpenseRecord |
| `category_history.jsonl` | `categorization/user_history.py` | One JSON line per categorization event |
| `faiss.index` | `deduplication/detector.py` | Binary Faiss IndexFlatL2 |
| `index_map.json` | `deduplication/detector.py` | `{id_list: [...], key_set: {...}}` |

Trained ML model: `models/categorizer.joblib` (~1 MB)

---

## 8. Architecture Decisions

1. **JSONL over SQLite**: Chosen for simplicity, debuggability, and append-only write patterns. Each line is independently valid JSON — easy to inspect, stream-read, and recover from corruption.

2. **SBERT (all-MiniLM-L6-v2)**: Lightweight 80 MB model producing 384-dim vectors. Good balance of speed vs. quality for vendor/item text. Lazy-loaded with LRU caching to avoid redundant encoding.

3. **LightGBM over neural classifiers**: Small training set (500 examples) favors gradient boosting over deep learning. Trains in <1s, inference <1ms. L1/L2 regularization prevents overfitting.

4. **Faiss over brute-force**: Even with small user histories, Faiss IndexFlatL2 provides a clean API for L2 search with normalization. Scales naturally as corpus grows. Persisted to disk for cross-session use.

5. **SHA-256 exact keys**: Deterministic, collision-resistant. Catches identical re-uploads with zero false positives. Independent of SBERT — works even if the model changes.

6. **Fallback to Mindee categories**: When ML confidence < 0.6, the system trusts Mindee's OCR-derived category rather than risking a bad prediction. This is critical for downstream anomaly detection.

7. **Per-user scoping everywhere**: Every index, store, and history file is user-partitioned. No cross-user data leakage. Supports future per-user model adaptation.

---

## 9. Current Evaluation Metrics

### Categorizer (v2, 500 balanced seed examples)
- **5-fold CV accuracy**: 72.6% ± 4.0%
- **5-fold CV macro-F1**: ~0.66
- Best per-class: Food (0.83), Education (0.86), Utilities (0.83)
- Weakest: Transport (0.50), Office (0.50), Other (0.45)
- Improvement path: real user data, per-user fine-tuning, more seed variety

### Duplicate Detector (20 corpus + 20 queries)
- **Precision@1**: 100%
- **Precision@3**: 100%
- **Precision@5**: 100%
- Handles: case variation, vendor name differences, item reordering, date drift ±1 day
- Non-duplicates correctly not flagged

---

## 10. Known Limitations & Future Work

- **Categorizer accuracy ceiling**: ~80% with synthetic data alone. Needs real receipts or active learning loop.
- **No image-level dedup**: Current dedup is text-based (post-OCR). A perceptual hash (pHash) on the image itself would catch pre-OCR duplicates even faster.
- **No API server**: Currently CLI-only. A FastAPI endpoint wrapping `process_receipt()` is the natural next step.
- **No anomaly detection**: The category history infrastructure exists but no anomaly detection module yet.
- **No multi-currency normalization**: Amounts are stored as-is. Cross-currency comparisons need exchange rate handling.
- **Confidence field from Mindee V2**: The `total_amount.confidence` score isn't yet used as a weight in anomaly detection.
