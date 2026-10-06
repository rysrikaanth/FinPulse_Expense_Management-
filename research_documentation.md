# Research Project: Personalized Expense Intelligence System (PEIS)

## 1. Executive Summary
The **Personalized Expense Intelligence System (PEIS)** is an end-to-end framework designed to automate expense management, detect fraudulent or anomalous claims, and provide personalized financial insights. By combining vector-based duplicate detection, behavioral anomaly ensembles, and LLM-grounded insights, the system significantly outperforms traditional rule-based expense software.

---

## 2. Research Problem & Objective
Traditional expense tracking systems suffer from:
1.  **High False Positives**: Static rules (e.g., "$>5000 is an anomaly") fail to account for a user’s role and history.
2.  **Duplicate Fraud**: Simple ID-based matching cannot catch fuzzy duplicates (e.g., submitting a bill twice with a different date).
3.  **Fragmented Data**: Receipts are scanned but not translated into actionable advice for the employee.

**Objective**: To build a system that learns a **user-specific behavioral profile** to identify anomalies and provide data-grounded financial summaries.

---

## 3. Core Technical Architecture
The system is built on a modular micro-service architecture:

### A. Data Ingestion & Normalization (`pipeline/`)
-   **OCR Engine**: Uses the **Mindee V2 API** (specifically the `ClientV2` with `enqueue_and_get_result` flow) to perform asynchronous extraction.
-   **Normalization**: Standardizes raw API outputs into an `ExpenseRecord` schema, converting various currencies and date formats to ISO-8601.

### B. Intelligent Categorization (`categorization/`)
-   **Approach**: Uses **Sentence-BERT (SBERT)** embeddings to map receipt text into a high-dimensional vector space.
-   **Classifier**: A **LightGBM** (Gradient Boosting) model trained on the standard **CORD** dataset, augmented with user-specific history to handle ambiguous vendors (e.g., "Amazon" as either `office` or `shopping`).

### C. Fuzzy Deduplication (`deduplication/`)
-   **Technology**: **Faiss L2 index** combined with SBERT embeddings.
-   **Logic**: Instead of exact matching, it calculates the cosine similarity between receipts. It can detect "fuzzy" duplicates where the date differs by ±2 days or the vendor name is slightly altered.

### D. Behavioral Anomaly Detection (`anomaly/`) — *The Core Research*
-   **9-Dimensional Feature Engineering**:
    -   **Amount**: Z-score and Percentile relative to the *user's specific history* for that category.
    -   **Vendor**: Novelty scores and frequency ratios.
    -   **Temporal (Receipt Time)**: Evaluates if a purchase occurred at an "implausible" time (e.g., `hour_of_day_score` and `day_of_week_score`).
-   **The Ensemble Model**: A weighted combination (0.6 IF + 0.4 AE) of:
    1.  **Isolation Forest**: Detects global multi-variate outliers.
    2.  **Autoencoder (PCA/NN)**: Measures reconstruction error to identify subtle deviations from "normal" behavior.
-   **Result**: The system explains *why* it flagged a claim (e.g., *"Amount ₹15,000 is 4.2σ above your usual spend"*).

### E. Time-Series Spend Forecasting (`forecasting/`)
-   **Model**: **Prophet** (Meta-research) with weekly and monthly seasonality.
-   **Early Warnings**: Maps forecasted spending against the user's defined limits to generate proactive notifications *before* a budget breach occurs.

### F. LLM Insight Layer (`insights/`)
-   **Model**: Gemini 2.0 Flash.
-   **RAG (Retrieval-Augmented Generation)**: Grounding the LLM in real KPI context (Spend ratios, Anomaly logs, Budgets) using a specialized **Hallucination Checker** to ensure numerical accuracy.

---

## 4. Evaluation Methodology & Results
The system was evaluated using both real-world benchmarks (SROIE/CORD) and synthetic anomaly injection.

| Module | Primary Metric | System Result | Improvement |
| :--- | :--- | :--- | :--- |
| **OCR Extraction** | Field Accuracy | **94.14%** | Target met (>90%) |
| **Categorization** | Macro F1 Score | **0.9073** | **+22%** vs. Mindee baseline |
| **Deduplication** | Precision@5 | **1.0000** | High-precision vector matching |
| **Anomaly Detection** | **AUC-ROC** | **0.9178** | **+45%** vs. Static Threshold |
| **System Latency** | End-to-End | **3.8s** | Under 5s requirement |

### Key Result: Anomaly Detection Performance
The use of **Receipt Time** (Day/Hour scores) proved to be one of the top 3 discriminative features in detectting synthetic fraud, with a **Cohen's d effect size of >0.8**.

---

## 5. Deployment & Usage
-   **API**: FastAPI-based backend with endpoints for `/receipts/upload`, `/users/dashboard`, and `/anomalies/list`.
-   **Dashboard**: A React-based UI that provides a "Review Queue" for admins and a "Personal KPI" view for users.
-   **Dockerized**: Full containerization via `docker-compose` for API and Dashboard services.

---

## 6. Conclusion & Potential
The PEIS proves that **personalized behavior modeling** is significantly more effective at fraud detection than generic enterprise rules. The system successfully bridges the gap between raw receipt data and financial intelligence.

---
