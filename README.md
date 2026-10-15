# ChainSentinel: Ethereum Wallet Fraud Risk Assessment

An academic machine learning case study and pipeline designed to assess Ethereum wallet fraud risk. The system bridges historical labeled blockchain training data (Kaggle dataset, 9,816 wallets) with live on-chain feature extraction using **Etherscan API V2**, evaluated by gradient-boosted tree decision models and demonstrated via an interactive local web application.

---

## Academic Disclaimer & Research Scope

> [!IMPORTANT]
> **Academic Case Study Only:** ChainSentinel is an experimental research project.
> 1. **Uncalibrated Output:** Output risk scores represent model decision rankings relative to an operational validation threshold ($T=0.51$), **not calibrated posterior probabilities of real-world fraud**.
> 2. **Unverified Feature Parity:** 15 of 24 canonical features remain unverifiable against historical training data because the Kaggle dataset discarded transaction receipts and error flags.
> 3. **Cold-Start Bias:** Legitimate newly created wallets ($< 24\text{ hours}$ or $\le 1$ tx) are at elevated risk of false-positive classification due to tree-based lifespan penalties.
> 4. **No Financial Warranty:** This system must not be used as sole evidence for transaction blocking, asset freezing, or financial decisions.

---

## Project Roadmap & Status

- **Phase 1 (Completed & Audited)**: Data quality inspection, 51-column audit, deduplication, resolution of 829 missing ERC-20 fraud rows, canonical 24-feature schema definition.
- **Phase 2 & 2.1 (Completed & Reconciled)**: Multi-model training and evaluation (Dummy, Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting) under stratified 70/15/15 protocol ($N=1,473$ held-out test split). Serialized pipeline: `models/best_model_pipeline.joblib`.
- **Phase 3 & 3.1 (Completed & Audited)**: Configurable Etherscan API V2 client, live feature calculation, input schema validation gate, and prediction service with cold-start warning and zero-activity rejection.
- **Phase 4 (Completed & Debugged)**: Prediction API, offline demonstration presets, error mappings, automatic `.env` loading, and responsive web interface (`src/app.py`).
- **Phase 5 (Completed)**: Final independent verification, secrets isolation check, live Mainnet verification, error scenario audit, and presentation readiness guide ([`reports/presentation_guide.md`](reports/presentation_guide.md)).


---

## Key Performance Metrics (Held-Out Test Set, N=1,473, Threshold = 0.51)

- **Best Model Pipeline:** `HistGradientBoostingClassifier(max_depth=6, min_samples_leaf=20, random_state=42)`
- **Confusion Matrix:**
  - True Negatives (TN): `1,131`
  - False Positives (FP): `15`
  - False Negatives (FN): `41`
  - True Positives (TP): `286`
- **Precision:** `95.02%` (0.9502)
- **Recall:** `87.46%` (0.8746)
- **Fraud F1 Score:** `0.9108`
- **Macro F1 Score:** `0.9433`
- **PR-AUC:** `0.9736`
- **ROC-AUC:** `0.9897`
- **Brier Score:** `0.0283` *(strictly proper scoring rule measuring mean squared probabilistic error, not ranking)*

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- `pip` package manager

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/kushal09-11/Ether-guard.git
cd Ether-guard

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Phase 4 Web Application

### 1. Launch Server Locally
Start the application using `uvicorn` or by running `src/app.py`:
```bash
# Option A: Directly using uvicorn
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

# Option B: Running the application script directly
python src/app.py
```

### 2. Access the Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

### 3. API Key Requirement for Live vs Offline Testing
- **Offline Demonstration (No API key needed):** The application includes 4 offline demonstration scenarios (Mature Normal, Flagged High-Risk, Cold-Start Warning, Zero-Activity Unscoreable). You can immediately test all UI states and prediction pipelines without an API key or internet access.
- **Live On-Chain Analysis (API key required):** To analyze arbitrary live Ethereum wallet addresses or run the "Vitalik.eth" live preset, obtain a free API key from [Etherscan](https://etherscan.io) and set it in your environment or `.env` file:
  ```bash
  # Windows PowerShell
  $env:ETHERSCAN_API_KEY="your_api_key_here"

  # Linux/macOS
  export ETHERSCAN_API_KEY="your_api_key_here"
  ```
  *(See `.env.example` for reference. Never commit `.env` to version control).*

---

## REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive single-page application frontend. |
| `GET` | `/api/health` | Health check, model status, and API key configuration check. |
| `GET` | `/api/model-info` | Complete model architecture, held-out test metrics, and parity status. |
| `GET` | `/api/presets` | Curated demonstration scenarios for offline evaluation. |
| `POST` | `/api/assess-wallet` | Live wallet risk assessment via Etherscan API V2. |
| `POST` | `/api/assess-preset/{id}` | Evaluates a curated demonstration preset offline. |
| `POST` | `/api/assess-features` | Evaluates a custom 24-feature dictionary directly. |

---

## Running the Automated Test Suite

Run all 86 tests offline with `pytest`:
```bash
pytest -v
```

The test suite covers:
- Etherscan client rate limiting, pagination, empty wallet parsing, and credential protection.
- Live 24-feature extraction, mathematical precision, and edge cases.
- Model pipeline inference, threshold enforcement, and joblib serialization.
- Input schema validation (rejection of missing, extra, NaN, and infinite features).
- Prediction service cold-start warning and zero-activity rejection.
- REST API error mappings (400, 429, 502, 503, 504) and static frontend serving.

---

## Remaining Scientific & Engineering Limitations

1. **Survival Bias in Historical Lifespans:** In historical training data, fraud wallets averaged short lifespans (median 7.5k mins vs 116.8k mins normal). Consequently, new legitimate wallets receive high risk scores unless mitigated by the cold-start warning.
2. **Kaggle Dataset Omissions:** Transaction receipts and error flags were discarded by the original Kaggle scrapers; 15 live feature parity rules remain approximate or unverifiable.
3. **Internal Transactions Omission:** The Kaggle dataset excluded internal transactions (`txlistinternal`), producing phantom negative balances in 978 wallets. The live pipeline replicates `total_ether_received - total_ether_sent` for consistency.
4. **Adversarial Drift:** Contemporary fraud vectors (sandwich attacks, flash loans, phishing permits) operate differently than 2018–2020 fraud patterns.