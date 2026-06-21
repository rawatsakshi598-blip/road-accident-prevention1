# 🚦 Delhi Road Accident Severity Predictor

A full-stack web application that predicts road accident severity across Delhi NCT using real historical accident data, a geospatial digital twin of Delhi's road network, and an ensemble of machine learning models.

---

## 📌 Project Overview

This project combines a **Python/FastAPI backend**, a **React frontend**, and a suite of **trained ML models** to assess and visualize accident risk on Delhi's road segments. It processes 10 real Delhi accident datasets spanning 2016–2024, maps them to OpenStreetMap road segments, and exposes predictions via a REST API.

---

## 🗂️ Repository Structure

```
road-accident-prevention1/
├── backend/                        # FastAPI Python backend
│   ├── main.py                     # FastAPI application entry point
│   ├── config.py                   # App configuration & constants
│   ├── requirements.txt            # Python dependencies
│   ├── ml/                         # Machine learning pipeline
│   │   ├── delhi_data_mapper.py    # Loads & maps all 10 Delhi datasets
│   │   ├── delhi_trainer.py        # Trains all 5 ML models
│   │   ├── digital_twin.py         # Digital twin orchestrator
│   │   ├── segment_risk_calculator.py
│   │   ├── road_network_loader.py  # OSM road network downloader
│   │   ├── predictor.py            # Inference pipeline
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── shap_analyzer.py
│   │   └── models/
│   │       ├── random_forest.py
│   │       ├── xgboost_model.py
│   │       ├── gradient_boosting.py
│   │       ├── svm_model.py
│   │       └── logistic_regression.py
│   ├── api/                        # API route definitions
│   ├── data/
│   │   ├── delhiDatasets/          # 10 real Delhi accident datasets
│   │   ├── road_networks/          # Cached OSM road network
│   │   └── mapped_accidents/       # Post-geocoding segment mappings
│   ├── outputs/                    # Saved models, charts, result JSONs
│   └── tests/                      # Unit, integration & system tests
├── frontend/                       # React frontend application
│   ├── src/
│   ├── package.json
│   └── ...
├── train_single_model.py           # Standalone script to train one model
├── localsetupfinal.md              # Detailed local setup guide
├── debug.txt                       # Debug log
└── .gitignore
```

---

## ✨ Features

- **10 Real Delhi Datasets** — Accident records from 2016 to 2024, totalling 8,000+ entries
- **Geospatial Digital Twin** — Full Delhi NCT road network (~30,000+ segments) via OpenStreetMap/OSMnx
- **5 ML Models** — XGBoost, Gradient Boosting, Random Forest, SVM, Logistic Regression, all trained with SMOTE balancing and hyperparameter tuning
- **SHAP Explainability** — Model-level feature importance via SHAP values
- **Risk Heat Map** — Interactive color-coded map with 5 risk tiers (Green → Red)
- **REST API** — FastAPI backend with auto-generated Swagger docs at `/docs`
- **React Frontend** — Interactive web UI running on port 3000
- **Full Test Suite** — Unit, integration, system, functional, non-functional, and performance tests

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| ML / Data | scikit-learn, XGBoost, imbalanced-learn (SMOTE), pandas, numpy |
| Geospatial | GeoPandas, OSMnx, Folium |
| Explainability | SHAP |
| Serialization | joblib, openpyxl |
| Frontend | React, Node.js 16+ |
| API Proxy (optional) | Caddy |

---

## ⚙️ Prerequisites

| Requirement | Minimum Version | Check Command |
|---|---|---|
| Python | 3.10+ | `python3 --version` |
| Node.js | 16+ | `node --version` |
| Git | 2.0+ | `git --version` |
| RAM | 4 GB free | — |

---

## 🚀 Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rawatsakshi598-blip/road-accident-prevention1.git
cd road-accident-prevention1
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

pip install -r requirements.txt
```

Verify:

```bash
python -c "import fastapi; import sklearn; import xgboost; import osmnx; print('All imports OK')"
```

> **Note:** If `osmnx` or `geopandas` fails, install system GEOS/GDAL first:
> `sudo apt install libgeos-dev libgdal-dev` (Ubuntu)

### 3. Frontend Setup

```bash
cd ../frontend
npm install
```

If `npm install` fails:

```bash
rm -rf node_modules package-lock.json
npm cache clean --force
npm install --legacy-peer-deps
```

---

## 🔄 First-Time Run Order (Important)

> Steps 1–4 are **first-time only**. Steps 5–6 are run every session.

### Step 1 — Download Delhi Road Network *(5–10 min)*

```bash
cd backend && source venv/bin/activate
python -c "from ml.road_network_loader import RoadNetworkLoader; loader = RoadNetworkLoader('delhi'); loader.get_or_download_network()"
```

> Delete any cached network first to ensure the full Delhi NCT download:
> `rm -rf data/road_networks/delhi/`

### Step 2 — Map Real Accident Data to Road Segments *(2–5 min)*

```bash
python -c "from ml.delhi_data_mapper import DelhiDataMapper; from ml.road_network_loader import RoadNetworkLoader; loader = RoadNetworkLoader('delhi'); edges = loader.get_edges_gdf(); mapper = DelhiDataMapper(edges, 'delhi'); mapper.geocode_and_map_all()"
```

This maps all 10 datasets to road segments:

| Dataset | Records |
|---|---|
| Dataset 1 | 77 rows with GPS coordinates |
| Dataset 2 | 2,433 rows with GPS coordinates |
| Dataset 3 | 30 CSV files (2021–2023 crash data) |
| Dataset 4 | 2019–2021 ML-ready data |
| Dataset 5 | 8 CSV files (2016 data) |
| Dataset 6 | Circle-wise data |
| Dataset 7 | 21 CSV files (2018 data) |
| Dataset 8 | 2022–2024 classification |
| Dataset 9 | 2020–2022 data |
| Dataset 10 | Comprehensive 2016–2020 data |

### Step 3 — Build Digital Twin *(1–3 min)*

```bash
python -c "from ml.digital_twin import DigitalTwin; twin = DigitalTwin('delhi'); twin.build()"
```

### Step 4 — Train ML Models *(10–30 min)*

```bash
python -c "from ml.delhi_trainer import DelhiTrainer; trainer = DelhiTrainer(); results = trainer.run()"
```

Models trained:

| Model | Type |
|---|---|
| XGBoost | Gradient Boosting |
| GradientBoosting | Scikit-learn Ensemble |
| RandomForest | Bagging Ensemble |
| SVM | Support Vector Machine |
| LogisticRegression | Linear Classifier |

All models use **SMOTE class balancing** and **hyperparameter tuning**.

### Step 5 — Start Backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API available at: **http://localhost:8000**  
Swagger docs at: **http://localhost:8000/docs**

### Step 6 — Start Frontend

```bash
cd frontend
npm start
```

Frontend available at: **http://localhost:3000**

### Step 7 — (Optional) Caddy API Proxy

If the frontend cannot reach the backend directly:

```bash
caddy run --config Caddyfile
```

This proxies `/api/*` → `localhost:8000` on port 81. Alternatively, add to `frontend/package.json`:

```json
{ "proxy": "http://localhost:8000" }
```

---

## 🧪 Training a Single Model

Use the standalone script to train one model at a time:

```bash
# From repo root
python train_single_model.py RandomForest
python train_single_model.py XGBoost
python train_single_model.py GradientBoosting
python train_single_model.py SVM
python train_single_model.py LogisticRegression
python train_single_model.py all           # Train all 5 models sequentially
python train_single_model.py XGBoost --no-shap   # Skip SHAP analysis
```

Output per model includes: Accuracy, F1 (Weighted & Macro), ROC-AUC, MCC, Cross-validation mean ± std, saved model artifact, and result JSON.

---

## 🧪 Running Tests

```bash
cd backend && source venv/bin/activate

python -m pytest tests/test_unit.py -v
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_system.py -v
python -m pytest tests/ -v                        # All tests
python -m pytest tests/ -v -m "not slow"          # Skip slow tests
python -m pytest tests/ -v -m functional
python -m pytest tests/ -v -m nonfunctional
python -m pytest tests/ -v -m performance
```

---

## 🗺️ Risk Map Color Scheme

| Risk Level | Color | Range |
|---|---|---|
| 🟢 Zero Accidents | Green `#22C55E` | 0–10% |
| 🔵 Low Risk | Blue `#3B82F6` | 10–40% |
| 🟡 Moderate Risk | Yellow `#EAB308` | 40–60% |
| 🟠 High Risk | Orange `#F97316` | 60–80% |
| 🔴 Very High Risk | Red `#EF4444` | 80–95%+ |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| OSMnx download timeout | Check internet connection; the system retries automatically |
| Low mapping counts | Delete `data/mapped_accidents/delhi/` and re-run Step 2 |
| ML training crashes | Ensure `segment_mapping.json` exists (run Step 2 first) |
| Frontend can't reach backend | Use Caddy (Step 7) or set `"proxy"` in `frontend/package.json` |
| `Model not found` error | Re-run Step 4 to regenerate model artifacts |
| Port 8000 in use | `lsof -i :8000` then `kill -9 <PID>`, or use `--port 8001` |
| `npm install` fails | `rm -rf node_modules && npm install --legacy-peer-deps` |
| High memory usage | The road network requires 2–4 GB RAM; ensure at least 4 GB free |

---

## 📊 Data Overview

- **10 real Delhi accident datasets** (2016–2024)
- **8,000+** total accident records
- Mapped to **500+ road segments** across Delhi NCT
- ~**30,000+** OSM road network segments loaded
- Virtual segments created for GPS points outside the road network

---

## 🌐 Languages

- **Python** — 79.4% (backend, ML pipeline)
- **JavaScript** — 20.1% (React frontend)
- **Other** — 0.5%

---

## 📄 License

This project is open source. See repository for details.

---

*Last updated: 2025*
