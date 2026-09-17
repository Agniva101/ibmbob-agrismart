# 🌾 AgriSmart: AI-Powered Crop Recommendation System

> Data-driven crop recommendations using soil and environmental conditions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Machine Learning Models](#machine-learning-models)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [Running the Application](#running-the-application)
8. [REST API Reference](#rest-api-reference)
9. [Application Pages](#application-pages)
10. [Technologies Used](#technologies-used)
11. [Limitations & Disclaimer](#limitations--disclaimer)

---

## Project Overview

**AgriSmart** is a professional machine-learning web application that recommends the most suitable
crop to grow based on soil nutrient levels and environmental conditions.
It trains and compares five classification algorithms on the Kaggle Crop Recommendation
Dataset and exposes both a Streamlit dashboard and a Flask REST API — all from a
**single Python file** (`app.py`).

### Problem Statement

Selecting the wrong crop for given soil and climate conditions causes reduced crop yields,
financial losses for farmers, and unnecessary environmental impact. AgriSmart provides
data-driven, ML-powered recommendations to support informed planting decisions.

---

## Features

| Feature | Details |
|---------|---------|
| ML Workflow | End-to-end: load → clean → train → evaluate → predict |
| 5 ML Models | LR, Decision Tree, Random Forest, KNN, SVM |
| Model Selector | Sidebar dropdown to switch the active model at any time |
| Auto Best Model | Default selection is the highest mean 5-fold CV weighted F1 on the training set |
| Streamlit Dashboard | Interactive 6-page professional UI |
| Flask REST API | 4 endpoints — start with `python app.py --api` on port 5000 |
| EDA | 10+ interactive Plotly charts |
| Dataset Explorer | Filterable viewer with CSV download |
| Prediction UI | Live crop prediction with predicted probability bar chart |
| Feature Importance | Shows importances for the active model (falls back to RF for models without it) |
| Reproducibility | `random_state=42` throughout |
| Caching | `st.cache_data` / `st.cache_resource` — no redundant retraining |

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| File | `Crop_recommendation.csv` |
| Records | 2,200 rows |
| Input Features | 7 numeric columns |
| Target | `label` (22 crop classes) |

### Input Features

| Column | Unit | Description |
|--------|------|-------------|
| `N` | kg/ha | Nitrogen content in soil |
| `P` | kg/ha | Phosphorus content in soil |
| `K` | kg/ha | Potassium content in soil |
| `temperature` | °C | Average temperature |
| `humidity` | % | Relative humidity |
| `ph` | — | Soil pH (0–14) |
| `rainfall` | mm | Annual rainfall |

### Target

- `label` — crop name (e.g., rice, maize, chickpea, … 22 classes total)

---

## Machine Learning Models

| Model | Notes |
|-------|-------|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Decision Tree | `random_state=42` |
| Random Forest | Tuned via GridSearchCV (training set only), `random_state=42` |
| K-Nearest Neighbors | `k=5` |
| Support Vector Machine | RBF kernel, `probability=True`, `random_state=42` |

### Feature Scaling & Data Leakage Prevention

Models requiring feature scaling (Logistic Regression, K-Nearest Neighbors, Support Vector Machine) use `StandardScaler` inside an sklearn `Pipeline`. This prevents data leakage during cross-validation because the scaler is fitted independently inside each CV fold. Tree-based models (Decision Tree, Random Forest) are used without scaling.

The final model is persisted as a single artifact (`best_model.joblib`), which includes the preprocessing pipeline where applicable.

### Evaluation Metrics

- Accuracy (train & test)
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Confusion Matrix
- Full Classification Report

### Model Selection & Switching

The model with the **highest mean CV F1 score (5-fold stratified cross-validation on the
training set)** is automatically selected as the default. You can switch to any of the
five trained models at any time using the **"Select Model"** dropdown in the sidebar.
Every page — Overview KPI cards, Crop Recommendation predictions, Confusion Matrix,
Classification Report, and Feature Importance — updates instantly to reflect your
chosen model.

The Flask REST API always uses the CV-selected best model (fixed at startup).

---

## Project Structure

```
AgriSmart/
├── app.py                   # ← entire application (frontend + backend)
├── Crop_recommendation.csv  # ← dataset (download from Kaggle)
├── requirements.txt
├── best_model.joblib        # ← saved model (includes scaling pipeline if applicable)
├── label_encoder.joblib     # ← saved label encoder
├── README.md
└── AgriSmart_Project_Documentation.docx
```

Everything lives in `app.py`. No separate modules, no frontend folder.

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Download the dataset

Go to:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Download `Crop_recommendation.csv` and place it in the **same directory as `app.py`**.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```powershell
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Streamlit Dashboard (default)

```bash
streamlit run app.py
```

The browser will open automatically at **`http://localhost:8501`**.

> **Note:** The Flask REST API does **not** start automatically alongside Streamlit.
> Start it separately (see below) if you need the API endpoints.

### Flask API only (standalone mode)

```bash
python app.py --api
```

This loads the dataset, trains the models, and starts only the Flask server on
`http://localhost:5000`. Useful for headless or server deployments.

---

## REST API Reference

Base URL: `http://localhost:5000`

> The REST API always uses the **CV-selected best model** (by mean CV F1, weighted).
> The sidebar model selector only affects the Streamlit UI.

---

### `GET /api/health`

Returns service status.

**Response:**
```json
{
  "status": "ok",
  "service": "AgriSmart Crop Recommendation API"
}
```

---

### `GET /api/dataset`

Returns dataset summary.

**Response:**
```json
{
  "rows": 2200,
  "columns": 8,
  "crop_count": 22,
  "feature_names": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
  "target_name": "label",
  "crops": ["apple", "banana", "blackgram", "..."]
}
```

---

### `GET /api/model-info`

Returns deployed model information and metrics for all models.

**Response:**
```json
{
  "selected_model": "Random Forest",
  "selection_criterion": "5-fold stratified cross-validation F1 (weighted)",
  "available_models": [
    "Random Forest",
    "..."
  ],
  "cross_validation": {
    "folds": 5,
    "scoring": "f1_weighted",
    "scores_by_model": {
      "Random Forest": 0.9945
    },
    "strategy": "StratifiedKFold"
  },
  "test_evaluation": {
    "Random Forest": {
      "test_accuracy": 0.9932,
      "test_f1": 0.9932
    }
  },
  "rf_best_params": {
    "max_depth": null,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 100
  },
  "rf_best_cv_f1": 0.9945,
  "feature_names": [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
  ],
  "target_name": "label"
}
```

---

### `POST /api/predict`

Predict the recommended crop.

**Request Body:**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 23.5,
  "humidity": 80.0,
  "ph": 6.5,
  "rainfall": 200.0
}
```

**Response:**
```json
{
  "predicted_crop": "rice",
  "probabilities": {
    "apple": 0.0,
    "rice": 0.94,
    "maize": 0.02
  },
  "input_values": {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 23.5,
    "humidity": 80.0,
    "ph": 6.5,
    "rainfall": 200.0
  }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":23.5,"humidity":80,"ph":6.5,"rainfall":200}'
```

---

## Application Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI cards (Train/Test Accuracy reflect selected model), project description, crop list, dataset stats |
| **Crop Recommendation** | Prediction form — uses the active model selected in the sidebar |
| **Exploratory Data Analysis** | Distributions, correlation heatmap, N-P-K comparison, crop comparisons |
| **Model Performance** | Metrics table, comparison chart, confusion matrix & classification report for active model, feature importance |
| **Dataset Explorer** | Filterable dataframe with range sliders and CSV download |
| **About Project** | Full technical documentation, limitations, future work |

### Sidebar Controls

| Control | Description |
|---------|-------------|
| Navigation | Radio buttons to switch between pages |
| Dataset Info | Record count, crop count, feature count |
| **Select Model** | Dropdown to switch the active model — updates all pages instantly |
| Active model badge | ★ Best model (green) or info banner showing which is best |
| Test Accuracy / F1 | Live metrics for the currently selected model |
| App URL | `http://localhost:8501` |
| REST API | Endpoints listed for quick reference |

---

## Technologies Used

| Category | Library / Tool |
|----------|---------------|
| Web UI | Streamlit |
| REST API | Flask |
| ML | scikit-learn |
| Data | pandas, numpy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Persistence | joblib |
| Language | Python 3.9+ |

---

## Limitations & Disclaimer

- Model performance reflects the training dataset's geographic and temporal scope.
- Soil and climate conditions vary significantly by region and season.
- Predicted probabilities are statistical estimates, not agronomic guarantees.
- The system does not account for market prices, crop rotation, or irrigation availability.
- The REST API always uses the best model — the sidebar selector does not affect API responses.

> **Disclaimer:** This system is a machine-learning project.
> Crop recommendations should be validated with local agricultural expertise
> and field conditions before making planting decisions.