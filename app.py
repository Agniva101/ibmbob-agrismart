"""
AgriSmart: AI-Powered Crop Recommendation System
=================================================
A professional machine-learning web application that predicts the most suitable
crop based on soil and environmental conditions.

Run: streamlit run app.py
API: python app.py --api   (starts Flask API on port 5000)
"""

# ── 1. Imports ──────────────────────────────────────────────────────────────
import sys
import os
import io
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import joblib
import json

warnings.filterwarnings("ignore")

# ── 2. Configuration ─────────────────────────────────────────────────────────
DATASET_PATH = "Crop_recommendation.csv"
REQUIRED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler.joblib"
LE_PATH = "label_encoder.joblib"
FLASK_PORT = 5000

# Shared StratifiedKFold used throughout for CV and GridSearch
SKF = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Random Forest hyperparameter grid
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}

FEATURE_DESCRIPTIONS = {
    "N": "Nitrogen content in soil (kg/ha)",
    "P": "Phosphorus content in soil (kg/ha)",
    "K": "Potassium content in soil (kg/ha)",
    "temperature": "Average temperature (°C)",
    "humidity": "Relative humidity (%)",
    "ph": "Soil pH value",
    "rainfall": "Annual rainfall (mm)",
}

# ── 3. Dataset Loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the crop recommendation dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: '{path}'. "
            "Please download Crop_recommendation.csv from "
            "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset "
            "and place it in the same directory as app.py."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df

# ── 4. Data Preprocessing ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame):
    """
    Clean data, encode labels, and split into raw train/test sets.
    Feature scaling is intentionally NOT performed here — it is handled
    inside each model's sklearn Pipeline so that no data leakage occurs
    during cross-validation or hyperparameter tuning.
    """
    df_clean = df.copy()

    # Drop duplicates
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Drop rows with any missing values in required columns
    df_clean = df_clean.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    X = df_clean[FEATURE_COLS].values
    y = df_clean[TARGET_COL].values

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Stratified train/test split — raw (unscaled) arrays
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    return df_clean, X_train, X_test, y_train, y_test, le

# ── 5. Model Training & CV-Based Selection ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(_X_train, _y_train):
    """
    Step 1 — Cross-validate all base models on the training set only.
    Step 2 — Run GridSearchCV on Random Forest (training set only).
    Step 3 — Select the best model by mean CV F1 (weighted).
    Step 4 — Refit the selected model on the full training set.

    Returns a tuple:
        models          dict of all fitted estimators/pipelines
        best_name       name of the CV-selected model
        best_model      the selected estimator or pipeline
        cv_results_df   DataFrame of CV F1 scores for all models
        rf_best_params  best params from GridSearchCV (dict)
        rf_best_cv_f1   best CV F1 from GridSearchCV (float)

    DATA-LEAKAGE NOTE:
        Models that require feature scaling (LR, KNN, SVM) are wrapped in an
        sklearn Pipeline([("scaler", StandardScaler()), ("model", ...)]).
        This guarantees that StandardScaler is fitted independently inside
        each CV fold and never sees the validation fold's data.
        Tree-based models (Decision Tree, Random Forest) do not require
        scaling and are used as bare estimators.
    """
    # Models that need scaling are wrapped in a Pipeline.
    # Tree-based models are used directly (no scaling needed).
    base_estimators = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }

    # --- CV comparison on training data only (raw, unscaled X_train) ---
    # Scaling happens inside each fold via the Pipeline.
    cv_scores = {}
    for name, estimator in base_estimators.items():
        scores = cross_val_score(
            estimator, _X_train, _y_train,
            cv=SKF, scoring="f1_weighted", n_jobs=-1
        )
        cv_scores[name] = round(float(scores.mean()), 4)

    # --- GridSearchCV for Random Forest (training data only, no scaling needed) ---
    # RF param grid keys are bare estimator params (no pipeline prefix needed).
    rf_gs = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=RF_PARAM_GRID,
        scoring="f1_weighted",
        cv=SKF,
        n_jobs=-1,
        refit=True,
    )
    rf_gs.fit(_X_train, _y_train)
    rf_best_params = rf_gs.best_params_
    rf_best_cv_f1 = round(float(rf_gs.best_score_), 4)

    # Replace the baseline RF CV score with the tuned RF CV score
    cv_scores["Random Forest"] = rf_best_cv_f1

    # --- Build CV results DataFrame sorted by CV F1 ---
    cv_results_df = pd.DataFrame([
        {"Model": name, "CV F1 (mean)": score}
        for name, score in cv_scores.items()
    ]).sort_values("CV F1 (mean)", ascending=False).reset_index(drop=True)

    # --- Select best model by CV F1 ---
    best_name = cv_results_df.iloc[0]["Model"]

    # --- Fit all estimators on the full training set ---
    fitted_models = {}
    for name, estimator in base_estimators.items():
        if name == "Random Forest":
            # Use the tuned estimator (already refit by GridSearchCV on full X_train)
            fitted_models[name] = rf_gs.best_estimator_
        else:
            estimator.fit(_X_train, _y_train)
            fitted_models[name] = estimator

    best_model = fitted_models[best_name]

    # --- Persist the best model (pipeline or estimator) ---
    try:
        joblib.dump(best_model, MODEL_PATH)
    except Exception:
        pass  # non-fatal

    return fitted_models, best_name, best_model, cv_results_df, rf_best_params, rf_best_cv_f1


# ── 6. Final Test Evaluation ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def evaluate_models(_models: dict, _cv_results_df: pd.DataFrame,
                    _X_train, _X_test, _y_train, _y_test) -> pd.DataFrame:
    """
    Evaluate all models on the held-out test set.
    The test set is used here only for final reporting, NOT for model selection.
    Returns a DataFrame merging CV F1 with test-set metrics, sorted by CV F1.
    """
    results = []
    for name, model in _models.items():
        y_pred = model.predict(_X_test)
        results.append({
            "Model": name,
            "Train Accuracy": round(accuracy_score(_y_train, model.predict(_X_train)), 4),
            "Test Accuracy":  round(accuracy_score(_y_test, y_pred), 4),
            "Test Precision": round(precision_score(_y_test, y_pred, average="weighted", zero_division=0), 4),
            "Test Recall":    round(recall_score(_y_test, y_pred, average="weighted", zero_division=0), 4),
            "Test F1":        round(f1_score(_y_test, y_pred, average="weighted", zero_division=0), 4),
        })
    test_df = pd.DataFrame(results)
    # Merge CV scores so the table is self-contained
    merged = pd.merge(test_df, _cv_results_df, on="Model")
    # Sort by CV F1 (the selection criterion)
    merged = merged.sort_values("CV F1 (mean)", ascending=False).reset_index(drop=True)
    # Reorder columns for readability
    cols = ["Model", "CV F1 (mean)", "Train Accuracy",
            "Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
    return merged[cols]


# ── 7. Best Model Lookup ─────────────────────────────────────────────────────
def get_best_model(models: dict, best_name: str):
    """Return the CV-selected model by name."""
    return best_name, models[best_name]

# ── 8. Prediction Helper ──────────────────────────────────────────────────────
def make_prediction(model, le: LabelEncoder, inputs: dict):
    """
    Run inference and return the predicted crop label plus class probabilities.
    The model may be a bare estimator (Decision Tree, Random Forest) or an
    sklearn Pipeline that includes its own StandardScaler step.
    In both cases raw (unscaled) feature values are passed in — the pipeline
    handles scaling internally where needed.
    inputs must be a dict with keys matching FEATURE_COLS.
    """
    x = np.array([[inputs[f] for f in FEATURE_COLS]], dtype=float)
    pred_idx = model.predict(x)[0]
    predicted_crop = le.inverse_transform([pred_idx])[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        probabilities = {
            le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
    return predicted_crop, probabilities

# ── 9. Streamlit UI helpers ──────────────────────────────────────────────────
def kpi_card(col, label: str, value, delta=None, icon: str = ""):
    """Render a styled KPI metric card inside a Streamlit column."""
    col.metric(label=f"{icon} {label}".strip(), value=value, delta=delta)


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def styled_prediction_card(crop: str, predicted_probability: float):
    """Display the prediction result in a styled card."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,#1a472a,#2d6a4f);
            border-radius:12px;
            padding:28px 32px;
            text-align:center;
            color:#fff;
            margin:16px 0;
        ">
            <div style="font-size:14px;letter-spacing:1px;opacity:.8;margin-bottom:6px;">
                RECOMMENDED CROP
            </div>
            <div style="font-size:42px;font-weight:700;letter-spacing:1px;">
                🌱 {crop.upper()}
            </div>
            <div style="margin-top:12px;font-size:16px;opacity:.85;">
                Predicted Probability: <strong>{predicted_probability:.1%}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── 10. Page: Overview ────────────────────────────────────────────────────────
def render_overview(df: pd.DataFrame, metrics_df: pd.DataFrame, selected_name: str):
    section_header("Overview", "AgriSmart — AI-Powered Crop Recommendation System")

    sel_row = metrics_df[metrics_df["Model"] == selected_name].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Records", f"{len(df):,}", icon="📋")
    kpi_card(c2, "Crop Types", df[TARGET_COL].nunique(), icon="🌾")
    kpi_card(c3, "Features", len(FEATURE_COLS), icon="📊")
    kpi_card(c4, "CV F1 Score", f"{sel_row['CV F1 (mean)']:.2%}", icon="🎯")
    kpi_card(c5, "Test Accuracy", f"{sel_row['Test Accuracy']:.2%}", icon="✅")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("About AgriSmart")
        st.markdown(
            """
            **AgriSmart** is a machine-learning system that recommends the most suitable
            crop based on soil nutrient levels and environmental conditions.

            **Problem Statement:** Farmers face yield losses due to planting crops in
            unsuitable soil and climate conditions. This system uses historical
            agricultural data to provide data-driven crop recommendations.

            **How it works:**
            1. Enter soil and environmental parameters in the *Crop Recommendation* page.
            2. The trained ML model analyses the combination of inputs.
            3. The system returns the most suitable crop with a predicted probability.
            """
        )

        with st.expander("ML Workflow"):
            st.markdown(
                """
                1. Load & validate CSV dataset
                2. Clean data — remove duplicates & missing values
                3. Encode target labels (LabelEncoder)
                4. Stratified train / test split (80 / 20, raw unscaled arrays)
                5. 5-fold stratified cross-validation on training set (scaling inside each fold via Pipeline)
                6. GridSearchCV hyperparameter tuning for Random Forest (training set only, no scaling needed)
                7. Select best model by mean CV F1 (weighted) — test set untouched
                8. Refit all models on full training set
                9. Evaluate all models once on held-out test set
                10. Serve live predictions via Streamlit UI & REST API (--api mode)
                """
            )

    with col_right:
        st.subheader("Supported Crops")
        crops = sorted(df[TARGET_COL].unique())
        crop_cols = st.columns(2)
        for i, crop in enumerate(crops):
            crop_cols[i % 2].markdown(f"• {crop.title()}")

    st.markdown("---")
    st.subheader("Dataset Quick Statistics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(
            df[FEATURE_COLS].describe().round(2).T.rename(
                columns={"50%": "median"}
            ),
            width='stretch',
        )
    with col_b:
        crop_counts = df[TARGET_COL].value_counts().reset_index()
        crop_counts.columns = ["Crop", "Count"]
        fig = px.bar(
            crop_counts, x="Crop", y="Count",
            title="Records per Crop",
            color="Count", color_continuous_scale="Greens",
            height=320,
        )
        fig.update_layout(showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig, width='stretch')

# ── 11. Page: Crop Recommendation ────────────────────────────────────────────
def render_recommendation(model, le: LabelEncoder, df: pd.DataFrame):
    section_header("Crop Recommendation", "Enter soil and environmental parameters to get a recommendation.")

    defaults = df[FEATURE_COLS].mean().to_dict()
    mins = df[FEATURE_COLS].min().to_dict()
    maxs = df[FEATURE_COLS].max().to_dict()

    with st.form("prediction_form"):
        st.markdown("#### Soil Nutrient Levels")
        c1, c2, c3 = st.columns(3)
        N_val = c1.number_input(
            "Nitrogen (N) kg/ha", min_value=float(0), max_value=float(maxs["N"] * 1.5),
            value=float(round(defaults["N"], 1)), step=1.0
        )
        P_val = c2.number_input(
            "Phosphorus (P) kg/ha", min_value=float(0), max_value=float(maxs["P"] * 1.5),
            value=float(round(defaults["P"], 1)), step=1.0
        )
        K_val = c3.number_input(
            "Potassium (K) kg/ha", min_value=float(0), max_value=float(maxs["K"] * 1.5),
            value=float(round(defaults["K"], 1)), step=1.0
        )

        st.markdown("#### Environmental Conditions")
        c4, c5, c6, c7 = st.columns(4)
        temp_val = c4.number_input(
            "Temperature (°C)", min_value=-10.0, max_value=60.0,
            value=float(round(defaults["temperature"], 1)), step=0.5
        )
        hum_val = c5.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0,
            value=float(round(defaults["humidity"], 1)), step=0.5
        )
        ph_val = c6.number_input(
            "Soil pH", min_value=0.0, max_value=14.0,
            value=float(round(defaults["ph"], 2)), step=0.1
        )
        rain_val = c7.number_input(
            "Rainfall (mm)", min_value=0.0, max_value=float(maxs["rainfall"] * 1.5),
            value=float(round(defaults["rainfall"], 1)), step=1.0
        )

        submitted = st.form_submit_button("🌱 Recommend Crop", width='stretch', type="primary")

    if submitted:
        # Validate
        errors = []
        if not (0 <= ph_val <= 14):
            errors.append("pH must be between 0 and 14.")
        if not (0 <= hum_val <= 100):
            errors.append("Humidity must be between 0 and 100%.")
        if temp_val < -10 or temp_val > 60:
            errors.append("Temperature seems out of realistic range (-10 to 60 °C).")

        if errors:
            for e in errors:
                st.error(e)
        else:
            inputs = {
                "N": N_val, "P": P_val, "K": K_val,
                "temperature": temp_val, "humidity": hum_val,
                "ph": ph_val, "rainfall": rain_val,
            }
            try:
                predicted_crop, probabilities = make_prediction(model, le, inputs)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

            top_prob = probabilities[predicted_crop] if probabilities else None
            styled_prediction_card(predicted_crop, top_prob if top_prob else 0.0)

            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown("#### Input Summary")
                summary_df = pd.DataFrame(
                    {"Parameter": list(FEATURE_DESCRIPTIONS.keys()),
                     "Value": [inputs[f] for f in FEATURE_COLS],
                     "Description": list(FEATURE_DESCRIPTIONS.values())}
                )
                st.dataframe(summary_df, width='stretch', hide_index=True)

            with col_r:
                if probabilities:
                    st.markdown("#### Top 5 Crop Probabilities")
                    top5 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
                    top5_df = pd.DataFrame(top5, columns=["Crop", "Probability"])
                    fig = px.bar(
                        top5_df, x="Probability", y="Crop", orientation="h",
                        color="Probability", color_continuous_scale="Greens",
                        height=280,
                    )
                    fig.update_layout(showlegend=False, margin=dict(t=10, b=0))
                    st.plotly_chart(fig, width='stretch')

            with st.expander("Why this recommendation?"):
                st.markdown(
                    f"""
                    The model identified **{predicted_crop.title()}** as the highest-probability
                    class for the submitted input values, based on patterns learned from
                    **{len(df):,}** labelled crop records during training.

                    **Input features used by the model:**

                    | Feature | Submitted Value | Dataset Average | Relative Position |
                    |---------|----------------|-----------------|-------------------|
                    | Nitrogen (N) | {N_val} kg/ha | {defaults["N"]:.1f} kg/ha | {"Above" if N_val > defaults["N"] else "Below"} average |
                    | Phosphorus (P) | {P_val} kg/ha | {defaults["P"]:.1f} kg/ha | {"Above" if P_val > defaults["P"] else "Below"} average |
                    | Potassium (K) | {K_val} kg/ha | {defaults["K"]:.1f} kg/ha | {"Above" if K_val > defaults["K"] else "Below"} average |
                    | Temperature | {temp_val} °C | {defaults["temperature"]:.1f} °C | {"Above" if temp_val > defaults["temperature"] else "Below"} average |
                    | Humidity | {hum_val}% | {defaults["humidity"]:.1f}% | {"Above" if hum_val > defaults["humidity"] else "Below"} average |
                    | Soil pH | {ph_val} | {defaults["ph"]:.2f} | {"Acidic" if ph_val < 6 else "Neutral" if ph_val <= 7.5 else "Alkaline"} |
                    | Rainfall | {rain_val} mm | {defaults["rainfall"]:.1f} mm | {"Above" if rain_val > defaults["rainfall"] else "Below"} average |

                    > ⚠️ **Important:** This output reflects statistical patterns in the
                    training data. The model does not have causal agricultural knowledge.
                    It does not account for regional conditions, crop rotation, market
                    prices, or factors outside the training dataset.
                    Always validate recommendations with qualified local agricultural
                    expertise before making planting decisions.
                    """
                )

# ── 12. Page: EDA ─────────────────────────────────────────────────────────────
def render_eda(df: pd.DataFrame):
    section_header("Exploratory Data Analysis", "Interactive visualisations of the crop recommendation dataset.")

    crops = sorted(df[TARGET_COL].unique())
    selected_crop = st.selectbox("Filter charts by crop (or select All):", ["All"] + crops)
    plot_df = df if selected_crop == "All" else df[df[TARGET_COL] == selected_crop]

    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Crop Comparisons"])

    with tab1:
        st.subheader("Feature Distributions")
        if selected_crop == "All":
            fig = make_subplots(rows=2, cols=4, subplot_titles=FEATURE_COLS + [""])
            positions = [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3)]
            for feat, (r, c) in zip(FEATURE_COLS, positions):
                fig.add_trace(
                    go.Histogram(x=df[feat], name=feat, marker_color="#2d6a4f", showlegend=False),
                    row=r, col=c
                )
            fig.update_layout(height=500, title_text="Feature Distributions (All Crops)")
            st.plotly_chart(fig, width='stretch')
        else:
            feat_choice = st.selectbox("Select feature:", FEATURE_COLS)
            fig = px.histogram(
                plot_df, x=feat_choice, nbins=30,
                title=f"{feat_choice} Distribution — {selected_crop.title()}",
                color_discrete_sequence=["#2d6a4f"]
            )
            st.plotly_chart(fig, width='stretch')

        # Crop distribution
        crop_counts = df[TARGET_COL].value_counts().reset_index()
        crop_counts.columns = ["Crop", "Count"]
        fig2 = px.bar(
            crop_counts, x="Crop", y="Count",
            title="Crop Distribution",
            color="Count", color_continuous_scale="Greens", height=350
        )
        st.plotly_chart(fig2, width='stretch')

    with tab2:
        st.subheader("Feature Correlations")
        corr = df[FEATURE_COLS].corr()
        fig_heat = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap", height=480
        )
        st.plotly_chart(fig_heat, width='stretch')

        st.subheader("N-P-K Comparison by Crop")
        npk_avg = df.groupby(TARGET_COL)[["N", "P", "K"]].mean().reset_index()
        fig_npk = px.bar(
            npk_avg.melt(id_vars=TARGET_COL, value_vars=["N", "P", "K"]),
            x=TARGET_COL, y="value", color="variable", barmode="group",
            title="Average N, P, K by Crop", height=400,
            color_discrete_sequence=["#1b4332", "#40916c", "#95d5b2"],
        )
        fig_npk.update_xaxes(tickangle=45)
        st.plotly_chart(fig_npk, width='stretch')

    with tab3:
        st.subheader("Crop vs Environmental Conditions")
        metric = st.selectbox("Compare metric:", ["rainfall", "temperature", "ph", "humidity"])
        avg_df = df.groupby(TARGET_COL)[metric].mean().reset_index().sort_values(metric, ascending=False)
        fig_comp = px.bar(
            avg_df, x=TARGET_COL, y=metric,
            title=f"Average {metric.title()} by Crop",
            color=metric, color_continuous_scale="Greens", height=380
        )
        fig_comp.update_xaxes(tickangle=45)
        st.plotly_chart(fig_comp, width='stretch')

        # Box plots
        fig_box = px.box(
            df, x=TARGET_COL, y=metric,
            title=f"{metric.title()} Distribution per Crop",
            color=TARGET_COL, height=420
        )
        fig_box.update_layout(showlegend=False)
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, width='stretch')

# ── 13. Page: Model Performance ───────────────────────────────────────────────
def render_model_performance(
    models: dict, metrics_df: pd.DataFrame,
    selected_name: str, best_name: str,
    X_train, X_test, y_train, y_test, le: LabelEncoder,
    rf_best_params: dict, rf_best_cv_f1: float,
):
    section_header("Model Performance", "Training results and evaluation metrics for all models.")

    # ── Active model banner ───────────────────────────────────────────────────
    if selected_name == best_name:
        st.info(
            f"**Active model:** {selected_name}  ★  "
            f"Selected by 5-fold stratified cross-validation (highest CV F1)"
        )
    else:
        st.info(
            f"**Active model:** {selected_name}  "
            f"(CV-selected best model: **{best_name}**)"
        )

    # ── Model comparison table ────────────────────────────────────────────────
    st.subheader("Model Comparison")
    st.caption(
        "Models are ranked by **CV F1 (mean)** — cross-validation on the training set. "
        "Test metrics are shown for reference only and were not used for model selection."
    )
    display_df = metrics_df.copy()
    display_df.index = display_df.index + 1
    fmt = {
        "CV F1 (mean)":   "{:.2%}",
        "Train Accuracy": "{:.2%}",
        "Test Accuracy":  "{:.2%}",
        "Test Precision": "{:.2%}",
        "Test Recall":    "{:.2%}",
        "Test F1":        "{:.2%}",
    }
    st.dataframe(
        display_df.style.background_gradient(
            subset=["CV F1 (mean)", "Test Accuracy", "Test F1"], cmap="Greens"
        ).format(fmt),
        width='stretch',
    )

    # ── Comparison bar chart ──────────────────────────────────────────────────
    st.subheader("Metric Comparison Chart")
    chart_cols = ["CV F1 (mean)", "Test Accuracy", "Test Precision", "Test F1"]
    melt_df = metrics_df.melt(
        id_vars="Model", value_vars=chart_cols,
        var_name="Metric", value_name="Score"
    )
    fig_cmp = px.bar(
        melt_df, x="Model", y="Score", color="Metric", barmode="group",
        title="Model Metrics Comparison (CV F1 vs Test Metrics)",
        color_discrete_sequence=["#1b4332", "#40916c", "#74c69d", "#d8f3dc"],
        height=400,
    )
    fig_cmp.update_xaxes(tickangle=15)
    st.plotly_chart(fig_cmp, width='stretch')

    st.markdown("---")

    # ── Best hyperparameters (Random Forest) ─────────────────────────────────
    st.subheader("Best Model Parameters — Random Forest (GridSearchCV)")
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.markdown(f"**Best CV F1 Score (GridSearchCV):** `{rf_best_cv_f1:.2%}`")
        param_df = pd.DataFrame(
            {"Parameter": list(rf_best_params.keys()),
             "Value": [str(v) for v in rf_best_params.values()]}
        )
        st.dataframe(param_df, hide_index=True, width='stretch')
    with col_p2:
        st.caption(
            "These are the hyperparameters returned by GridSearchCV after searching "
            "across the defined parameter grid using 5-fold stratified cross-validation "
            "on the training set. The test set was not used during tuning."
        )

    st.markdown("---")

    # ── Overfitting check ─────────────────────────────────────────────────────
    st.subheader("Model Validation — Overfitting Check")
    sel_row = metrics_df[metrics_df["Model"] == selected_name].iloc[0]
    train_acc  = sel_row["Train Accuracy"]
    cv_f1      = sel_row["CV F1 (mean)"]
    test_acc   = sel_row["Test Accuracy"]
    test_f1    = sel_row["Test F1"]

    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Train Accuracy",  f"{train_acc:.2%}")
    oc2.metric("CV F1 (mean)",    f"{cv_f1:.2%}")
    oc3.metric("Test Accuracy",   f"{test_acc:.2%}")
    oc4.metric("Test F1",         f"{test_f1:.2%}")

    gap = train_acc - test_acc
    if gap > 0.10:
        interp = (
            f"The training accuracy ({train_acc:.2%}) is substantially higher than "
            f"the test accuracy ({test_acc:.2%}), a gap of {gap:.2%}. "
            "This may indicate possible overfitting — the model may have learned "
            "patterns specific to the training set that do not generalise fully."
        )
        st.warning(interp)
    elif gap > 0.03:
        interp = (
            f"There is a modest gap between training accuracy ({train_acc:.2%}) and "
            f"test accuracy ({test_acc:.2%}) ({gap:.2%}). "
            "This is within a typical range for most classifiers and does not strongly "
            "indicate overfitting."
        )
        st.info(interp)
    else:
        interp = (
            f"Training accuracy ({train_acc:.2%}) and test accuracy ({test_acc:.2%}) "
            f"are close (gap: {gap:.2%}), suggesting limited evidence of severe overfitting "
            "for this model on this dataset."
        )
        st.success(interp)

    st.markdown("---")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.subheader(f"Confusion Matrix — {selected_name}")
    y_pred = models[selected_name].predict(X_test)
    labels = le.classes_
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm, x=labels, y=labels, text_auto=True,
        color_continuous_scale="Greens",
        title=f"Confusion Matrix — {selected_name}",
        labels=dict(x="Predicted", y="Actual"),
        height=550,
    )
    fig_cm.update_xaxes(tickangle=45)
    st.plotly_chart(fig_cm, width='stretch')

    with st.expander(f"Classification Report — {selected_name}"):
        report = classification_report(y_test, y_pred, target_names=labels)
        st.text(report)

    # ── Feature Importance ────────────────────────────────────────────────────
    # Pipelines wrap the estimator under the "model" step; bare estimators
    # expose feature_importances_ directly.
    def _get_estimator(m):
        """Return the inner estimator from a Pipeline, or the model itself."""
        if isinstance(m, Pipeline):
            return m.named_steps.get("model", m)
        return m

    active_model = models[selected_name]
    active_estimator = _get_estimator(active_model)
    if hasattr(active_estimator, "feature_importances_"):
        st.subheader(f"Feature Importance — {selected_name}")
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": active_estimator.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            importance_df, x="Importance", y="Feature", orientation="h",
            title=f"{selected_name} — Feature Importance",
            color="Importance", color_continuous_scale="Greens", height=320
        )
        st.plotly_chart(fig_fi, width='stretch')
    elif "Random Forest" in models:
        st.subheader("Feature Importance — Random Forest")
        st.caption(
            f"{selected_name} does not expose feature importances. "
            "Showing Random Forest importances as a reference."
        )
        rf_estimator = _get_estimator(models["Random Forest"])
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": rf_estimator.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            importance_df, x="Importance", y="Feature", orientation="h",
            title="Random Forest — Feature Importance (reference)",
            color="Importance", color_continuous_scale="Greens", height=320
        )
        st.plotly_chart(fig_fi, width='stretch')

# ── 14. Page: Dataset Explorer ───────────────────────────────────────────────
def render_dataset_explorer(df: pd.DataFrame):
    section_header("Dataset Explorer", "Browse, filter, and download the crop recommendation dataset.")

    crops = ["All"] + sorted(df[TARGET_COL].unique())
    selected = st.selectbox("Filter by crop:", crops)
    filtered = df if selected == "All" else df[df[TARGET_COL] == selected]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows displayed", f"{len(filtered):,}")
    col_b.metric("Total rows", f"{len(df):,}")
    col_c.metric("Columns", len(df.columns))
    col_d.metric("Crop types", df[TARGET_COL].nunique())

    with st.expander("Numeric Range Filters"):
        range_cols = st.columns(len(FEATURE_COLS))
        range_filters = {}
        for i, feat in enumerate(FEATURE_COLS):
            col_min = float(df[feat].min())
            col_max = float(df[feat].max())
            range_filters[feat] = range_cols[i].slider(
                feat, col_min, col_max, (col_min, col_max), key=f"slider_{feat}"
            )

        for feat, (lo, hi) in range_filters.items():
            filtered = filtered[(filtered[feat] >= lo) & (filtered[feat] <= hi)]
        st.caption(f"Rows after range filters: {len(filtered):,}")

    st.dataframe(filtered.reset_index(drop=True), width='stretch', height=420)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download filtered dataset as CSV",
        data=csv_bytes,
        file_name=f"crop_recommendation_{selected.lower()}.csv",
        mime="text/csv",
    )

    with st.expander("Dataset Summary Statistics"):
        st.dataframe(df[FEATURE_COLS].describe().round(3).T, width='stretch')

    with st.expander("Data Quality Report"):
        quality = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Dtype": df.dtypes.values,
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(quality, width='stretch', hide_index=True)

# ── 15. Page: About ───────────────────────────────────────────────────────────
def render_about():
    section_header("About Project", "Technical documentation and project overview.")

    st.markdown(
        """
        ## AgriSmart: AI-Powered Crop Recommendation System

        ### Problem Statement
        Selecting the wrong crop for given soil and climate conditions leads to 
        reduced yields, financial losses, and environmental impact. 
        AgriSmart provides data-driven recommendations to support informed planting decisions.

        ### Dataset
        | Attribute | Detail |
        |-----------|--------|
        | Source | [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
        | Records | 2,200 rows |
        | Features | 7 numeric input features |
        | Target | `label` — crop name (22 classes) |

        ### Input Features
        | Feature | Unit | Description |
        |---------|------|-------------|
        | N | kg/ha | Nitrogen content in soil |
        | P | kg/ha | Phosphorus content in soil |
        | K | kg/ha | Potassium content in soil |
        | temperature | °C | Average temperature |
        | humidity | % | Relative humidity |
        | ph | — | Soil pH value |
        | rainfall | mm | Annual rainfall |

        ### Technologies Used
        | Category | Technology |
        |----------|-----------|
        | Web Framework | Streamlit |
        | API Framework | Flask |
        | ML Library | scikit-learn |
        | Data Processing | pandas, numpy |
        | Visualisation | Plotly, Matplotlib, Seaborn |
        | Model Persistence | joblib |

        ### ML Algorithms
        - Logistic Regression
        - Decision Tree Classifier
        - Random Forest Classifier with GridSearchCV hyperparameter tuning
        - K-Nearest Neighbors (k=5)
        - Support Vector Machine (RBF kernel)

        ### Evaluation Metrics
        - Accuracy (train & test)
        - Weighted Precision
        - Weighted Recall
        - Weighted F1 Score
        - Confusion Matrix
        - Classification Report

        ### Project Workflow
        ```
        Load CSV → Validate Columns → Remove Duplicates/Nulls →
        Encode Labels → Stratified Train/Test Split (80/20) →
        5-Fold Stratified CV on Training Set (scaling inside Pipelines where required) →
        GridSearchCV Hyperparameter Tuning for Random Forest (training set only) →
        Select Best Model by CV F1 (test set untouched) →
        Final Evaluation on Held-Out Test Set →
        Serve Predictions (Streamlit UI + Flask API)
        ```

        ### Limitations
        - Model performance depends on training data quality and geographic scope.
        - Environmental and soil conditions vary significantly across regions.
        - The system does not account for seasonal variation or market factors.
        - Predictions are probabilistic — predicted probabilities are not guarantees.

        ### Future Improvements
        - Incorporate geospatial and seasonal data.
        - Add soil microbiome and irrigation data as features.
        - Deploy with a dedicated REST API (FastAPI) and database backend.
        - Implement model retraining pipeline with new field data.
        - Add multi-language support for regional farmers.
        """
    )

# ── 16. Flask API ─────────────────────────────────────────────────────────────
def create_api(model, le: LabelEncoder, df: pd.DataFrame,
               metrics_df: pd.DataFrame, best_name: str,
               rf_best_params: dict, rf_best_cv_f1: float):
    """
    Create and return a Flask app with API endpoints.
    model may be a bare estimator or an sklearn Pipeline; it accepts raw
    (unscaled) feature values and handles preprocessing internally.
    """
    from flask import Flask, request, jsonify
    api = Flask(__name__)

    @api.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "service": "AgriSmart Crop Recommendation API"})

    @api.route("/api/dataset", methods=["GET"])
    def dataset_info():
        return jsonify({
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "crop_count": int(df[TARGET_COL].nunique()),
            "feature_names": FEATURE_COLS,
            "target_name": TARGET_COL,
            "crops": sorted(df[TARGET_COL].unique().tolist()),
        })

    @api.route("/api/model-info", methods=["GET"])
    def model_info():
        return jsonify({
            "selected_model": best_name,
            "selection_criterion": f"{CV_FOLDS}-fold stratified cross-validation F1 (weighted)",
            "available_models": list(metrics_df["Model"].tolist()),
            "cross_validation": {
                "folds": CV_FOLDS,
                "strategy": "StratifiedKFold",
                "scoring": "f1_weighted",
                "scores_by_model": metrics_df.set_index("Model")["CV F1 (mean)"].to_dict(),
            },
            "test_evaluation": {
                m: {
                    "test_accuracy": float(row["Test Accuracy"]),
                    "test_f1":       float(row["Test F1"]),
                }
                for m, row in metrics_df.set_index("Model").iterrows()
            },
            "rf_best_params": {k: (None if v is None else v) for k, v in rf_best_params.items()},
            "rf_best_cv_f1": rf_best_cv_f1,
            "feature_names": FEATURE_COLS,
            "target_name": TARGET_COL,
        })

    @api.route("/api/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided."}), 400

        missing = [f for f in FEATURE_COLS if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        try:
            inputs = {f: float(data[f]) for f in FEATURE_COLS}
        except (ValueError, TypeError) as exc:
            return jsonify({"error": f"Invalid input values: {exc}"}), 400

        try:
            predicted_crop, probabilities = make_prediction(model, le, inputs)
        except Exception as exc:
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

        return jsonify({
            "predicted_crop": predicted_crop,
            "probabilities": probabilities,
            "input_values": inputs,
        })

    return api


def run_flask_api(model, le, df, metrics_df, best_name, rf_best_params, rf_best_cv_f1):
    """Run the Flask API server (used by --api standalone mode)."""
    api = create_api(model, le, df, metrics_df, best_name, rf_best_params, rf_best_cv_f1)
    api.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

# ── 17. Streamlit Main App ────────────────────────────────────────────────────
def run_streamlit():
    st.set_page_config(
        page_title="AgriSmart — Crop Recommendation",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding:18px 0 6px 0;">
            <h1 style="margin:0;font-size:2.2rem;font-weight:800;color:#1b4332;">
                🌾 AgriSmart
            </h1>
            <p style="margin:4px 0 0 2px;font-size:1.05rem;color:#40916c;font-weight:600;">
                AI-Powered Crop Recommendation System
            </p>
            <p style="margin:2px 0 0 2px;font-size:0.9rem;color:#555;">
                Data-driven crop recommendations using soil and environmental conditions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load data & train ────────────────────────────────────────────────────
    try:
        with st.spinner("Loading dataset..."):
            df_raw = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    try:
        with st.spinner("Preprocessing data, running cross-validation and hyperparameter tuning — this may take a moment..."):
            # preprocess_data returns raw (unscaled) arrays; scaling is inside each Pipeline
            df_clean, X_train, X_test, y_train, y_test, le = preprocess_data(df_raw)
            models, best_name, best_model, cv_results_df, rf_best_params, rf_best_cv_f1 = train_models(X_train, y_train)
            metrics_df = evaluate_models(models, cv_results_df, X_train, X_test, y_train, y_test)
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

    # Persist label encoder (best_model already saved inside train_models)
    try:
        joblib.dump(le, LE_PATH)
    except Exception:
        pass

    # Flask is NOT started automatically here.
    # Use: python app.py --api   to start the Flask API separately.

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Go to",
            ["Overview", "Crop Recommendation", "Exploratory Data Analysis",
             "Model Performance", "Dataset Explorer", "About Project"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("#### Dataset Info")
        st.markdown(f"**Records:** {len(df_clean):,}")
        st.markdown(f"**Crops:** {df_clean[TARGET_COL].nunique()}")
        st.markdown(f"**Features:** {len(FEATURE_COLS)}")
        st.divider()
        st.markdown("#### Select Model")
        model_names = list(metrics_df["Model"].tolist())
        selected_name = st.selectbox(
            "Active model",
            model_names,
            index=model_names.index(best_name),
            help=f"CV-selected best model: {best_name}",
        )
        selected_model = models[selected_name]
        sel_row = metrics_df[metrics_df["Model"] == selected_name].iloc[0]
        if selected_name == best_name:
            st.success("CV-selected best ★")
        else:
            st.info(f"CV best: {best_name}")
        st.markdown(f"CV F1: **{sel_row['CV F1 (mean)']:.2%}**")
        st.markdown(f"Test Accuracy: **{sel_row['Test Accuracy']:.2%}**")
        st.markdown(f"Test F1: **{sel_row['Test F1']:.2%}**")
        st.divider()
        st.markdown("#### App URL")
        st.markdown("`http://localhost:8501`")
        st.markdown("#### REST API")
        st.markdown(
            f"API running on `localhost:{FLASK_PORT}`  \n"
            f"`GET /api/health`  \n"
            f"`GET /api/dataset`  \n"
            f"`GET /api/model-info`  \n"
            f"`POST /api/predict`"
        )
        st.divider()
        st.caption("AgriSmart by Agniva Bhattacharya")

    # ── Route pages ──────────────────────────────────────────────────────────
    if page == "Overview":
        render_overview(df_clean, metrics_df, selected_name)
    elif page == "Crop Recommendation":
        render_recommendation(selected_model, le, df_clean)
    elif page == "Exploratory Data Analysis":
        render_eda(df_clean)
    elif page == "Model Performance":
        render_model_performance(
            models, metrics_df, selected_name, best_name,
            X_train, X_test, y_train, y_test, le,
            rf_best_params, rf_best_cv_f1,
        )
    elif page == "Dataset Explorer":
        render_dataset_explorer(df_clean)
    elif page == "About Project":
        render_about()

# ── 18. Entry Point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AgriSmart Crop Recommendation System")
    parser.add_argument(
        "--api", action="store_true",
        help="Run Flask API server only (port 5000), without Streamlit."
    )
    # argparse and streamlit both consume sys.argv; parse only known args
    args, _ = parser.parse_known_args()

    if args.api:
        # Standalone API mode: python app.py --api
        # Scaling happens inside each Pipeline; no standalone scaler is created here.
        print("Loading dataset, running CV and hyperparameter tuning for API mode...")
        if not os.path.exists(DATASET_PATH):
            print(f"ERROR: Dataset not found: {DATASET_PATH}")
            sys.exit(1)
        df_raw = pd.read_csv(DATASET_PATH)
        missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
        if missing:
            print(f"ERROR: Missing columns: {missing}")
            sys.exit(1)
        # Preprocess — returns raw (unscaled) arrays, consistent with preprocess_data()
        df_clean = df_raw.drop_duplicates().dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
        X = df_clean[FEATURE_COLS].values
        y = df_clean[TARGET_COL].values
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
        )
        # Train using the same pipeline as Streamlit mode (bypassing Streamlit cache)
        _train_fn = getattr(train_models, "__wrapped__", train_models)
        _eval_fn  = getattr(evaluate_models, "__wrapped__", evaluate_models)
        models, best_name, best_model, cv_results_df, rf_best_params, rf_best_cv_f1 = \
            _train_fn(X_train, y_train)
        metrics_df = _eval_fn(
            models, cv_results_df, X_train, X_test, y_train, y_test
        )
        # Fix 5: print the CV F1 score for the actually selected model, not always RF
        selected_cv_f1 = cv_results_df.loc[
            cv_results_df["Model"] == best_name, "CV F1 (mean)"
        ].iloc[0]
        print(f"CV-selected best model: {best_name}  (CV F1: {selected_cv_f1:.4f})")
        api = create_api(best_model, le, df_clean, metrics_df,
                         best_name, rf_best_params, rf_best_cv_f1)
        print(f"Starting Flask API on http://0.0.0.0:{FLASK_PORT}")
        api.run(host="0.0.0.0", port=FLASK_PORT, debug=False)
    else:
        # Default: Streamlit mode (launched via `streamlit run app.py`)
        run_streamlit()


if __name__ == "__main__":
    main()
