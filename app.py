from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


APP_DIR = Path(__file__).resolve().parent
DEMO_CSV = APP_DIR / "sample_training_data_demo.csv"

FEATURES = [
    "soil_moisture_pct",
    "soil_temperature_c",
    "soil_ph",
    "electrical_conductivity_ds_m",
    "air_temperature_c",
    "relative_humidity_pct",
    "rainfall_24h_mm",
    "crop_type",
]
TARGET = "irrigation_class"
NUMERIC_FEATURES = [c for c in FEATURES if c != "crop_type"]
CROP_OPTIONS = ["Tomato", "Wheat", "Rice", "Maize", "Soybean", "Cotton", "Potato"]
CLASS_ORDER = ["No irrigation", "Moderate irrigation", "High irrigation"]


st.set_page_config(
    page_title="AI Soil Health & Precision Irrigation",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 AI Soil Health & Precision Irrigation Simulator")
st.caption("Random Forest classifier • Interactive sensor controls • CSV-based model training")

st.warning(
    "DEMO MODE NOTE: the included demo CSV is synthetic and its labels are generated from illustrative rules. "
    "It is only for testing the interface and explaining the ML workflow—not for real farming decisions. "
    "For a scientifically defensible result, upload a real dataset with observed irrigation labels."
)

with st.expander("What this model does—and what it does not do", expanded=False):
    st.markdown(
        "- Inputs: soil moisture, soil temperature, soil pH, electrical conductivity (EC), air temperature, "
        "relative humidity, rainfall in the previous 24 hours, and crop type.\n"
        "- Output: one category—**No irrigation**, **Moderate irrigation**, or **High irrigation**.\n"
        "- The classifier predicts a category, not an exact water volume in litres or millimetres. To estimate an exact amount, "
        "you would need a suitable numeric target such as observed irrigation depth (mm) and a regression model.\n"
        "- Soil pH and EC are included as model inputs, but the irrigation decision must ultimately be validated for the selected crop, "
        "soil texture, root depth, and growth stage. This prototype is for an academic simulation, not operational irrigation advice."
    )


# ------------------------------ Data loading ------------------------------

def load_demo_data() -> pd.DataFrame:
    if DEMO_CSV.exists():
        return pd.read_csv(DEMO_CSV)
    # Fallback generator, if the project CSV is missing.
    rng = np.random.default_rng(42)
    n = 600
    crops = rng.choice(CROP_OPTIONS, size=n)
    crop_targets = {
        "Tomato": 30, "Wheat": 25, "Rice": 36, "Maize": 29,
        "Soybean": 28, "Cotton": 27, "Potato": 32,
    }
    moisture = rng.uniform(5, 45, n)
    soil_temp = rng.uniform(12, 38, n)
    soil_ph = rng.uniform(4.5, 9.0, n)
    ec = rng.uniform(0.1, 5.5, n)
    air_temp = rng.uniform(8, 43, n)
    humidity = rng.uniform(15, 100, n)
    rainfall = np.clip(rng.gamma(1.4, 5.0, n), 0, 70)
    score = np.array([crop_targets[c] for c in crops]) - moisture
    score += 0.22 * np.maximum(air_temp - 25, 0)
    score += 0.10 * np.maximum(soil_temp - 20, 0)
    score -= 0.30 * rainfall
    score -= 0.025 * (humidity - 50)
    labels = np.where(score < 0, CLASS_ORDER[0], np.where(score < 10, CLASS_ORDER[1], CLASS_ORDER[2]))
    return pd.DataFrame({
        "soil_moisture_pct": moisture,
        "soil_temperature_c": soil_temp,
        "soil_ph": soil_ph,
        "electrical_conductivity_ds_m": ec,
        "air_temperature_c": air_temp,
        "relative_humidity_pct": humidity,
        "rainfall_24h_mm": rainfall,
        "crop_type": crops,
        "irrigation_class": labels,
    })


def normalize_target(value: object) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    mapping = {
        "no irrigation": "No irrigation",
        "no irrigation required": "No irrigation",
        "none": "No irrigation",
        "no": "No irrigation",
        "0": "No irrigation",
        "moderate irrigation": "Moderate irrigation",
        "moderate": "Moderate irrigation",
        "medium": "Moderate irrigation",
        "high irrigation": "High irrigation",
        "high": "High irrigation",
    }
    return mapping.get(text, str(value).strip())


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    clean = df[FEATURES + [TARGET]].copy()
    for col in NUMERIC_FEATURES:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean["crop_type"] = clean["crop_type"].astype("string").fillna("Unknown").astype(str)
    clean[TARGET] = clean[TARGET].apply(normalize_target)
    clean = clean.dropna(subset=[TARGET])
    clean = clean[clean[TARGET].isin(CLASS_ORDER)]
    if len(clean) < 12:
        raise ValueError("At least 12 usable rows with a valid irrigation_class are recommended for this demonstration.")
    if clean[TARGET].nunique() < 2:
        raise ValueError("The uploaded data must contain at least two different irrigation classes.")

    # Remove rows that have no numeric sensor readings at all.
    clean = clean.loc[clean[NUMERIC_FEATURES].notna().any(axis=1)].copy()
    if len(clean) < 12:
        raise ValueError("Too few rows remain after cleaning. Check the numeric input columns and units.")
    X = clean[FEATURES]
    y = clean[TARGET]
    notes = []
    for col in NUMERIC_FEATURES:
        missing_count = int(X[col].isna().sum())
        if missing_count:
            notes.append(f"{col}: {missing_count} missing value(s); these are imputed using the training-set median.")
    return X, y, notes


def train_model(df: pd.DataFrame):
    X, y, data_notes = prepare_data(df)
    class_counts = y.value_counts()
    can_stratify = y.nunique() > 1 and int(class_counts.min()) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, ["crop_type"]),
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=CLASS_ORDER)
    return model, X, y, accuracy, report, matrix, X_test, y_test, predictions, data_notes


with st.sidebar:
    st.header("1. Training data")
    data_source = st.radio(
        "Choose the data source",
        ["Illustrative demo dataset", "Upload my CSV"],
        help="Use the demo to test the interface. Upload a published dataset after adding a valid irrigation_class target column.",
    )
    uploaded_file = None
    if data_source == "Upload my CSV":
        uploaded_file = st.file_uploader("Upload a CSV training file", type=["csv"])
    st.download_button(
        "Download CSV column template",
        data=("soil_moisture_pct,soil_temperature_c,soil_ph,electrical_conductivity_ds_m,air_temperature_c,relative_humidity_pct,rainfall_24h_mm,crop_type,irrigation_class\n"),
        file_name="irrigation_training_template.csv",
        mime="text/csv",
        help="The template lists the exact column names the app expects. Add your real data rows underneath.",
    )
    if DEMO_CSV.exists():
        st.download_button(
            "Download demo training data",
            data=DEMO_CSV.read_bytes(),
            file_name="sample_training_data_demo.csv",
            mime="text/csv",
        )


if data_source == "Upload my CSV":
    if uploaded_file is None:
        st.info("Upload a CSV file in the sidebar to train the model on your data. Until then, the simulator is waiting for the file.")
        st.stop()
    try:
        source_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
    dataset_label = f"Uploaded file: {uploaded_file.name}"
else:
    source_df = load_demo_data()
    dataset_label = "Synthetic demonstration data"

try:
    model, X_all, y_all, accuracy, report, matrix, X_test, y_test, y_pred, data_notes = train_model(source_df)
except Exception as exc:
    st.error(f"The model could not be trained: {exc}")
    st.markdown(
        "**Required CSV columns**\n\n"
        "`soil_moisture_pct`, `soil_temperature_c`, `soil_ph`, `electrical_conductivity_ds_m`, "
        "`air_temperature_c`, `relative_humidity_pct`, `rainfall_24h_mm`, `crop_type`, `irrigation_class`.\n\n"
        "The last column is the target label. Without observed labels (or a documented rule for creating labels), "
        "a supervised Random Forest classifier cannot learn the three categories."
    )
    st.stop()

if data_source == "Illustrative demo dataset":
    st.info("Currently using synthetic sample rows. Replace them with a labelled, published dataset before presenting the model as research-based.")
else:
    st.success(f"Model trained from {dataset_label}.")

for note in data_notes:
    st.caption("Data note: " + note)

# ------------------------------- Model stats ------------------------------
metric_cols = st.columns(4)
metric_cols[0].metric("Training rows", f"{len(X_all):,}")
metric_cols[1].metric("Test accuracy", f"{accuracy:.1%}")
metric_cols[2].metric("Classes detected", str(y_all.nunique()))
metric_cols[3].metric("Algorithm", "Random Forest")

with st.expander("Model evaluation (held-out test set)", expanded=False):
    st.caption("The test set was held out from fitting. Accuracy is meaningful only when the uploaded labels are real and the data split is appropriate.")
    report_df = pd.DataFrame(report).T
    report_df = report_df.loc[[c for c in CLASS_ORDER if c in report_df.index] + [c for c in ["accuracy", "macro avg", "weighted avg"] if c in report_df.index]]
    st.dataframe(report_df.round(3), use_container_width=True)
    cm_df = pd.DataFrame(matrix, index=[f"Actual: {c}" for c in CLASS_ORDER], columns=[f"Predicted: {c}" for c in CLASS_ORDER])
    st.write("Confusion matrix")
    st.dataframe(cm_df, use_container_width=True)

with st.expander("Feature importance", expanded=False):
    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importance = model.named_steps["classifier"].feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        importance_df["Feature"] = importance_df["Feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__crop_type_", "crop_type = ", regex=False)
        importance_df = importance_df.sort_values("Importance", ascending=False).head(15)
        st.bar_chart(importance_df.set_index("Feature"))
    except Exception:
        st.caption("Feature importance could not be displayed for this dataset.")

st.divider()
st.header("2. Enter current field conditions")
st.write("Adjust the sliders or type a value. These controls act like virtual sensor dials.")

left, mid, right = st.columns(3)
with left:
    available_crops = sorted(X_all["crop_type"].astype(str).unique().tolist())
    if not available_crops:
        available_crops = CROP_OPTIONS
    crop_type = st.selectbox("Crop type", available_crops, index=0)
    soil_moisture = st.slider("Soil moisture (%)", 0.0, 100.0, 24.0, 0.5, help="Use percent. If your source reports m³/m³, multiply by 100 first.")
    soil_temp = st.slider("Soil temperature (°C)", -5.0, 60.0, 24.0, 0.5)
    soil_ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)
with mid:
    ec = st.slider("Electrical conductivity — EC (dS/m)", 0.0, 20.0, 1.0, 0.1, help="Keep units consistent with the training CSV. dS/m and mS/cm are numerically equivalent.")
    air_temp = st.slider("Air temperature (°C)", -10.0, 55.0, 30.0, 0.5)
    humidity = st.slider("Relative humidity (%)", 0.0, 100.0, 55.0, 1.0)
with right:
    rainfall = st.slider("Rainfall in previous 24 hours (mm)", 0.0, 200.0, 2.0, 0.5)
    st.markdown("#### Try changing the sliders")
    st.caption("For a dry scenario, lower moisture and rainfall and raise temperature. For a rainy scenario, raise moisture, humidity, and rainfall.")

input_row = pd.DataFrame([{
    "soil_moisture_pct": soil_moisture,
    "soil_temperature_c": soil_temp,
    "soil_ph": soil_ph,
    "electrical_conductivity_ds_m": ec,
    "air_temperature_c": air_temp,
    "relative_humidity_pct": humidity,
    "rainfall_24h_mm": rainfall,
    "crop_type": crop_type,
}])

if st.button("🔍 Predict irrigation requirement", type="primary", use_container_width=True):
    predicted_class = str(model.predict(input_row[FEATURES])[0])
    probability_values = model.predict_proba(input_row[FEATURES])[0]
    classes = model.named_steps["classifier"].classes_
    probability_df = pd.DataFrame({"Irrigation category": classes, "Model probability": probability_values})
    probability_df = probability_df.sort_values("Model probability", ascending=False)

    st.subheader("Prediction")
    if predicted_class == "No irrigation":
        st.success(f"Recommended category: **{predicted_class}**")
    elif predicted_class == "Moderate irrigation":
        st.warning(f"Recommended category: **{predicted_class}**")
    else:
        st.error(f"Recommended category: **{predicted_class}**")
    st.caption("This is the classifier's category prediction, not a validated irrigation volume recommendation.")
    st.write("Class probabilities")
    st.dataframe(probability_df.style.format({"Model probability": "{:.1%}"}), use_container_width=True, hide_index=True)

    soil_flags = []
    if soil_ph < 5.5:
        soil_flags.append("pH is acidic; check crop-specific tolerance and soil amendment guidance.")
    elif soil_ph > 7.8:
        soil_flags.append("pH is alkaline; check crop-specific nutrient availability and soil guidance.")
    if ec >= 4:
        soil_flags.append("EC is high enough to warrant a salinity check for many crops; crop tolerance varies.")
    if soil_moisture >= 80:
        soil_flags.append("Moisture reading is very high; check drainage and sensor calibration before irrigating.")
    if soil_flags:
        st.markdown("**Rule-based soil-condition flags (separate from the Random Forest output):**")
        for flag in soil_flags:
            st.write("- " + flag)
    else:
        st.caption("No simple pH/EC/moisture flag was triggered by the prototype's screening rules.")

    result = input_row.copy()
    result["predicted_irrigation_class"] = predicted_class
    result["prediction_probability"] = float(np.max(probability_values))
    st.download_button(
        "Download this prediction as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="irrigation_prediction.csv",
        mime="text/csv",
    )

st.divider()
st.header("3. Training data preview")
st.caption(f"Loaded dataset: {dataset_label}. Valid categories after cleaning: {', '.join(sorted(y_all.unique()))}.")
st.dataframe(source_df.head(12), use_container_width=True)
st.caption("For your report, cite the original source of each dataset and document any unit conversions, filtering, or rules used to create the target labels.")
