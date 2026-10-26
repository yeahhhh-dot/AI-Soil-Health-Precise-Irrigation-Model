from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


APP_DIR = Path(__file__).resolve().parent
DEMO_CSV = APP_DIR / "demo_100_conditions.csv"

REGION = "Khandesh, North Maharashtra"
CROP = "Cotton"
SOIL_OPTIONS = [
    "Sandy loam",
    "Clay loam",
    "Black soil",
    "Red-mixed-black soil",
    "Deep alluvial soil",
    "Black clayey soil",
    "Red soil",
    "Red + red-mixed-black soil",
]
SEASON_OPTIONS = ["April–June", "July–September", "October–December", "January–March"]

# IMPORTANT: These are illustrative starting values for a classroom simulation,
# NOT measured hydraulic values for every Khandesh field. Replace them with
# field/laboratory measurements or verified location-specific published data.
# Moisture values are volumetric water content (% volume/volume).
SOIL_DEFAULTS = {
    "Sandy loam": {"field_capacity_vwc_pct": 22.0, "saturation_vwc_pct": 43.0},
    "Clay loam": {"field_capacity_vwc_pct": 32.0, "saturation_vwc_pct": 48.0},
    "Black soil": {"field_capacity_vwc_pct": 36.0, "saturation_vwc_pct": 53.0},
    "Red-mixed-black soil": {"field_capacity_vwc_pct": 29.0, "saturation_vwc_pct": 48.0},
    "Deep alluvial soil": {"field_capacity_vwc_pct": 28.0, "saturation_vwc_pct": 45.0},
    "Black clayey soil": {"field_capacity_vwc_pct": 38.0, "saturation_vwc_pct": 54.0},
    "Red soil": {"field_capacity_vwc_pct": 23.0, "saturation_vwc_pct": 44.0},
    "Red + red-mixed-black soil": {"field_capacity_vwc_pct": 26.0, "saturation_vwc_pct": 46.0},
}

NUMERIC_FEATURES = [
    "soil_moisture_vwc_pct",
    "target_moisture_vwc_pct",
    "net_refill_depth_mm",
    "soil_temperature_c",
    "soil_ph",
    "electrical_conductivity_ds_m",
    "air_temperature_c",
    "relative_humidity_pct",
    "rainfall_24h_mm",
    "root_zone_depth_cm",
    "field_capacity_vwc_pct",
    "saturation_vwc_pct",
    "irrigation_efficiency_pct",
]
CATEGORICAL_FEATURES = ["soil_type", "season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "irrigation_depth_mm"
REQUIRED_COLUMNS = FEATURES + [TARGET]
# The refill depth is a deterministic derived feature; uploaded files may omit it.
REQUIRED_INPUT_COLUMNS = [c for c in REQUIRED_COLUMNS if c != "net_refill_depth_mm"]

st.set_page_config(
    page_title="Cotton Drip Irrigation Simulator | Khandesh",
    page_icon="🌱",
    layout="wide",
)


# ------------------------------- Calculations ------------------------------
def estimate_gross_irrigation_depth_mm(
    current_moisture_pct: float,
    target_moisture_pct: float,
    root_zone_depth_cm: float,
    efficiency_pct: float,
) -> tuple[float, float]:
    """Return (net root-zone refill depth, gross applied depth), both in mm.

    Assumes the moisture percentages are volumetric water content (% v/v),
    the same root-zone depth is represented by the current sensor reading, and
    no spatial variation/runoff/deep drainage is explicitly simulated.
    """
    delta_vwc_fraction = max(target_moisture_pct - current_moisture_pct, 0.0) / 100.0
    root_depth_mm = root_zone_depth_cm * 10.0
    net_depth_mm = delta_vwc_fraction * root_depth_mm
    efficiency_fraction = max(min(efficiency_pct / 100.0, 1.0), 0.01)
    gross_depth_mm = net_depth_mm / efficiency_fraction
    return net_depth_mm, gross_depth_mm


@st.cache_data(show_spinner=False)
def generate_demo_data(n_rows: int = 100, seed: int = 2026) -> pd.DataFrame:
    """Generate 100 illustrative scenario labels from the transparent refill formula.

    These are simulated scenarios, not observed field measurements. The random
    forest demo therefore learns to approximate this formula; it is not evidence
    of real-world accuracy.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        soil = SOIL_OPTIONS[i % len(SOIL_OPTIONS)]
        season = SEASON_OPTIONS[(i // len(SOIL_OPTIONS)) % len(SEASON_OPTIONS)]
        defaults = SOIL_DEFAULTS[soil]
        fc = float(np.clip(rng.normal(defaults["field_capacity_vwc_pct"], 1.5), 12, 45))
        sat = float(max(fc + 8, rng.normal(defaults["saturation_vwc_pct"], 1.8)))
        sat = min(sat, 60.0)
        current = float(rng.uniform(6.0, min(sat - 1.0, 48.0)))
        # The target is usually field capacity, with some educational near-saturation cases.
        if rng.random() < 0.75:
            target_vwc = fc
        else:
            target_vwc = sat * float(rng.uniform(0.95, 0.99))
        root_depth = float(rng.choice([30, 45, 60, 75, 90]))
        efficiency = float(rng.uniform(80, 95))
        soil_temp = float(rng.uniform(18, 38))
        air_temp = float(rng.uniform(20, 43))
        humidity = float(rng.uniform(20, 95))
        rainfall = float(np.clip(rng.gamma(1.4, 5.0), 0, 60))
        ph = float(rng.uniform(5.0, 8.8))
        ec = float(rng.uniform(0.2, 4.0))
        net_depth, gross_depth = estimate_gross_irrigation_depth_mm(current, target_vwc, root_depth, efficiency)
        rows.append({
            "region": REGION,
            "crop_type": CROP,
            "soil_type": soil,
            "season": season,
            "soil_moisture_vwc_pct": round(current, 2),
            "target_moisture_vwc_pct": round(target_vwc, 2),
            "net_refill_depth_mm": round(net_depth, 3),
            "soil_temperature_c": round(soil_temp, 2),
            "soil_ph": round(ph, 2),
            "electrical_conductivity_ds_m": round(ec, 2),
            "air_temperature_c": round(air_temp, 2),
            "relative_humidity_pct": round(humidity, 1),
            "rainfall_24h_mm": round(rainfall, 2),
            "root_zone_depth_cm": root_depth,
            "field_capacity_vwc_pct": round(fc, 2),
            "saturation_vwc_pct": round(sat, 2),
            "irrigation_efficiency_pct": round(efficiency, 1),
            "irrigation_depth_mm": round(gross_depth, 3),
        })
    return pd.DataFrame(rows)


def make_regression_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    clean = df.copy()
    # Compute this derived feature rather than trusting a manually entered value.
    for column in [c for c in NUMERIC_FEATURES if c != "net_refill_depth_mm"] + [TARGET]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean["net_refill_depth_mm"] = (
        np.maximum(clean["target_moisture_vwc_pct"] - clean["soil_moisture_vwc_pct"], 0.0)
        / 100.0 * clean["root_zone_depth_cm"] * 10.0
    )
    for column in CATEGORICAL_FEATURES:
        clean[column] = clean[column].astype("string").str.strip()
        clean[column] = clean[column].replace("", pd.NA)
    clean = clean[FEATURES + [TARGET]].copy()
    clean = clean.dropna(subset=[TARGET])
    clean = clean.dropna(subset=["soil_type", "season"])
    clean = clean[clean[TARGET] >= 0]
    clean = clean[clean["soil_type"].isin(SOIL_OPTIONS)]
    clean = clean[clean["season"].isin(SEASON_OPTIONS)]
    if len(clean) < 20:
        raise ValueError("Please provide at least 20 usable labelled rows; 100 or more is a better class-project target.")
    if clean[TARGET].nunique() < 2:
        raise ValueError("The irrigation_depth_mm target must contain more than one numeric value.")
    return clean


def train_regressor(df: pd.DataFrame):
    clean = prepare_training_data(df)
    X = clean[FEATURES]
    y = clean[TARGET]
    test_size = 0.2 if len(clean) >= 25 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = make_regression_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")
    return model, clean, X_test, y_test, predictions, mae, rmse, r2


# --------------------------------- Sidebar ----------------------------------
with st.sidebar:
    st.header("1. Model training")
    model_source = st.radio(
        "Training dataset",
        ["Built-in 100 simulated conditions", "Upload labelled field data CSV"],
        help=(
            "The built-in rows are generated from the water-balance equation to test the workflow. "
            "For research claims, upload real measurements with observed irrigation-depth labels."
        ),
    )
    uploaded_csv = None
    if model_source == "Upload labelled field data CSV":
        uploaded_csv = st.file_uploader("Upload labelled CSV", type=["csv"])
    template_text = ",".join(REQUIRED_COLUMNS) + "\n"
    st.download_button(
        "Download training-data template",
        data=template_text,
        file_name="cotton_khandesh_training_template.csv",
        mime="text/csv",
    )
    demo_df = generate_demo_data()
    st.download_button(
        "Download the 100 simulated conditions",
        data=demo_df.to_csv(index=False).encode("utf-8"),
        file_name="demo_100_conditions.csv",
        mime="text/csv",
        help="For interface/model workflow testing only—not measured field data.",
    )

if model_source == "Upload labelled field data CSV" and uploaded_csv is not None:
    try:
        raw_training_df = pd.read_csv(uploaded_csv)
        dataset_name = f"Uploaded field dataset: {uploaded_csv.name}"
        is_synthetic = False
    except Exception as exc:
        st.error(f"I could not read the uploaded CSV: {exc}")
        st.stop()
else:
    raw_training_df = demo_df
    dataset_name = "100 simulated classroom scenarios"
    is_synthetic = True
    if model_source == "Upload labelled field data CSV" and uploaded_csv is None:
        st.info("Upload a labelled CSV to train on your own data. The app is showing the 100-scenario demo model for now.")

try:
    model, trained_df, X_test, y_test, y_pred, mae, rmse, r2 = train_regressor(raw_training_df)
except Exception as exc:
    st.error(f"The Random Forest regressor could not be trained: {exc}")
    st.markdown("**Required columns:**\n\n`" + "`, `".join(REQUIRED_COLUMNS) + "`\n\nThe `irrigation_depth_mm` column must be an observed gross irrigation depth (in mm per event) for research data. Do not use class labels such as High/Moderate/No irrigation in this regression target.")
    st.stop()

st.title("🌱 Cotton Drip Irrigation Simulator")
st.caption(f"Fixed region: **{REGION}** · Fixed crop: **{CROP}** · Random Forest regression + transparent soil-water balance")

if is_synthetic:
    st.warning(
        "DEMONSTRATION MODE: the 100 training rows are simulated, and the target water depths are calculated from the same simplified water-balance equation used below. "
        "The displayed model metrics only measure how well Random Forest approximates these generated rows; they are NOT real-world accuracy."
    )
else:
    st.success(f"Model trained from {dataset_name}. Verify the paper's methods, units, site, dates, and irrigation-event labels before interpreting results.")

with st.expander("Important scientific assumptions", expanded=False):
    st.markdown(
        "- Cotton and Khandesh (North Maharashtra) are fixed in this prototype. Soil type and seasonal quarter are selectable.\n"
        "- Moisture readings must be **volumetric water content (VWC, % by volume)** or calibrated to VWC. A sensor's arbitrary 0–100 display is not automatically VWC.\n"
        "- A target of 95–100% of saturation is **not the same as 95–100% sensor moisture** and is generally not a sensible routine target: waterlogging and oxygen stress are possible. Field capacity is the recommended default for this educational calculator.\n"
        "- Soil field-capacity and saturation defaults are illustrative placeholders, not measured Khandesh soil-profile values. Change them to match the selected field's lab/published values.\n"
        "- The refill equation is: net depth (mm) = max(target VWC − current VWC, 0) × root-zone depth (mm); gross depth = net depth ÷ application efficiency. 1 mm over 1 m² equals 1 litre.\n"
        "- The displayed volume assumes a uniform root-zone refill over the entered area. Actual drip wetting bulbs, emitter spacing, runoff, drainage, crop stage, and weather-based ET should be calibrated separately.\n"
        "- Previous-24-hour rainfall is shown to the model but is not subtracted again from the transparent refill equation because the current moisture sensor reading should already reflect rain that has reached the measured soil."
    )

metric_cols = st.columns(4)
metric_cols[0].metric("Training rows", f"{len(trained_df):,}")
metric_cols[1].metric("Test MAE", f"{mae:.2f} mm")
metric_cols[2].metric("Test RMSE", f"{rmse:.2f} mm")
metric_cols[3].metric("Test R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")

with st.expander("Random Forest evaluation (held-out rows)", expanded=False):
    st.caption("For the built-in data, these are synthetic self-consistency metrics only. With real data, use a suitable time/site-aware split when rows are correlated.")
    metrics_df = pd.DataFrame({
        "Observed irrigation depth (mm)": y_test.to_numpy(),
        "Random Forest prediction (mm)": y_pred,
        "Absolute error (mm)": np.abs(y_test.to_numpy() - y_pred),
    })
    st.dataframe(metrics_df.round(3), use_container_width=True, hide_index=True)

with st.expander("Feature importance", expanded=False):
    try:
        prep = model.named_steps["preprocessor"]
        names = prep.get_feature_names_out()
        importances = model.named_steps["regressor"].feature_importances_
        fi = pd.DataFrame({"Feature": names, "Importance": importances})
        fi["Feature"] = fi["Feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
        fi = fi.sort_values("Importance", ascending=False).head(15)
        st.bar_chart(fi.set_index("Feature"))
    except Exception:
        st.caption("Feature importance is unavailable for this dataset.")

st.divider()
st.header("2. Enter current field conditions")
st.write("Choose one of the listed soils and seasonal windows, then enter current virtual sensor readings.")

col1, col2, col3 = st.columns(3)
with col1:
    soil_type = st.selectbox("Soil type", SOIL_OPTIONS, index=2)
    season = st.selectbox("Time of year", SEASON_OPTIONS, index=3)
    soil_moisture = st.slider(
        "Current soil moisture — VWC (%)", 0.0, 60.0, 17.0, 0.5,
        help="Use volumetric water content (volume of water / volume of soil × 100), not an uncalibrated sensor scale.",
    )
    soil_temp = st.slider("Soil temperature (°C)", -5.0, 60.0, 28.0, 0.5)
    soil_ph = st.slider("Soil pH", 0.0, 14.0, 7.5, 0.1)
with col2:
    ec = st.slider("Electrical conductivity — EC (dS/m)", 0.0, 20.0, 1.0, 0.1)
    air_temp = st.slider("Air temperature (°C)", -10.0, 55.0, 32.0, 0.5)
    humidity = st.slider("Relative humidity (%)", 0.0, 100.0, 40.0, 1.0)
    rainfall = st.slider("Rainfall in previous 24 hours (mm)", 0.0, 200.0, 0.0, 0.5)
with col3:
    st.markdown("#### Soil hydraulic values")
    profile = SOIL_DEFAULTS[soil_type]
    field_capacity = st.number_input(
        "Estimated field capacity (% VWC)", min_value=1.0, max_value=55.0,
        value=float(profile["field_capacity_vwc_pct"]), step=0.5,
        key=f"field_capacity_{soil_type}",
        help="Illustrative placeholder. Replace with measured/published values for your field and sensor depth.",
    )
    saturation = st.number_input(
        "Estimated saturation/porosity (% VWC)", min_value=5.0, max_value=65.0,
        value=float(profile["saturation_vwc_pct"]), step=0.5,
        key=f"saturation_{soil_type}",
        help="Illustrative placeholder. Saturation VWC should be greater than field capacity.",
    )
    root_depth = st.slider("Active root-zone depth (cm)", 10, 150, 60, 5)
    efficiency = st.slider("Drip application efficiency (%)", 50, 100, 90, 1)

if saturation <= field_capacity:
    st.error("Saturation/porosity should be higher than field capacity. Please correct the hydraulic values before calculating.")
    st.stop()

st.subheader("Choose the refill target")
target_mode = st.radio(
    "Target method",
    ["Refill to estimated field capacity (recommended)", "Near saturation (95–100% of estimated saturation — demo only)"],
    horizontal=False,
)
if target_mode.startswith("Refill"):
    target_moisture = float(field_capacity)
    st.info(f"Target moisture is set to estimated field capacity: **{target_moisture:.1f}% VWC**.")
else:
    sat_fraction_pct = st.slider("Target as a percentage of saturation", 95, 100, 95, 1)
    target_moisture = float(saturation * sat_fraction_pct / 100.0)
    st.warning(
        f"This sets target VWC to {target_moisture:.1f}% from an estimated saturation of {saturation:.1f}% VWC. "
        "Near-saturation irrigation can cause waterlogging/deep drainage and is not recommended as a routine cotton target without agronomic supervision."
    )

st.subheader("Drip-system setup")
area_col, emitter_col, flow_col = st.columns(3)
with area_col:
    area_m2 = st.number_input("Area being irrigated (m²)", min_value=0.1, max_value=100000.0, value=1.0, step=1.0)
with emitter_col:
    emitter_count = st.number_input("Number of emitters serving this area", min_value=1, max_value=100000, value=1, step=1)
with flow_col:
    emitter_flow_lph = st.number_input("Rated flow per emitter (L/hour)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)

net_depth_mm, gross_depth_mm = estimate_gross_irrigation_depth_mm(
    soil_moisture, target_moisture, float(root_depth), float(efficiency)
)
water_litres = gross_depth_mm * float(area_m2)  # 1 mm depth on 1 m² = 1 L
required_flow_lph = water_litres  # Requested duration is exactly 1 hour
installed_capacity_lph = float(emitter_count) * float(emitter_flow_lph)
runtime_hours = water_litres / installed_capacity_lph if installed_capacity_lph > 0 else float("inf")

input_row = pd.DataFrame([{
    "soil_moisture_vwc_pct": soil_moisture,
    "target_moisture_vwc_pct": target_moisture,
    "net_refill_depth_mm": net_depth_mm,
    "soil_temperature_c": soil_temp,
    "soil_ph": soil_ph,
    "electrical_conductivity_ds_m": ec,
    "air_temperature_c": air_temp,
    "relative_humidity_pct": humidity,
    "rainfall_24h_mm": rainfall,
    "root_zone_depth_cm": root_depth,
    "field_capacity_vwc_pct": field_capacity,
    "saturation_vwc_pct": saturation,
    "irrigation_efficiency_pct": efficiency,
    "soil_type": soil_type,
    "season": season,
}])[FEATURES]
rf_depth_mm = max(float(model.predict(input_row)[0]), 0.0)

if st.button("💧 Calculate drip irrigation requirement", type="primary", use_container_width=True):
    st.header("3. Irrigation estimate")
    if soil_moisture >= target_moisture:
        st.success("No refill water is indicated by the selected target: current moisture is already at or above it.")
    else:
        st.caption("Transparent water-balance estimate (the primary result; based on your stated assumptions)")
        metrics = st.columns(4)
        metrics[0].metric("Current VWC", f"{soil_moisture:.1f}%")
        metrics[1].metric("Target VWC", f"{target_moisture:.1f}%")
        metrics[2].metric("Net refill depth", f"{net_depth_mm:.2f} mm")
        metrics[3].metric("Gross applied depth", f"{gross_depth_mm:.2f} mm")

        volume_cols = st.columns(3)
        volume_cols[0].metric("Total water for selected area", f"{water_litres:,.2f} L")
        volume_cols[1].metric("Flow needed to deliver in 1 hour", f"{required_flow_lph:,.2f} L/h")
        volume_cols[2].metric("Installed emitter capacity", f"{installed_capacity_lph:,.2f} L/h")

        st.write(f"**Estimated runtime at the entered emitter capacity:** {runtime_hours:,.2f} hour(s).")
        if runtime_hours > 1:
            st.warning(
                "The entered emitters cannot deliver the calculated volume in one hour. The calculation reports the flow required for a one-hour application and the runtime at your current emitter capacity; "
                "do not increase flow blindly—check emitter spacing, infiltration, root depth and avoid runoff/deep percolation."
            )
        if target_mode.startswith("Near saturation"):
            st.error("Near-saturation result is an educational scenario only. It should not be interpreted as a field irrigation recommendation.")
        if soil_moisture > saturation:
            st.warning("The current moisture reading exceeds the entered saturation estimate. Check sensor calibration, units, and soil-profile parameters.")
        if soil_ph < 5.5:
            st.info("Soil pH is acidic; check crop-specific soil testing and nutrient guidance.")
        elif soil_ph > 8.2:
            st.info("Soil pH is alkaline; check crop-specific soil testing and nutrient availability.")
        if ec >= 4:
            st.info("EC is relatively high and may warrant a salinity assessment; crop response depends on measurement method and local conditions.")

        st.subheader("Random Forest regression estimate (experimental comparison)")
        compare_cols = st.columns(2)
        compare_cols[0].metric("Random Forest gross depth", f"{rf_depth_mm:.2f} mm")
        compare_cols[1].metric("Water-balance gross depth", f"{gross_depth_mm:.2f} mm")
        st.caption(
            "The Random Forest output is an empirical estimate from the selected training CSV. With demo data it only approximates formula-generated labels; "
            "for a research model, train and test on real, matched irrigation-event observations. Use the transparent water-balance value to understand the units and assumptions."
        )

        result_df = input_row.copy()
        result_df["region"] = REGION
        result_df["crop_type"] = CROP
        result_df["target_mode"] = target_mode
        result_df["net_irrigation_depth_mm"] = net_depth_mm
        result_df["water_balance_gross_depth_mm"] = gross_depth_mm
        result_df["water_required_litres_for_area"] = water_litres
        result_df["flow_required_for_one_hour_lph"] = required_flow_lph
        result_df["installed_emitter_capacity_lph"] = installed_capacity_lph
        result_df["estimated_runtime_hours_at_installed_capacity"] = runtime_hours
        result_df["random_forest_estimated_gross_depth_mm"] = rf_depth_mm
        st.download_button(
            "Download this calculation as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="cotton_drip_irrigation_estimate.csv",
            mime="text/csv",
        )

st.divider()
st.header("4. Training scenarios and data preview")
st.caption(f"Current training source: {dataset_name}. The app trains the Random Forest regressor on {len(trained_df)} labelled rows.")
with st.expander("View training data rows", expanded=False):
    st.dataframe(trained_df, use_container_width=True, hide_index=True)
st.caption(
    "Research requirement: cite each source, align measurements to the same site/date/soil depth where possible, document all unit conversions, "
    "and use measured irrigation-depth labels. A soil-moisture-only paper cannot train a supervised irrigation-volume regressor by itself."
)

st.markdown("""
---
**Data sources to start your literature review**

- [Central Ground Water Board: Jalgaon district ground-water information](https://www.cgwb.gov.in/sites/default/files/2022-11/jalgaon_f.pdf) — describes local soil/formation zones including deep and medium black soils, Tapi alluvial black clay, loamy and sandy soils.
- [Maharashtra cotton drip-fertigation field study](https://pub.isa-india.in/index.php/ija/article/view/5660) — a Bt-cotton field experiment in Rahuri, Maharashtra; useful background, but not automatically a Khandesh row-level sensor dataset.
- [FAO: Soil and water](https://www.fao.org/4/r4082e/r4082e03.htm) — distinguishes saturation, field capacity and plant-available water.
- [FAO: Irrigation water needs](https://www.fao.org/4/S2022E/s2022e08.htm) — explains crop water need, effective rainfall and irrigation need.
""")
