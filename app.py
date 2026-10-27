from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

import pandas as pd
import streamlit as st

from irrigation_logic import (
    CROP_STAGES,
    SEASONS,
    SOIL_TYPES,
    evaluate_scenario,
    get_parameter_row,
    load_parameter_table,
)

APP_DIR = Path(__file__).resolve().parent
PARAMETER_FILE = APP_DIR / "soil_stage_parameters.csv"
DEMO_FILE = APP_DIR / "demo_100_conditions.csv"
TEMPLATE_FILE = APP_DIR / "data_template.csv"

st.set_page_config(
    page_title="Cotton Fuzzy Drip Irrigation Simulator",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Cotton Drip Irrigation Simulator")
st.caption("Fuzzy-logic decision support + transparent soil-water balance | Cotton · Khandesh, North Maharashtra")

st.warning(
    "CLASSROOM DEMO ONLY: The included field-capacity, saturation, root-depth and fuzzy-stage settings are illustrative placeholders, "
    "not verified measurements for a particular Khandesh farm. Replace them with locally relevant published or measured values before "
    "making research claims or real irrigation decisions. The 100 sample rows are synthetic test scenarios; fuzzy logic does not train on them."
)

try:
    parameters = load_parameter_table(str(PARAMETER_FILE))
except Exception as exc:
    st.error(f"Could not load soil_stage_parameters.csv: {exc}")
    st.stop()

with st.expander("How the model works (read this before interpreting the result)", expanded=False):
    st.markdown(
        """
        1. Select **black soil or red soil**, the season, and the cotton growth stage; enter the current conditions.
        2. The fuzzy controller uses the root-zone moisture deficit relative to field capacity, crop-stage sensitivity, approximate
        soil temperature, vapour-pressure deficit (from air temperature and relative humidity), and forecast rainfall to choose a target between
        **95% and 100% of field capacity** and estimate irrigation priority.
        3. The water-balance calculation turns the chosen target into refill depth, litres for the selected area, and the theoretical
        flow needed to deliver that volume in exactly one hour.
        4. Previous rainfall is shown as context but is **not subtracted again** from the current moisture reading. Forecast rain affects
        the fuzzy urgency signal only; it is not deducted from the theoretical refill volume if you irrigate now.

        **Key assumption:** current VWC represents the average water content over the selected effective root-zone depth. A single shallow
        sensor reading may not represent the whole root zone. The requested L/h is the flow needed to apply the calculated refill in one hour,
        not a claim that the soil or a real drip system can safely accept that rate.

        pH and electrical conductivity are displayed as soil-health context; they do not arbitrarily increase or decrease the water-volume equation.
        Season is recorded for context but does not apply an unsupported seasonal multiplier. Stage affects effective root depth and fuzzy sensitivity.
        """
    )

single_tab, batch_tab, params_tab = st.tabs(["Single scenario", "Test 100 scenarios", "Parameter table"])

with single_tab:
    st.subheader("1. Field and crop settings")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.8, 1.1])
    with c1:
        soil_type = st.selectbox("Soil type", SOIL_TYPES, index=0)
    with c2:
        season = st.selectbox("Time of year", SEASONS, index=1)
    with c3:
        crop_stage = st.selectbox("Cotton growth stage", CROP_STAGES, index=2)
    with c4:
        area_m2 = st.number_input("Field area (m²)", min_value=0.1, max_value=1_000_000.0, value=20.0, step=1.0)

    defaults = get_parameter_row(parameters, soil_type, crop_stage)

    with st.expander("2. Soil/crop parameters used for the calculation (editable)", expanded=True):
        st.info(
            "These defaults are placeholders to make the simulator runnable. Replace them with reliable measurements or published values "
            "for the selected soil profile and crop stage. Editing here affects this single-scenario calculation only; update "
            "soil_stage_parameters.csv to change the batch defaults."
        )
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            field_capacity = st.number_input(
                "Field capacity (% VWC)", min_value=1.0, max_value=60.0,
                value=float(defaults["field_capacity_vwc_pct"]), step=0.5, key=f"fc_{soil_type}_{crop_stage}"
            )
        with p2:
            saturation = st.number_input(
                "Saturation (% VWC)", min_value=2.0, max_value=80.0,
                value=float(defaults["saturation_vwc_pct"]), step=0.5, key=f"sat_{soil_type}_{crop_stage}"
            )
        with p3:
            root_depth_cm = st.number_input(
                "Effective root-zone depth (cm)", min_value=1.0, max_value=150.0,
                value=float(defaults["effective_root_zone_depth_cm"]), step=5.0, key=f"depth_{soil_type}_{crop_stage}"
            )
        with p4:
            efficiency_pct = st.number_input(
                "Drip application efficiency (%)", min_value=1.0, max_value=100.0,
                value=90.0, step=1.0, key="single_efficiency"
            )
        st.caption(f"Parameter status: {defaults.get('parameter_status', 'not specified')} · {defaults.get('parameter_note', '')}")

    st.subheader("3. Current virtual sensor readings")
    c1, c2, c3 = st.columns(3)
    with c1:
        moisture_default = min(17.0, float(saturation))
        current_moisture = st.slider(
            "Current soil moisture (% VWC)", min_value=0.0,
            max_value=float(max(1.0, saturation)), value=float(moisture_default), step=0.5,
            key=f"moisture_{soil_type}",
            help="Use volumetric water content (% of total soil volume), not an uncalibrated sensor percentage. Ideally this is root-zone-average VWC."
        )
        soil_temp = st.slider("Soil temperature (°C)", min_value=-5.0, max_value=70.0, value=28.0, step=0.5)
        soil_ph = st.slider("Soil pH (screening only)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    with c2:
        air_temp = st.slider("Air temperature (°C)", min_value=-10.0, max_value=60.0, value=32.0, step=0.5)
        humidity = st.slider("Relative humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
        ec = st.number_input("Electrical conductivity (dS/m; screening only)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
    with c3:
        rain_previous = st.number_input(
            "Rainfall in previous 24 hours (mm)", min_value=0.0, max_value=500.0, value=0.0, step=1.0,
            help="Context only; not subtracted again because current soil moisture should already reflect rain that reached the root zone."
        )
        rain_forecast = st.number_input(
            "Forecast rainfall in next 24 hours (mm)", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
            help="Used by fuzzy logic to adjust target selection within 95–100% of field capacity and to adjust urgency. It is not deducted from the theoretical refill volume if irrigating now."
        )
        st.metric("Region", "Khandesh, North Maharashtra")
        st.metric("Crop", "Cotton")

    scenario = {
        "soil_type": soil_type,
        "season": season,
        "crop_stage": crop_stage,
        "area_m2": area_m2,
        "current_soil_moisture_vwc_pct": current_moisture,
        "soil_temperature_c": soil_temp,
        "air_temperature_c": air_temp,
        "relative_humidity_pct": humidity,
        "rainfall_previous_24h_mm": rain_previous,
        "forecast_rain_next_24h_mm": rain_forecast,
        "soil_ph": soil_ph,
        "electrical_conductivity_ds_m": ec,
        "irrigation_efficiency_pct": efficiency_pct,
    }
    overrides = {
        "field_capacity_vwc_pct": field_capacity,
        "saturation_vwc_pct": saturation,
        "effective_root_zone_depth_cm": root_depth_cm,
    }
    scenario_signature = json.dumps({"scenario": scenario, "overrides": overrides}, sort_keys=True)

    st.subheader("4. Calculate irrigation requirement")
    calculate_single = st.button(
        "💧 Calculate irrigation requirement",
        type="primary",
        key="calculate_single_scenario",
        use_container_width=True,
    )

    if calculate_single:
        st.session_state["single_result_signature"] = scenario_signature
        st.session_state["single_result"] = None
        st.session_state["single_error"] = None
        if field_capacity >= saturation:
            st.session_state["single_error"] = "Invalid soil parameters: field capacity must be lower than saturation."
        elif current_moisture > saturation:
            st.session_state["single_error"] = "Current moisture is greater than the saturation value for the selected parameter set. Check the units/parameters."
        else:
            try:
                st.session_state["single_result"] = evaluate_scenario(
                    scenario, parameters, parameter_overrides=overrides
                )
            except Exception as exc:
                st.session_state["single_error"] = f"Cannot calculate this scenario: {exc}"

    saved_signature = st.session_state.get("single_result_signature")
    if saved_signature != scenario_signature:
        st.info("Change the settings above, then click **Calculate irrigation requirement** to update the result.")
    elif st.session_state.get("single_error"):
        st.error(st.session_state["single_error"])
    elif st.session_state.get("single_result") is not None:
        result = st.session_state["single_result"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Required flow for 1 hour", f"{result['required_flow_for_one_hour_lph']:,.1f} L/h")
        k2.metric("Gross water volume", f"{result['gross_water_volume_l']:,.1f} L")
        k3.metric("Gross refill depth", f"{result['gross_applied_depth_mm']:.2f} mm")
        k4.metric("Target VWC", f"{result['target_moisture_vwc_pct']:.2f}%")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Net refill depth", f"{result['net_refill_depth_mm']:.2f} mm")
        k6.metric("Net water volume", f"{result['net_water_volume_l']:,.1f} L")
        k7.metric("Target selected by fuzzy logic", f"{result['target_pct_of_field_capacity']:.2f}% of FC")
        k8.metric("Vapour-pressure deficit", f"{result['vapour_pressure_deficit_kpa']:.2f} kPa")

        if result["irrigation_needed_now"]:
            st.info(f"**Fuzzy priority: {result['fuzzy_priority']}** (score {result['fuzzy_priority_score']:.1f}/100). {result['recommendation']}")
        else:
            st.success(result["recommendation"])

        with st.expander("Calculation details"):
            st.code(
                "target VWC = field capacity × fuzzy target fraction\n"
                "net refill depth (mm) = max(0, target VWC − current VWC) / 100 × root depth (mm)\n"
                "net volume (L) = net refill depth (mm) × area (m²)\n"
                "gross volume (L) = net volume / (drip efficiency / 100)\n"
                "required flow (L/h) = gross volume (L) / 1 hour"
            )
            detail_cols = [
                "field_capacity_vwc_pct", "saturation_vwc_pct", "effective_root_zone_depth_cm",
                "current_soil_moisture_vwc_pct", "target_moisture_vwc_pct", "deficit_pct_of_field_capacity",
                "net_refill_depth_mm", "gross_applied_depth_mm", "net_water_volume_l", "gross_water_volume_l",
                "irrigation_efficiency_pct", "fuzzy_priority_score", "fuzzy_priority", "parameter_status"
            ]
            st.dataframe(pd.DataFrame([{key: result[key] for key in detail_cols}]), use_container_width=True)

        st.warning(
            "This is the theoretical flow needed to apply the estimated root-zone refill volume within one hour. "
            "If the number is large, it does not mean a drip system should actually apply that flow continuously: "
            "the field's irrigation design and soil's ability to accept water need separate engineering/agronomic review."
        )

with batch_tab:
    st.subheader("Test the logic against 100 sample observations")
    st.write(
        "The included 100 rows are synthetic input scenarios for testing the same fuzzy rules and water-balance calculation. "
        "They are **not training data**, field measurements, or evidence of real-world prediction accuracy."
    )
    try:
        demo_df = pd.read_csv(DEMO_FILE)
    except Exception as exc:
        st.error(f"Could not read demo_100_conditions.csv: {exc}")
        st.stop()

    uploaded = st.file_uploader("Optionally upload your own scenario CSV", type=["csv"], key="batch_upload")
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            dataset_label = f"Uploaded: {uploaded.name}"
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")
            st.stop()
    else:
        input_df = demo_df
        dataset_label = "Built-in synthetic 100-observation test set"

    st.caption(f"Dataset selected: {dataset_label} · Rows: {len(input_df)}")
    st.dataframe(input_df.head(12), use_container_width=True)
    st.download_button(
        "Download the 100-observation sample CSV",
        data=demo_df.to_csv(index=False).encode("utf-8"),
        file_name="demo_100_conditions.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download input CSV template",
        data=pd.read_csv(TEMPLATE_FILE).to_csv(index=False).encode("utf-8"),
        file_name="data_template.csv",
        mime="text/csv",
    )

    required_input = [
        "scenario_id", "soil_type", "season", "crop_stage", "area_m2",
        "current_soil_moisture_vwc_pct", "soil_temperature_c", "air_temperature_c",
        "relative_humidity_pct", "rainfall_previous_24h_mm", "forecast_rain_next_24h_mm",
        "irrigation_efficiency_pct", "soil_ph", "electrical_conductivity_ds_m",
    ]
    missing = [col for col in required_input if col not in input_df.columns]
    input_signature = hashlib.sha256(input_df.to_csv(index=False).encode("utf-8")).hexdigest()

    run_col, clear_col = st.columns([2, 1])
    with run_col:
        run_batch = st.button(
            "▶ Run all scenarios",
            type="primary",
            key="run_batch_scenarios",
            use_container_width=True,
        )
    with clear_col:
        clear_batch = st.button("Clear previous results", key="clear_batch_results", use_container_width=True)

    if clear_batch:
        st.session_state.pop("batch_result_df", None)
        st.session_state.pop("batch_result_signature", None)
        st.rerun()

    if run_batch:
        if missing:
            st.error("The CSV is missing these required columns: " + ", ".join(missing))
            st.session_state.pop("batch_result_df", None)
            st.session_state.pop("batch_result_signature", None)
        else:
            results: list[dict[str, Any]] = []
            with st.spinner("Running the fuzzy logic on each scenario..."):
                for _, in_row in input_df.iterrows():
                    scenario = in_row.to_dict()
                    try:
                        output = evaluate_scenario(scenario, parameters)
                        output["processing_status"] = "OK"
                        output["error_message"] = ""
                    except Exception as exc:
                        output = {
                            "scenario_id": scenario.get("scenario_id", ""),
                            "soil_type": scenario.get("soil_type", ""),
                            "season": scenario.get("season", ""),
                            "crop_stage": scenario.get("crop_stage", ""),
                            "processing_status": "ERROR",
                            "error_message": str(exc),
                        }
                    results.append(output)
            st.session_state["batch_result_df"] = pd.DataFrame(results)
            st.session_state["batch_result_signature"] = input_signature

    if missing:
        st.error("The selected CSV is missing these required columns: " + ", ".join(missing))
    elif st.session_state.get("batch_result_signature") != input_signature:
        st.info("Click **Run all scenarios** to calculate outputs for the selected dataset.")
    elif "batch_result_df" in st.session_state:
        result_df = st.session_state["batch_result_df"]
        success_n = int((result_df["processing_status"] == "OK").sum())
        error_n = len(result_df) - success_n
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows processed", f"{len(result_df)}")
        m2.metric("Successful", f"{success_n}")
        m3.metric("Rows with errors", f"{error_n}")

        display_cols = [
            "scenario_id", "soil_type", "season", "crop_stage", "current_soil_moisture_vwc_pct",
            "target_moisture_vwc_pct", "net_refill_depth_mm", "gross_water_volume_l",
            "required_flow_for_one_hour_lph", "fuzzy_priority", "irrigation_needed_now",
            "processing_status", "error_message",
        ]
        existing_display = [col for col in display_cols if col in result_df.columns]
        st.dataframe(result_df[existing_display], use_container_width=True, height=440)
        st.download_button(
            "Download all processed outputs as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="cotton_fuzzy_irrigation_results.csv",
            mime="text/csv",
        )

with params_tab:
    st.subheader("Soil and crop-stage parameter table")
    st.write(
        "This file controls the default field capacity, saturation, effective root-zone depth, and fuzzy stage-sensitivity value. "
        "Every current row is explicitly an illustrative placeholder. Replace it with defensible values before describing the output as research-based."
    )
    st.dataframe(parameters, use_container_width=True)
    st.download_button(
        "Download soil_stage_parameters.csv",
        data=parameters.to_csv(index=False).encode("utf-8"),
        file_name="soil_stage_parameters.csv",
        mime="text/csv",
    )
    st.markdown(
        "**How to update parameters:** edit the CSV while preserving the column names and one row per soil/stage combination, "
        "then commit the updated file to the same GitHub repository as `app.py`. Streamlit will redeploy from the repository."
    )

st.divider()
st.caption(
    "Model type: hand-authored Mamdani-style fuzzy inference + deterministic soil-water balance. "
    "There is no supervised model training in this version. Use for education and software testing, not direct farm operation."
)
