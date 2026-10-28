"""Cotton root-zone water calculator with fixed study context and scenario toggles.

The calculator estimates the water needed to raise the specified root-zone
volume from current volumetric water content (VWC) to 95–100% of its assumed
full-saturation VWC. Temperature and pH controls are scenario/context readings;
they do not change the instantaneous soil-water-deficit equation.

Run locally with: streamlit run app.py
"""
from __future__ import annotations

import math
import streamlit as st

PARTICLE_DENSITY_G_CM3 = 2.65  # assumed mineral-particle density
MAX_SATURATION_VWC_PCT = 80.0  # sanity bound, not the target saturation percentage

# Fixed project conditions requested by the user/professor.
FIXED_CROP = "Cotton"
FIXED_SOIL = "Black soil"
FIXED_STAGE = "Flowering stage"
FIXED_PERIOD = "September–November"
FIXED_REGION = "Khandesh, north-western Maharashtra"
FIXED_RAINFALL_MM = 0.0

SHAPES = {
    "Cylinder (width is diameter)": "cylinder",
    "Square block (width × width)": "block",
}


def root_zone_volume_l(width_cm: float, depth_cm: float, shape: str) -> float:
    """Return geometric root-zone soil volume for one plant, in litres."""
    if width_cm <= 0 or depth_cm <= 0:
        raise ValueError("Root-zone width and depth must both be greater than zero.")
    if shape == "cylinder":
        volume_cm3 = math.pi * (width_cm / 2.0) ** 2 * depth_cm
    elif shape == "block":
        volume_cm3 = width_cm * width_cm * depth_cm
    else:
        raise ValueError(f"Unknown root-zone shape: {shape!r}")
    return volume_cm3 / 1000.0  # 1000 cm³ = 1 litre


def saturation_from_bulk_density(bulk_density_g_cm3: float) -> float:
    """Estimate saturated VWC (%) from porosity assuming particle density 2.65 g/cm³."""
    if not 0.5 <= bulk_density_g_cm3 < PARTICLE_DENSITY_G_CM3:
        raise ValueError(
            f"Bulk density must be at least 0.5 and less than "
            f"{PARTICLE_DENSITY_G_CM3:.2f} g/cm³."
        )
    return (1.0 - bulk_density_g_cm3 / PARTICLE_DENSITY_G_CM3) * 100.0


def calculate(
    width_cm: float,
    depth_cm: float,
    shape: str,
    saturation_vwc_pct: float,
    current_vwc_pct: float,
    target_saturation_pct: float,
    plants: int = 1,
) -> dict[str, float | int | str]:
    """Calculate water volume per plant and total for identical plant root zones.

    saturation_vwc_pct: water content at full saturation, as % of total soil volume.
    target_saturation_pct: target as a % of the full-saturation VWC (95–100%).
    current_vwc_pct: measured current volumetric water content (% VWC).

    Assumption: each plant has an identical, non-overlapping root-zone volume.
    """
    if not 0 < saturation_vwc_pct <= MAX_SATURATION_VWC_PCT:
        raise ValueError(
            f"Saturation water content must be above 0 and no more than "
            f"{MAX_SATURATION_VWC_PCT:.0f}% VWC in this calculator. This is "
            "a plausibility bound for mineral soil, not the target percentage."
        )
    if not 0 <= current_vwc_pct <= MAX_SATURATION_VWC_PCT:
        raise ValueError(
            f"Current soil moisture must be between 0 and "
            f"{MAX_SATURATION_VWC_PCT:.0f}% VWC."
        )
    if current_vwc_pct > saturation_vwc_pct:
        raise ValueError(
            f"Current moisture ({current_vwc_pct:.2f}% VWC) cannot exceed the "
            f"entered saturation water content ({saturation_vwc_pct:.2f}% VWC). "
            "Check the units, sensor, or saturation value."
        )
    if not 95 <= target_saturation_pct <= 100:
        raise ValueError("Target must be between 95% and 100% of full saturation.")
    if int(plants) != plants or plants < 1:
        raise ValueError("Number of plants must be a whole number of at least 1.")

    soil_volume_l = root_zone_volume_l(width_cm, depth_cm, shape)
    target_vwc_pct = saturation_vwc_pct * target_saturation_pct / 100.0
    water_now_per_plant_l = current_vwc_pct / 100.0 * soil_volume_l
    water_at_saturation_per_plant_l = saturation_vwc_pct / 100.0 * soil_volume_l
    water_at_target_per_plant_l = target_vwc_pct / 100.0 * soil_volume_l
    water_needed_per_plant_l = max(
        0.0, (target_vwc_pct - current_vwc_pct) / 100.0 * soil_volume_l
    )
    plant_count = int(plants)

    return {
        "shape": shape,
        "soil_volume_per_plant_l": soil_volume_l,
        "saturation_vwc_pct": saturation_vwc_pct,
        "target_saturation_pct": target_saturation_pct,
        "target_vwc_pct": target_vwc_pct,
        "current_vwc_pct": current_vwc_pct,
        "water_now_per_plant_l": water_now_per_plant_l,
        "water_at_saturation_per_plant_l": water_at_saturation_per_plant_l,
        "water_at_target_per_plant_l": water_at_target_per_plant_l,
        "water_needed_per_plant_l": water_needed_per_plant_l,
        "plants": plant_count,
        "soil_volume_total_l": soil_volume_l * plant_count,
        "water_now_total_l": water_now_per_plant_l * plant_count,
        "water_at_saturation_total_l": water_at_saturation_per_plant_l * plant_count,
        "water_at_target_total_l": water_at_target_per_plant_l * plant_count,
        "water_needed_total_l": water_needed_per_plant_l * plant_count,
    }


def show_results(
    result: dict[str, float | int | str],
    air_temp_c: float,
    soil_temp_c: float,
    soil_ph: float,
) -> None:
    plants = int(result["plants"])
    water_total = float(result["water_needed_total_l"])
    water_per_plant = float(result["water_needed_per_plant_l"])
    target_vwc = float(result["target_vwc_pct"])
    target_sat = float(result["target_saturation_pct"])
    saturation_vwc = float(result["saturation_vwc_pct"])
    current_vwc = float(result["current_vwc_pct"])

    st.divider()
    st.subheader("Result")
    st.caption(
        f"Target: {target_sat:.0f}% of full saturation = {target_vwc:.2f}% VWC; "
        f"full-saturation water content = {saturation_vwc:.2f}% VWC."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric(f"TOTAL water for {plants:,} plant(s)", f"{water_total:,.2f} L")
    col2.metric("Water per plant", f"{water_per_plant:,.3f} L")
    col3.metric(
        "Root-zone volume per plant",
        f"{float(result['soil_volume_per_plant_l']):,.2f} L",
    )

    if current_vwc >= target_vwc:
        st.success(
            "Current moisture is already at or above the selected target. "
            "The calculated additional water requirement is 0 L."
        )
    else:
        st.info(
            f"Estimated water to add: {water_per_plant:.3f} L per plant, "
            f"or {water_total:,.2f} L for {plants:,} plant(s), under the assumptions shown."
        )

    st.subheader("Environmental scenario used")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Air temperature", f"{air_temp_c:.1f} °C")
    t2.metric("Soil temperature", f"{soil_temp_c:.1f} °C")
    t3.metric("Soil pH", f"{soil_ph:.1f}")
    t4.metric("Rainfall", f"{FIXED_RAINFALL_MM:.1f} mm")
    st.caption(
        "Air temperature, soil temperature, and pH are displayed as scenario readings. "
        "They do not change this immediate fill-volume result: with the same measured "
        "current moisture, target saturation, and root-zone volume, the water deficit is "
        "the same. A model of future water loss/evapotranspiration would be needed for "
        "those environmental readings to change irrigation timing."
    )

    per_plant_volume = float(result["soil_volume_per_plant_l"])
    total_volume = float(result["soil_volume_total_l"])
    st.markdown(
        "| Quantity | Per plant | Total for all plants |\n"
        "|---|---:|---:|\n"
        f"| Root-zone soil volume | {per_plant_volume:.3f} L | {total_volume:.3f} L |\n"
        f"| Water currently in root zone ({current_vwc:.1f}% VWC) | {float(result['water_now_per_plant_l']):.3f} L | {float(result['water_now_total_l']):.3f} L |\n"
        f"| Water at selected target | {float(result['water_at_target_per_plant_l']):.3f} L | {float(result['water_at_target_total_l']):.3f} L |\n"
        f"| Water at full saturation | {float(result['water_at_saturation_per_plant_l']):.3f} L | {float(result['water_at_saturation_total_l']):.3f} L |\n"
        f"| **Additional water needed** | **{water_per_plant:.3f} L** | **{water_total:.3f} L** |"
    )
    st.warning(
        "The total assumes each plant has an identical, non-overlapping root zone. "
        "Actual root zones may overlap, and this geometric estimate does not account "
        "for sideways/downward water movement or whether the soil can safely absorb "
        "the full amount at once. Saturating cotton's root zone is a classroom target, "
        "not a validated field irrigation recommendation."
    )


def main() -> None:
    st.set_page_config(
        page_title="Cotton Root-Zone Water Calculator",
        page_icon="🌱",
        layout="wide",
    )
    st.title("🌱 Cotton Root-Zone Water Calculator")
    st.caption("Water needed to bring the defined soil volume around cotton roots to 95–100% of its saturation level.")

    st.subheader("Fixed study conditions")
    fixed = st.columns(3)
    fixed[0].info(f"**Crop**\n\n{FIXED_CROP}")
    fixed[1].info(f"**Soil**\n\n{FIXED_SOIL}")
    fixed[2].info(f"**Growth stage**\n\n{FIXED_STAGE}")
    fixed2 = st.columns(3)
    fixed2[0].info(f"**Period**\n\n{FIXED_PERIOD}")
    fixed2[1].info(f"**Rainfall**\n\nNo rain ({FIXED_RAINFALL_MM:.0f} mm)")
    fixed2[2].info(f"**Region**\n\n{FIXED_REGION}")

    with st.expander("What do the saturation percentages mean?", expanded=False):
        st.markdown(
            "- **Full-saturation water content (% VWC):** the actual percentage of total soil volume occupied by water when the soil pores are full.\n"
            "- **Target (% of saturation):** how close you want to get to full saturation; choose 95–100%.\n"
            "- Example: if the soil's full-saturation value is 47% VWC, a target of 95% is 44.65% VWC—not 95% VWC.\n"
            "- The 80% maximum is a broad plausibility bound for this mineral-soil calculator, not the target. Use measured data for the actual black-soil profile."
        )

    with st.form("water_calculation_form", clear_on_submit=False, border=True):
        st.subheader("1. Root-zone dimensions and plant count")
        st.caption("Width and depth describe the modelled root-zone volume for ONE plant.")
        col_w, col_d = st.columns(2)
        width_cm = col_w.number_input(
            "Root-zone width (cm)", min_value=1.0, max_value=300.0,
            value=47.0, step=1.0, key="width_cm",
        )
        depth_cm = col_d.number_input(
            "Root depth (cm)", min_value=1.0, max_value=300.0,
            value=25.0, step=1.0, key="depth_cm",
        )
        shape_label = st.radio(
            "Root-zone shape", list(SHAPES), horizontal=True, key="shape_label",
            help="Cylinder uses width as its diameter. Square block uses width × width.",
        )
        plants = st.number_input(
            "Number of plants", min_value=1, max_value=1_000_000,
            value=1, step=1, key="plants",
        )

        st.subheader("2. Soil-moisture readings")
        sat_mode = st.radio(
            "How will you enter full-saturation water content?",
            ["Enter saturation water content (% VWC)", "Estimate from bulk density (g/cm³)"],
            horizontal=True, key="sat_mode",
        )
        if sat_mode == "Enter saturation water content (% VWC)":
            saturation_vwc_pct = st.number_input(
                "Full-saturation water content (% VWC)",
                min_value=1.0, max_value=MAX_SATURATION_VWC_PCT,
                value=47.0, step=0.5, key="saturation_vwc_pct",
                help="The soil's actual moisture content at full saturation, not the target percent.",
            )
        else:
            bulk_density = st.number_input(
                "Bulk density (g/cm³)", min_value=0.5, max_value=2.5,
                value=1.40, step=0.01, key="bulk_density",
                help="Estimates porosity using an assumed mineral particle density of 2.65 g/cm³.",
            )
            saturation_vwc_pct = saturation_from_bulk_density(float(bulk_density))
            st.caption(
                f"Estimated full-saturation VWC = (1 − {bulk_density:.2f} ÷ "
                f"{PARTICLE_DENSITY_G_CM3:.2f}) × 100 = {saturation_vwc_pct:.2f}% VWC. "
                "A measured porosity value is preferable."
            )

        target_saturation_pct = st.slider(
            "Target saturation (%)", min_value=95, max_value=100, value=95,
            step=1, key="target_saturation_pct",
            help="This is a percentage of full-saturation VWC, not a VWC reading itself.",
        )
        st.caption(
            f"Target moisture for this scenario: "
            f"{saturation_vwc_pct * target_saturation_pct / 100.0:.2f}% VWC."
        )
        current_vwc_pct = st.number_input(
            "Current soil moisture (% VWC)", min_value=0.0,
            max_value=MAX_SATURATION_VWC_PCT, value=30.0, step=0.5,
            key="current_vwc_pct",
            help="Enter volumetric water content, not percent of dry-soil weight.",
        )

        st.subheader("3. Environmental toggle controls")
        st.caption(
            "The fixed scenario assumes no rain. These controls let you vary environmental readings "
            "for the demonstration; they do not artificially change the immediate water-deficit formula."
        )
        temp_col, soil_col, ph_col = st.columns(3)
        base_air_temp_c = temp_col.slider(
            "Base air temperature (°C)", min_value=10, max_value=45,
            value=32, step=1, key="base_air_temp_c",
        )
        hot_air_toggle = temp_col.toggle(
            "Hot-air scenario (+5°C)", value=False, key="hot_air_toggle",
            help="When on, adds 5°C to the base air temperature.",
        )
        air_temp_c = float(base_air_temp_c + (5 if hot_air_toggle else 0))
        temp_col.caption(f"Effective air temperature: **{air_temp_c:.1f}°C**")

        soil_warmer_toggle = soil_col.toggle(
            "Soil is 5°C warmer than air", value=False, key="soil_warmer_toggle",
            help="On = soil is 5°C warmer than air; off = soil is 5°C cooler than air.",
        )
        soil_temp_c = air_temp_c + 5.0 if soil_warmer_toggle else air_temp_c - 5.0
        soil_col.metric("Effective soil temperature", f"{soil_temp_c:.1f} °C")
        soil_col.caption("Toggle on = air + 5°C; toggle off = air − 5°C.")

        acidic_ph_toggle = ph_col.toggle(
            "Acidic pH scenario", value=False, key="acidic_ph_toggle",
            help="On = pH 5.5 (acidic scenario); off = pH 7.0 (near-neutral scenario).",
        )
        soil_ph = 5.5 if acidic_ph_toggle else 7.0
        ph_col.metric("Soil pH used", f"{soil_ph:.1f}")
        ph_col.caption("Toggle on = pH 5.5; toggle off = pH 7.0.")

        submitted = st.form_submit_button(
            "🧮 Calculate / Check water requirement", type="primary", use_container_width=True,
        )

    if submitted:
        try:
            result = calculate(
                float(width_cm), float(depth_cm), SHAPES[shape_label],
                float(saturation_vwc_pct), float(current_vwc_pct),
                float(target_saturation_pct), int(plants),
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        show_results(result, air_temp_c, soil_temp_c, soil_ph)


if __name__ == "__main__":
    main()


