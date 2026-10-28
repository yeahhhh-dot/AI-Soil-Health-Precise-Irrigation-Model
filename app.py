"""Cotton root-zone water calculator.

Fixed study context: cotton, black soil, flowering stage (September-November).
The app estimates water needed to reach a selected percentage (95–100%) of
saturation in the specified root-zone volume. It is a simplified calculator,
not a field-validated irrigation prescription.

Run with: streamlit run app.py
"""
from __future__ import annotations

import math
import streamlit as st

PARTICLE_DENSITY_G_CM3 = 2.65  # assumed mineral-particle density
MAX_SATURATION_VWC_PCT = 80.0  # interface sanity bound for mineral-soil VWC

SHAPES = {
    "Cylinder (width is diameter)": "cylinder",
    "Square block (width × width)": "block",
}
SAT_BY_VWC = "Enter saturation water content (% VWC)"
SAT_BY_BULK_DENSITY = "Estimate from bulk density (g/cm³)"


def root_zone_volume_l(width_cm: float, depth_cm: float, shape: str) -> float:
    """Return the geometric soil volume per plant in litres."""
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
    """Estimate saturated VWC (%) from porosity and mineral particle density."""
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
    """Calculate target and water volumes per plant and for all plants.

    saturation_vwc_pct is the soil's actual water content at full saturation
    (% of total soil volume). target_saturation_pct is the desired percentage
    of that saturation level (95–100%), not a VWC percentage itself.

    Assumes each plant has an identical, non-overlapping root-zone volume.
    """
    if not 0 < saturation_vwc_pct <= MAX_SATURATION_VWC_PCT:
        raise ValueError(
            f"Saturation water content must be above 0 and no more than "
            f"{MAX_SATURATION_VWC_PCT:.0f}% VWC in this calculator."
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
        raise ValueError("Target must be between 95% and 100% of saturation.")
    if int(plants) != plants or plants < 1:
        raise ValueError("Number of plants must be a whole number of at least 1.")

    volume_l = root_zone_volume_l(width_cm, depth_cm, shape)
    target_vwc_pct = saturation_vwc_pct * target_saturation_pct / 100.0

    water_now_per_plant_l = current_vwc_pct / 100.0 * volume_l
    water_at_saturation_per_plant_l = saturation_vwc_pct / 100.0 * volume_l
    water_at_target_per_plant_l = target_vwc_pct / 100.0 * volume_l
    water_needed_per_plant_l = max(
        0.0, (target_vwc_pct - current_vwc_pct) / 100.0 * volume_l
    )
    plant_count = int(plants)

    return {
        "shape": shape,
        "soil_volume_per_plant_l": volume_l,
        "saturation_vwc_pct": saturation_vwc_pct,
        "target_saturation_pct": target_saturation_pct,
        "target_vwc_pct": target_vwc_pct,
        "current_vwc_pct": current_vwc_pct,
        "water_now_per_plant_l": water_now_per_plant_l,
        "water_at_saturation_per_plant_l": water_at_saturation_per_plant_l,
        "water_at_target_per_plant_l": water_at_target_per_plant_l,
        "water_needed_per_plant_l": water_needed_per_plant_l,
        "plants": plant_count,
        "soil_volume_total_l": volume_l * plant_count,
        "water_now_total_l": water_now_per_plant_l * plant_count,
        "water_at_saturation_total_l": water_at_saturation_per_plant_l * plant_count,
        "water_at_target_total_l": water_at_target_per_plant_l * plant_count,
        "water_needed_total_l": water_needed_per_plant_l * plant_count,
    }


def show_results(result: dict[str, float | int | str]) -> None:
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
        f"Target: {target_sat:.0f}% of saturation = {target_vwc:.2f}% VWC "
        f"(full saturation is {saturation_vwc:.2f}% VWC)."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric(f"TOTAL water for {plants:,} plant(s)", f"{water_total:,.2f} L")
    col2.metric("Water per plant", f"{water_per_plant:,.3f} L")
    col3.metric("Root-zone volume per plant", f"{float(result['soil_volume_per_plant_l']):,.2f} L")

    if current_vwc >= target_vwc:
        st.success(
            "The current moisture is already at or above the selected target. "
            "The calculated additional water requirement is 0 L."
        )
    else:
        st.info(
            f"To reach the selected target, the simplified calculation estimates "
            f"{water_per_plant:.3f} L per plant, or {water_total:,.2f} L for "
            f"{plants:,} plant(s)."
        )

    per_plant_volume = float(result["soil_volume_per_plant_l"])
    total_volume = float(result["soil_volume_total_l"])
    lines = [
        "| Quantity | Per plant | Total for all plants |",
        "|---|---:|---:|",
        f"| Root-zone soil volume | {per_plant_volume:.3f} L | {total_volume:.3f} L |",
        f"| Water currently in root zone | {float(result['water_now_per_plant_l']):.3f} L | {float(result['water_now_total_l']):.3f} L |",
        f"| Water at selected target | {float(result['water_at_target_per_plant_l']):.3f} L | {float(result['water_at_target_total_l']):.3f} L |",
        f"| Water at full saturation | {float(result['water_at_saturation_per_plant_l']):.3f} L | {float(result['water_at_saturation_total_l']):.3f} L |",
        f"| **Additional water needed** | **{water_per_plant:.3f} L** | **{water_total:.3f} L** |",
    ]
    st.markdown("\n".join(lines))

    st.warning(
        "This total multiplies the per-plant result by the plant count, assuming "
        "identical, non-overlapping root zones. In a real field, neighbouring root "
        "zones can overlap, so multiplying may overestimate total water."
    )


def main() -> None:
    st.set_page_config(page_title="Cotton Root-Zone Water Calculator", page_icon="🌱", layout="centered")
    st.title("🌱 Cotton Root-Zone Water Calculator")
    st.caption("Cotton · black soil · flowering stage (September–November)")
    st.write(
        "Calculate the water needed to bring the specified soil volume around the "
        "roots to **95–100% of its saturation level**. Measurements are volumetric "
        "water content (VWC), unless stated otherwise."
    )

    with st.expander("What do the two percentages mean?", expanded=True):
        st.markdown(
            "- **Saturation water content (% VWC):** the actual percentage of the "
            "whole soil volume that would be water when the soil pores are full. "
            "This is a soil property, usually estimated from porosity or measured.\n"
            "- **Target (% of saturation):** how close you want to get to that "
            "full-saturation level. This separate control is restricted to **95–100%**.\n"
            "- Example: if saturation is 47% VWC, a target of 95% of saturation "
            "means 44.65% VWC—not 95% VWC.\n"
            "- The 80% maximum applies only to the entered saturation VWC. It is a "
            "sanity bound for this mineral-soil calculator, not the target. A VWC of "
            "95–100% would leave practically no soil volume for solid particles."
        )

    st.subheader("1. Root-zone dimensions and plant count")
    st.caption("Width and depth describe the root-zone volume for ONE plant.")

    error_message = None
    submitted = False
    submitted_values = None

    with st.form("water_calculation_form", clear_on_submit=False, border=True):
        col_w, col_d = st.columns(2)
        width_cm = col_w.number_input(
            "Root-zone width (cm)", min_value=1.0, max_value=300.0,
            value=47.0, step=1.0, key="width_cm"
        )
        depth_cm = col_d.number_input(
            "Root depth (cm)", min_value=1.0, max_value=300.0,
            value=25.0, step=1.0, key="depth_cm"
        )
        shape_label = st.radio(
            "Root-zone shape", list(SHAPES), horizontal=True, key="shape_label",
            help="Cylinder uses width as its diameter. Square block uses width × width."
        )
        plants = st.number_input(
            "Number of plants", min_value=1, max_value=1_000_000,
            value=1, step=1, key="plants"
        )

        st.subheader("2. Black-soil water measurements")
        sat_mode = st.radio(
            "How will you set full-saturation water content?",
            [SAT_BY_VWC, SAT_BY_BULK_DENSITY], horizontal=True, key="sat_mode"
        )
        if sat_mode == SAT_BY_VWC:
            saturation_vwc_pct = st.number_input(
                "Full-saturation water content (% VWC)",
                min_value=1.0, max_value=MAX_SATURATION_VWC_PCT,
                value=47.0, step=0.5, key="saturation_vwc_pct",
                help="This is the soil's full-saturation VWC, not the target percent of saturation."
            )
            bulk_density = None
        else:
            bulk_density = st.number_input(
                "Bulk density (g/cm³)", min_value=0.5, max_value=2.5,
                value=1.40, step=0.01, key="bulk_density",
                help="Used to estimate porosity assuming mineral particle density of 2.65 g/cm³."
            )
            saturation_vwc_pct = saturation_from_bulk_density(float(bulk_density))
            st.caption(
                f"Estimated full-saturation VWC = (1 − {bulk_density:.2f} ÷ "
                f"{PARTICLE_DENSITY_G_CM3:.2f}) × 100 = {saturation_vwc_pct:.2f}% VWC. "
                "This is an estimate; measured porosity is preferable."
            )

        target_saturation_pct = st.slider(
            "Target saturation (% of full saturation)",
            min_value=95, max_value=100, value=95, step=1,
            key="target_saturation_pct",
            help="95 means 95% of the saturation VWC; 100 means full saturation."
        )
        st.caption(
            f"Current target from the settings above: "
            f"{saturation_vwc_pct * target_saturation_pct / 100.0:.2f}% VWC."
        )
        current_vwc_pct = st.number_input(
            "Current soil moisture (% VWC)", min_value=0.0,
            max_value=MAX_SATURATION_VWC_PCT, value=30.0, step=0.5,
            key="current_vwc_pct",
            help="Enter volumetric water content, not percent of dry soil weight."
        )
        submitted = st.form_submit_button(
            "Calculate / Check water requirement", type="primary", use_container_width=True
        )

        submitted_values = {
            "width_cm": float(width_cm),
            "depth_cm": float(depth_cm),
            "shape": SHAPES[shape_label],
            "saturation_vwc_pct": float(saturation_vwc_pct),
            "current_vwc_pct": float(current_vwc_pct),
            "target_saturation_pct": float(target_saturation_pct),
            "plants": int(plants),
            "sat_mode": sat_mode,
            "bulk_density": None if bulk_density is None else float(bulk_density),
        }

    if submitted:
        try:
            result = calculate(
                width_cm=submitted_values["width_cm"],
                depth_cm=submitted_values["depth_cm"],
                shape=submitted_values["shape"],
                saturation_vwc_pct=submitted_values["saturation_vwc_pct"],
                current_vwc_pct=submitted_values["current_vwc_pct"],
                target_saturation_pct=submitted_values["target_saturation_pct"],
                plants=submitted_values["plants"],
            )
            st.session_state["last_result"] = result
            st.session_state["last_inputs"] = submitted_values
            st.session_state.pop("last_error", None)
        except ValueError as exc:
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_inputs", None)
            st.session_state["last_error"] = str(exc)

    if "last_error" in st.session_state:
        st.error(st.session_state["last_error"])
    elif "last_result" in st.session_state:
        st.subheader("3. Calculated result")
        show_results(st.session_state["last_result"])
        st.caption("Results shown are from the last time you pressed the calculation button.")
    else:
        st.info("Adjust the measurements above, then press **Calculate / Check water requirement**.")

    with st.expander("Calculation method and assumptions"):
        st.markdown(
            "**1. Root-zone volume per plant**\n\n"
            "Cylinder: π × (width ÷ 2)² × depth. Square block: width × width × depth. "
            "Dimensions are in cm; divide cm³ by 1000 to convert to litres.\n\n"
            "**2. Convert target saturation to target VWC**\n\n"
            "Target VWC (%) = full-saturation VWC (%) × target saturation (%) ÷ 100.\n\n"
            "**3. Water needed per plant**\n\n"
            "max(0, target VWC − current VWC) ÷ 100 × root-zone volume (L).\n\n"
            "**4. Total for the selected number of plants**\n\n"
            "Water per plant × number of plants. The same multiplication is shown for "
            "the total soil volume and water stored in the root zone.\n\n"
            "**Why no field-capacity control?**\n\n"
            "Field capacity is removed from this version because the requested calculation "
            "is for 95–100% of saturation. Field capacity is a different soil-water state "
            "and is not needed in this particular formula.\n\n"
            "**Limits of the estimate**\n\n"
            "The answer assumes the stated shape and dimensions represent the soil around "
            "one plant and that the current moisture is uniform throughout that volume. "
            "Root zones may overlap in a real field, and delivering the calculated water "
            "does not guarantee that the soil can safely absorb it without drainage or ponding."
        )


if __name__ == "__main__":
    main()

