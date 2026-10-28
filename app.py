"""
Cotton root-zone water calculator
=================================
Crop: cotton    Soil: black soil    Stage: flowering (Sept-Nov)

Question: how much water does it take to bring the soil around the roots
(the root zone) up to SATURATION?

    root-zone soil volume  V  = volume of the root-zone shape (from width and depth)
    water to add (litres)     = (saturation % - current moisture %) / 100 x V

Every soil-water percentage here is VOLUMETRIC: the share of the soil's volume
that is water.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import math

import streamlit as st

PARTICLE_DENSITY = 2.65  # g/cm3, standard value for mineral soil

SHAPES = {
    "Cylinder (width = diameter)": "cylinder",
    "Square block (width × width)": "block",
}
SAT_BY_PERCENT = "Saturation water content (%)"
SAT_BY_DENSITY = "Bulk density (g/cm³)"


# ----------------------------------------------------------------------------
# Calculation (no Streamlit in here, so it can be tested on its own)
# ----------------------------------------------------------------------------
def root_zone_volume_l(width_cm: float, depth_cm: float, shape: str) -> float:
    """Volume of soil in the root zone, in litres (1 L = 1000 cm³)."""
    if width_cm <= 0 or depth_cm <= 0:
        raise ValueError("Width and root depth must both be greater than zero.")
    if shape == "cylinder":
        volume_cm3 = math.pi * (width_cm / 2.0) ** 2 * depth_cm
    elif shape == "block":
        volume_cm3 = width_cm * width_cm * depth_cm
    else:
        raise ValueError(f"Unknown shape: {shape!r}")
    return volume_cm3 / 1000.0


def saturation_from_bulk_density(bulk_density_g_cm3: float) -> float:
    """Saturation water content (% of soil volume) = total porosity."""
    if not 0.5 <= bulk_density_g_cm3 < PARTICLE_DENSITY:
        raise ValueError(f"Bulk density must be between 0.5 and {PARTICLE_DENSITY} g/cm³.")
    return (1.0 - bulk_density_g_cm3 / PARTICLE_DENSITY) * 100.0


def water_needed_l(volume_l: float, target_pct: float, current_pct: float) -> float:
    """Litres needed to raise soil water from current_pct to target_pct (both % of soil volume)."""
    if current_pct < 0 or target_pct < 0:
        raise ValueError("Soil-water percentages cannot be negative.")
    return max(0.0, (target_pct - current_pct) / 100.0 * volume_l)


def calculate(
    width_cm: float,
    depth_cm: float,
    shape: str,
    saturation_pct: float,
    current_pct: float,
    field_capacity_pct: float | None = None,
    plants: int = 1,
) -> dict:
    """All results in litres, per plant unless the name says total."""
    if not 0 < saturation_pct <= 100:
        raise ValueError("Saturation water content must be between 0 and 100 %.")
    if not 0 <= current_pct <= 100:
        raise ValueError("Current soil moisture must be between 0 and 100 %.")
    if plants < 1:
        raise ValueError("Number of plants must be at least 1.")

    volume = root_zone_volume_l(width_cm, depth_cm, shape)
    to_saturation = water_needed_l(volume, saturation_pct, current_pct)

    result = {
        "soil_volume_l": volume,
        "water_now_l": current_pct / 100.0 * volume,
        "water_at_saturation_l": saturation_pct / 100.0 * volume,
        "to_saturation_l": to_saturation,
        "to_saturation_total_l": to_saturation * plants,
        "to_field_capacity_l": None,
        "above_field_capacity_l": None,
    }
    if field_capacity_pct is not None and 0 < field_capacity_pct < saturation_pct:
        result["to_field_capacity_l"] = water_needed_l(volume, field_capacity_pct, current_pct)
        result["above_field_capacity_l"] = (saturation_pct - field_capacity_pct) / 100.0 * volume
    return result


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Cotton Root-Zone Water Calculator", page_icon="🌱")
    st.title("🌱 Cotton Root-Zone Water Calculator")
    st.caption(
        "Cotton · black soil · flowering stage (Sept–Nov) · "
        "water needed to saturate the soil around the roots"
    )

    st.subheader("1. Root zone (the soil just around the roots)")
    col_w, col_d = st.columns(2)
    width_cm = col_w.number_input(
        "Root-zone width (cm)", min_value=1.0, max_value=300.0, value=47.0, step=1.0, key="width_cm"
    )
    depth_cm = col_d.number_input(
        "Root depth (cm)", min_value=1.0, max_value=300.0, value=25.0, step=1.0, key="depth_cm"
    )
    shape_label = st.radio("Shape of the root zone", list(SHAPES), horizontal=True, key="shape")
    plants = st.number_input(
        "Number of plants", min_value=1, max_value=1_000_000, value=1, step=1, key="plants"
    )

    st.subheader("2. Black soil")
    sat_mode = st.radio(
        "Give saturation as", [SAT_BY_PERCENT, SAT_BY_DENSITY], horizontal=True, key="sat_mode"
    )
    if sat_mode == SAT_BY_PERCENT:
        saturation_pct = st.number_input(
            "Saturation water content (% of soil volume)",
            min_value=20.0, max_value=80.0, value=47.0, step=0.5, key="sat_pct",
        )
    else:
        bulk_density = st.number_input(
            "Bulk density (g/cm³)", min_value=0.8, max_value=2.0, value=1.40, step=0.01, key="bulk_density"
        )
        saturation_pct = saturation_from_bulk_density(bulk_density)
        st.caption(
            f"Saturation = (1 − {bulk_density:.2f} ÷ {PARTICLE_DENSITY}) × 100 "
            f"= {saturation_pct:.1f} % of soil volume"
        )
    current_pct = st.number_input(
        "Current soil moisture (% of soil volume)",
        min_value=0.0, max_value=80.0, value=30.0, step=0.5, key="current_pct",
    )
    fc_pct = st.number_input(
        "Field capacity (% of soil volume), used only for the comparison below",
        min_value=5.0, max_value=80.0, value=36.0, step=0.5, key="fc_pct",
    )

    if current_pct > saturation_pct:
        st.error(
            f"Current moisture ({current_pct:.1f} %) is above saturation ({saturation_pct:.1f} %), "
            "which is not possible. Check the units (it must be % of soil volume) or the sensor calibration."
        )
        return
    try:
        res = calculate(
            width_cm, depth_cm, SHAPES[shape_label], saturation_pct, current_pct, fc_pct, int(plants)
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.divider()
    st.subheader("Result")
    col_a, col_b = st.columns(2)
    col_a.metric(
        "Water needed to reach saturation",
        f"{res['to_saturation_l']:.2f} L",
        help="Per plant: the water that fits into the empty pores of the root-zone soil.",
    )
    col_b.metric("Root-zone soil volume", f"{res['soil_volume_l']:.2f} L")
    st.caption(f"= {res['to_saturation_l'] * 1000:,.0f} mL per plant")
    if int(plants) > 1:
        st.metric(f"Total for {int(plants):,} plants", f"{res['to_saturation_total_l']:,.1f} L")

    st.markdown(
        "| Quantity (per plant) | Litres |\n"
        "|---|---|\n"
        f"| Soil in the root zone | {res['soil_volume_l']:.2f} |\n"
        f"| Water already in the soil ({current_pct:.1f} %) | {res['water_now_l']:.2f} |\n"
        f"| Water the soil holds at saturation ({saturation_pct:.1f} %) | {res['water_at_saturation_l']:.2f} |\n"
        f"| **Water to add** | **{res['to_saturation_l']:.2f}** |"
    )

    if res["to_field_capacity_l"] is not None:
        st.info(
            f"**For comparison:** bringing the same soil only to field capacity ({fc_pct:.1f} %) "
            f"needs {res['to_field_capacity_l']:.2f} L. Of the water the soil holds at saturation, "
            f"about {res['above_field_capacity_l']:.2f} L sits above field capacity and drains away. "
            "Black soil drains slowly, so a saturated root zone can stay waterlogged, "
            "which cotton tolerates poorly."
        )
    else:
        st.warning("Field capacity must be lower than saturation, so the comparison is hidden.")

    with st.expander("How this is calculated"):
        st.markdown(
            "1. **Root-zone soil volume**: cylinder = π × (width ÷ 2)² × depth; "
            "square block = width × width × depth. Centimetres give cm³, and 1000 cm³ = 1 litre.\n"
            "2. **Water in the soil now** = current moisture % ÷ 100 × soil volume.\n"
            "3. **Water at saturation** = saturation % ÷ 100 × soil volume.\n"
            "4. **Water to add** = (saturation % − current %) ÷ 100 × soil volume, never below zero.\n\n"
            "**Check these before trusting the number**\n\n"
            "- Moisture and saturation must be *volumetric* (% of soil volume). "
            "A reading in % of dry-soil weight becomes volumetric when you multiply by bulk density.\n"
            "- 47 % saturation is a textbook figure for clay, not a measurement of your field. "
            "To get your own, measure bulk density: saturation = (1 − bulk density ÷ 2.65) × 100.\n"
            "- The result is the water that fits into the pores of the block of soil you described. "
            "In a real field some water spreads sideways and downward past that block."
        )


if __name__ == "__main__":
    main()
