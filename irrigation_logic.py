"""Interpretable fuzzy-control + water-balance logic for a cotton irrigation demo.

The parameter values shipped with the project are explicitly illustrative classroom
placeholders, not validated values for all black/red soils in Khandesh.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Mapping

import numpy as np
import pandas as pd

SOIL_TYPES = ["Black soil", "Red soil"]
SEASONS = ["January-March", "April-June", "July-September", "October-December"]
CROP_STAGES = [
    "Establishment",
    "Vegetative growth and squaring",
    "Flowering",
    "Boll development",
    "Boll opening and maturity",
]

# Membership labels, in increasing order of deficit / sensitivity / VPD / rainfall.
DEFICIT_LABELS = ["negligible", "small", "moderate", "large", "severe"]
STAGE_LABELS = ["low", "medium", "high"]
VPD_LABELS = ["low", "medium", "high"]
SOIL_TEMP_LABELS = ["cool", "normal", "hot"]
RAIN_LABELS = ["none", "light", "useful", "heavy"]
TARGET_LABELS = ["lower", "middle", "upper"]
PRIORITY_LABELS = ["none", "low", "moderate", "high"]


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Version-compatible trapezoid integration (NumPy 1.x and 2.x)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _trapmf_scalar(x: float, a: float, b: float, c: float, d: float) -> float:
    """Membership degree for a trapezoidal fuzzy set, including shoulder sets."""
    x = float(x)
    if a == b and x <= b:
        return 1.0
    if x < a:
        return 0.0
    if a < b and x < b:
        return float((x - a) / (b - a))
    if b <= x <= c:
        return 1.0
    if c == d and x >= c:
        return 1.0
    if c < x < d:
        return float((d - x) / (d - c))
    return 0.0


def _trimf_scalar(x: float, a: float, b: float, c: float) -> float:
    """Membership degree for a triangular fuzzy set."""
    return _trapmf_scalar(x, a, b, b, c)


def _trapmf_array(x: np.ndarray, params: tuple[float, float, float, float]) -> np.ndarray:
    """Vectorized trapezoidal membership function for output aggregation."""
    a, b, c, d = params
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    if a == b:
        y[x <= b] = 1.0
    else:
        mask = (x > a) & (x < b)
        y[mask] = (x[mask] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    if c == d:
        y[x >= c] = 1.0
    else:
        mask = (x > c) & (x < d)
        y[mask] = (d - x[mask]) / (d - c)
    return np.clip(y, 0.0, 1.0)


def _trimf_array(x: np.ndarray, params: tuple[float, float, float]) -> np.ndarray:
    a, b, c = params
    return _trapmf_array(x, (a, b, b, c))


def saturation_vapour_pressure_kpa(air_temperature_c: float) -> float:
    """Tetens approximation to saturation vapour pressure (kPa)."""
    t = float(air_temperature_c)
    denominator = t + 237.3
    if denominator <= 0:
        raise ValueError("Air temperature is outside the valid range for the vapour-pressure calculation.")
    return float(0.6108 * np.exp((17.27 * t) / denominator))


def vapour_pressure_deficit_kpa(air_temperature_c: float, relative_humidity_pct: float) -> float:
    """Approximate VPD from air temperature and RH; this is not full crop ET."""
    if not 0.0 <= relative_humidity_pct <= 100.0:
        raise ValueError("Relative humidity must be between 0 and 100%.")
    es = saturation_vapour_pressure_kpa(air_temperature_c)
    return float(max(0.0, es * (1.0 - relative_humidity_pct / 100.0)))


def _fuzzify_deficit(value: float) -> dict[str, float]:
    return {
        "negligible": _trapmf_scalar(value, 0, 0, 3, 8),
        "small": _trimf_scalar(value, 4, 12, 22),
        "moderate": _trimf_scalar(value, 15, 28, 42),
        "large": _trimf_scalar(value, 35, 52, 70),
        "severe": _trapmf_scalar(value, 60, 78, 100, 100),
    }


def _fuzzify_stage(value: float) -> dict[str, float]:
    return {
        "low": _trapmf_scalar(value, 0, 0, 0.25, 0.45),
        "medium": _trimf_scalar(value, 0.25, 0.50, 0.75),
        "high": _trapmf_scalar(value, 0.55, 0.78, 1.0, 1.0),
    }


def _fuzzify_vpd(value: float) -> dict[str, float]:
    return {
        "low": _trapmf_scalar(value, 0, 0, 0.5, 1.2),
        "medium": _trimf_scalar(value, 0.7, 1.6, 2.6),
        "high": _trapmf_scalar(value, 2.0, 3.0, 6.0, 6.0),
    }


def _fuzzify_soil_temperature(value: float) -> dict[str, float]:
    return {
        "cool": _trapmf_scalar(value, -5, -5, 12, 20),
        "normal": _trimf_scalar(value, 16, 25, 32),
        "hot": _trapmf_scalar(value, 28, 36, 70, 70),
    }


def _fuzzify_rain(value: float) -> dict[str, float]:
    # Expected rain over the next 24 h influences urgency/timing and can nudge the
    # target toward the lower end of the allowed band, but is not subtracted from
    # the measured current soil-water deficit.
    return {
        "none": _trapmf_scalar(value, 0, 0, 1, 4),
        "light": _trimf_scalar(value, 2, 7, 15),
        "useful": _trimf_scalar(value, 10, 25, 40),
        "heavy": _trapmf_scalar(value, 30, 50, 100, 100),
    }


def _rule_consequents(
    deficit_label: str,
    stage_label: str,
    vpd_label: str,
    soil_temp_label: str,
    rain_label: str,
) -> tuple[str, str]:
    """Map a fuzzy antecedent combination to target band and irrigation priority.

    This is a transparent, hand-authored classroom rule base, not a rule set
    learned from 100 observations. Thresholds should be validated agronomically.
    """
    d_idx = DEFICIT_LABELS.index(deficit_label)
    s_idx = STAGE_LABELS.index(stage_label)
    v_idx = VPD_LABELS.index(vpd_label)
    temp_idx = SOIL_TEMP_LABELS.index(soil_temp_label)
    r_idx = RAIN_LABELS.index(rain_label)

    # Consequent for target = 95-100% of field capacity. Greater deficits use a
    # higher target within that narrow permitted band; high sensitivity/demand
    # can move a medium deficit toward the upper target.
    target_idx = [0, 0, 1, 2, 2][d_idx]
    if d_idx >= 1 and s_idx == 2:
        target_idx += 1
    if d_idx >= 2 and v_idx == 2:
        target_idx += 1
    if d_idx >= 2 and temp_idx == 2:
        target_idx += 1
    # If useful/heavy rain is forecast, choose a conservative lower target
    # within the allowed band for non-severe deficits. Rainfall is not deducted
    # from the water balance because it has not yet occurred.
    if d_idx <= 3 and r_idx >= 2:
        target_idx -= 1
    target_idx = int(np.clip(target_idx, 0, 2))

    # Urgency depends on deficit, crop-stage sensitivity and VPD; expected rain
    # can lower the urgency but does not change the mathematical refill volume.
    priority_idx = [0, 1, 2, 3, 3][d_idx]
    if d_idx >= 1 and s_idx == 2:
        priority_idx += 1
    if d_idx >= 2 and v_idx == 2:
        priority_idx += 1
    if d_idx >= 1 and temp_idx == 2:
        priority_idx += 1
    if r_idx == 2:
        priority_idx -= 1
    elif r_idx == 3:
        priority_idx -= 2

    if d_idx == 0:
        priority_idx = 0
    elif d_idx == 1:
        priority_idx = max(priority_idx, 1)
    elif d_idx >= 3:
        # Even with expected rain, a large/severe deficit should not be labelled
        # as trivial; it remains at least moderate priority for review.
        priority_idx = max(priority_idx, 2)
    priority_idx = int(np.clip(priority_idx, 0, 3))

    return TARGET_LABELS[target_idx], PRIORITY_LABELS[priority_idx]


def fuzzy_controller(
    deficit_pct_of_fc: float,
    stage_sensitivity: float,
    vpd_kpa: float,
    soil_temperature_c: float,
    forecast_rain_next_24h_mm: float,
) -> dict[str, float | str]:
    """Mamdani-style fuzzy inference for target (% FC) and urgency (0-100).

    Output includes a target between 95 and 100 percent of field capacity and a
    qualitative urgency score. The target is defuzzified by centroid of area.
    """
    if not 0.0 <= deficit_pct_of_fc <= 100.0:
        raise ValueError("Deficit as a percentage of field capacity must be from 0 to 100.")
    if not 0.0 <= stage_sensitivity <= 1.0:
        raise ValueError("Crop-stage sensitivity must be between 0 and 1.")
    if not 0.0 <= forecast_rain_next_24h_mm <= 100.0:
        raise ValueError("Forecast rainfall must be between 0 and 100 mm.")

    memberships = [
        _fuzzify_deficit(deficit_pct_of_fc),
        _fuzzify_stage(stage_sensitivity),
        _fuzzify_vpd(min(float(vpd_kpa), 6.0)),
        _fuzzify_soil_temperature(float(soil_temperature_c)),
        _fuzzify_rain(forecast_rain_next_24h_mm),
    ]
    label_sets = [DEFICIT_LABELS, STAGE_LABELS, VPD_LABELS, SOIL_TEMP_LABELS, RAIN_LABELS]

    target_universe = np.linspace(95.0, 100.0, 501)
    target_mfs = {
        "lower": _trapmf_array(target_universe, (95.0, 95.0, 95.6, 97.0)),
        "middle": _trimf_array(target_universe, (96.0, 97.5, 99.0)),
        "upper": _trapmf_array(target_universe, (98.0, 99.0, 100.0, 100.0)),
    }
    priority_universe = np.linspace(0.0, 100.0, 501)
    priority_mfs = {
        "none": _trapmf_array(priority_universe, (0.0, 0.0, 10.0, 25.0)),
        "low": _trimf_array(priority_universe, (10.0, 30.0, 50.0)),
        "moderate": _trimf_array(priority_universe, (35.0, 55.0, 75.0)),
        "high": _trapmf_array(priority_universe, (60.0, 80.0, 100.0, 100.0)),
    }
    target_aggregate = np.zeros_like(target_universe)
    priority_aggregate = np.zeros_like(priority_universe)

    # The Cartesian rule table is generated from the transparent consequent
    # logic above; activation is the minimum antecedent membership (Mamdani AND).
    for d_label, s_label, v_label, soil_temp_label, r_label in product(*label_sets):
        activation = min(
            memberships[0][d_label],
            memberships[1][s_label],
            memberships[2][v_label],
            memberships[3][soil_temp_label],
            memberships[4][r_label],
        )
        if activation <= 0.0:
            continue
        target_label, priority_label = _rule_consequents(
            d_label, s_label, v_label, soil_temp_label, r_label
        )
        target_aggregate = np.maximum(
            target_aggregate, np.minimum(activation, target_mfs[target_label])
        )
        priority_aggregate = np.maximum(
            priority_aggregate, np.minimum(activation, priority_mfs[priority_label])
        )

    target_area = _integrate_trapezoid(target_aggregate, target_universe)
    priority_area = _integrate_trapezoid(priority_aggregate, priority_universe)
    if target_area <= 0 or priority_area <= 0:
        # This should be rare because membership sets overlap; fail safely.
        target_pct_fc = 95.0
        priority_score = 20.0
    else:
        target_pct_fc = float(
            _integrate_trapezoid(target_universe * target_aggregate, target_universe) / target_area
        )
        priority_score = float(
            _integrate_trapezoid(priority_universe * priority_aggregate, priority_universe) / priority_area
        )

    target_pct_fc = float(np.clip(target_pct_fc, 95.0, 100.0))
    priority_score = float(np.clip(priority_score, 0.0, 100.0))
    if priority_score < 25:
        priority_label = "Low"
    elif priority_score < 50:
        priority_label = "Moderate"
    elif priority_score < 75:
        priority_label = "High"
    else:
        priority_label = "Very high"

    return {
        "target_pct_of_field_capacity": target_pct_fc,
        "priority_score": priority_score,
        "priority_label": priority_label,
    }


def load_parameter_table(path: str) -> pd.DataFrame:
    """Read soil/stage parameter table and validate required columns."""
    df = pd.read_csv(path)
    required = {
        "soil_type",
        "field_capacity_vwc_pct",
        "saturation_vwc_pct",
        "crop_stage",
        "effective_root_zone_depth_cm",
        "stage_sensitivity",
        "parameter_status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Parameter file is missing columns: " + ", ".join(missing))
    for col in [
        "field_capacity_vwc_pct",
        "saturation_vwc_pct",
        "effective_root_zone_depth_cm",
        "stage_sensitivity",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(required - {"soil_type", "crop_stage", "parameter_status"})].isna().any().any():
        raise ValueError("Parameter file contains missing/nonnumeric hydraulic or stage parameters.")
    if df.duplicated(["soil_type", "crop_stage"]).any():
        raise ValueError("Parameter file must have only one row per soil type and crop stage.")
    return df


def get_parameter_row(parameter_table: pd.DataFrame, soil_type: str, crop_stage: str) -> dict[str, Any]:
    matched = parameter_table[
        (parameter_table["soil_type"] == soil_type)
        & (parameter_table["crop_stage"] == crop_stage)
    ]
    if matched.empty:
        raise ValueError(f"No parameter row for {soil_type} / {crop_stage}. Check soil_stage_parameters.csv.")
    return matched.iloc[0].to_dict()


def evaluate_scenario(
    scenario: Mapping[str, Any],
    parameter_table: pd.DataFrame,
    *,
    parameter_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Validate and calculate a single scenario, returning transparent outputs.

    The current VWC must represent the average water content over the effective
    root-zone depth. If it is just one shallow sensor reading, this refill estimate
    can be badly biased; the app states this assumption explicitly.
    """
    row = dict(scenario)
    soil = str(row.get("soil_type", "")).strip()
    stage = str(row.get("crop_stage", "")).strip()
    if soil not in SOIL_TYPES:
        raise ValueError(f"soil_type must be one of: {', '.join(SOIL_TYPES)}")
    if stage not in CROP_STAGES:
        raise ValueError(f"crop_stage must be one of: {', '.join(CROP_STAGES)}")

    params = get_parameter_row(parameter_table, soil, stage)
    if parameter_overrides:
        params.update(parameter_overrides)

    def number(key: str, default: float | None = None) -> float:
        raw = row.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a numeric value.")
        if not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        return value

    fc = float(params["field_capacity_vwc_pct"])
    saturation = float(params["saturation_vwc_pct"])
    root_depth_cm = float(params["effective_root_zone_depth_cm"])
    sensitivity = float(params["stage_sensitivity"])
    if not (0 < fc < saturation <= 100):
        raise ValueError("Hydraulic parameters must satisfy 0 < field capacity < saturation <= 100% VWC.")
    if not (0 < root_depth_cm <= 150):
        raise ValueError("Effective root-zone depth must be between 0 and 150 cm.")
    if not (0 <= sensitivity <= 1):
        raise ValueError("stage_sensitivity must be between 0 and 1.")

    area_m2 = number("area_m2")
    moisture = number("current_soil_moisture_vwc_pct")
    soil_temp = number("soil_temperature_c")
    air_temp = number("air_temperature_c")
    humidity = number("relative_humidity_pct")
    rain_prev = number("rainfall_previous_24h_mm", 0.0)
    rain_forecast = number("forecast_rain_next_24h_mm", 0.0)
    efficiency_pct = number("irrigation_efficiency_pct", 90.0)
    soil_ph = number("soil_ph", 7.0)
    ec = number("electrical_conductivity_ds_m", 1.0)
    season = str(row.get("season", "")).strip()

    if area_m2 <= 0:
        raise ValueError("area_m2 must be greater than 0.")
    if not (0 <= moisture <= saturation):
        raise ValueError(
            f"Current VWC must be between 0 and this soil's saturation value ({saturation:.1f}%)."
        )
    if not (-5 <= soil_temp <= 70):
        raise ValueError("Soil temperature must be between -5 and 70°C for this simulator.")
    if not (-10 <= air_temp <= 60):
        raise ValueError("Air temperature must be between -10 and 60°C for this simulator.")
    if not (0 <= humidity <= 100):
        raise ValueError("Relative humidity must be between 0 and 100%.")
    if not (0 <= rain_prev <= 500):
        raise ValueError("Previous-24-hour rainfall must be between 0 and 500 mm.")
    if not (0 <= rain_forecast <= 100):
        raise ValueError("Forecast rainfall must be between 0 and 100 mm.")
    if not (1 <= efficiency_pct <= 100):
        raise ValueError("Drip application efficiency must be between 1 and 100%.")
    if not (0 <= soil_ph <= 14):
        raise ValueError("Soil pH must be between 0 and 14.")
    if ec < 0:
        raise ValueError("Electrical conductivity cannot be negative.")

    vpd = vapour_pressure_deficit_kpa(air_temp, humidity)
    deficit_pct_fc = float(max(0.0, (fc - moisture) / fc * 100.0))
    fuzzy = fuzzy_controller(deficit_pct_fc, sensitivity, vpd, soil_temp, rain_forecast)
    fuzzy_target_pct_fc = float(fuzzy["target_pct_of_field_capacity"])
    target_pct_fc = fuzzy_target_pct_fc
    target_vwc = fc * (target_pct_fc / 100.0)

    # If current measured VWC is already >= 95% FC, no refill is needed now.
    # In that case report the current in-band moisture as the effective target
    # (capped at 100% FC), rather than displaying a higher target with zero flow.
    # Forecast rainfall is not subtracted from the measured current deficit.
    minimum_target_vwc = fc * 0.95
    irrigation_needed = moisture < minimum_target_vwc
    if not irrigation_needed:
        target_pct_fc = float(np.clip((moisture / fc) * 100.0, 95.0, 100.0))
        target_vwc = fc * (target_pct_fc / 100.0)
        target_vwc_for_calc = moisture
    else:
        target_vwc_for_calc = target_vwc

    net_depth_mm = max(0.0, (target_vwc_for_calc - moisture) / 100.0 * root_depth_cm * 10.0)
    efficiency = efficiency_pct / 100.0
    gross_depth_mm = net_depth_mm / efficiency if net_depth_mm > 0 else 0.0
    net_volume_l = net_depth_mm * area_m2
    gross_volume_l = gross_depth_mm * area_m2
    required_flow_lph = gross_volume_l  # explicitly for a one-hour delivery

    priority_label = str(fuzzy["priority_label"])
    if not irrigation_needed:
        recommendation = "No additional refill is required now: measured moisture is already at or above 95% of field capacity."
        priority_label = "No refill needed"
        priority_score = 0.0
    else:
        priority_score = float(fuzzy["priority_score"])
        if rain_forecast >= 15 and priority_score < 50:
            recommendation = (
                "A moisture deficit exists, but forecast rain may justify reviewing whether to delay. "
                "The flow below is the theoretical requirement if irrigating now."
            )
        elif priority_score >= 65:
            recommendation = "High immediate priority under the illustrative fuzzy rules; verify readings and local agronomic guidance."
        elif priority_score >= 40:
            recommendation = "Moderate immediate priority under the illustrative fuzzy rules; verify local conditions."
        else:
            recommendation = "Lower immediate priority under the illustrative fuzzy rules; the volume below is still the calculated refill if irrigating now."

    return {
        "scenario_id": row.get("scenario_id", ""),
        "soil_type": soil,
        "season": season,
        "crop_stage": stage,
        "area_m2": area_m2,
        "field_capacity_vwc_pct": fc,
        "saturation_vwc_pct": saturation,
        "effective_root_zone_depth_cm": root_depth_cm,
        "current_soil_moisture_vwc_pct": moisture,
        "target_pct_of_field_capacity": target_pct_fc,
        "fuzzy_proposed_target_pct_of_field_capacity": fuzzy_target_pct_fc,
        "target_moisture_vwc_pct": target_vwc,
        "soil_temperature_c": soil_temp,
        "air_temperature_c": air_temp,
        "relative_humidity_pct": humidity,
        "vapour_pressure_deficit_kpa": vpd,
        "rainfall_previous_24h_mm": rain_prev,
        "forecast_rain_next_24h_mm": rain_forecast,
        "soil_ph": soil_ph,
        "electrical_conductivity_ds_m": ec,
        "irrigation_efficiency_pct": efficiency_pct,
        "deficit_pct_of_field_capacity": deficit_pct_fc,
        "irrigation_needed_now": irrigation_needed,
        "fuzzy_priority_score": priority_score,
        "fuzzy_priority": priority_label,
        "net_refill_depth_mm": net_depth_mm,
        "gross_applied_depth_mm": gross_depth_mm,
        "net_water_volume_l": net_volume_l,
        "gross_water_volume_l": gross_volume_l,
        "required_flow_for_one_hour_lph": required_flow_lph,
        "recommendation": recommendation,
        "parameter_status": params.get("parameter_status", "Status not supplied"),
    }
