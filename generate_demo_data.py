"""Create the 100-row synthetic scenario file used to test the Streamlit app.

These are constructed scenarios, not collected field observations.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from irrigation_logic import CROP_STAGES, SEASONS, SOIL_TYPES

OUT = Path(__file__).resolve().parent / "demo_100_conditions.csv"
rng = np.random.default_rng(20261027)
fc_by_soil = {"Black soil": 36.0, "Red soil": 23.0}  # illustrative placeholders only
sat_by_soil = {"Black soil": 53.0, "Red soil": 44.0}  # illustrative placeholders only
rows = []

for i in range(100):
    soil = SOIL_TYPES[i % len(SOIL_TYPES)]
    season = SEASONS[(i // len(SOIL_TYPES)) % len(SEASONS)]
    stage = CROP_STAGES[(i // (len(SOIL_TYPES) * len(SEASONS))) % len(CROP_STAGES)]
    fc = fc_by_soil[soil]
    sat = sat_by_soil[soil]

    if season == "April-June":
        air_temp = rng.uniform(30, 43)
        humidity = rng.uniform(18, 60)
        rain_prev = rng.choice([0.0, rng.uniform(0, 8)], p=[0.65, 0.35])
        rain_forecast = rng.choice([0.0, rng.uniform(0, 12), rng.uniform(12, 35)], p=[0.55, 0.30, 0.15])
    elif season == "July-September":
        air_temp = rng.uniform(24, 34)
        humidity = rng.uniform(55, 96)
        rain_prev = rng.choice([0.0, rng.uniform(1, 35), rng.uniform(35, 90)], p=[0.20, 0.60, 0.20])
        rain_forecast = rng.choice([0.0, rng.uniform(1, 20), rng.uniform(20, 60)], p=[0.25, 0.45, 0.30])
    elif season == "October-December":
        air_temp = rng.uniform(20, 34)
        humidity = rng.uniform(30, 80)
        rain_prev = rng.choice([0.0, rng.uniform(0, 15)], p=[0.70, 0.30])
        rain_forecast = rng.choice([0.0, rng.uniform(0, 15)], p=[0.70, 0.30])
    else:
        air_temp = rng.uniform(20, 36)
        humidity = rng.uniform(20, 70)
        rain_prev = rng.choice([0.0, rng.uniform(0, 8)], p=[0.80, 0.20])
        rain_forecast = rng.choice([0.0, rng.uniform(0, 10)], p=[0.80, 0.20])

    # Deliberately include dry, near-target, and above-field-capacity cases.
    moisture_ratio = rng.choice(
        [rng.uniform(0.30, 0.65), rng.uniform(0.65, 0.90), rng.uniform(0.90, 1.03)],
        p=[0.40, 0.35, 0.25],
    )
    moisture = min(fc * moisture_ratio, sat - 0.5)
    root_stage_depth = {
        "Establishment": 20,
        "Vegetative growth and squaring": 40,
        "Flowering": 60,
        "Boll development": 80,
        "Boll opening and maturity": 80,
    }[stage]
    soil_temp = float(np.clip(air_temp - rng.uniform(1, 7), 12, 45))

    rows.append({
        "scenario_id": f"SCN-{i + 1:03d}",
        "soil_type": soil,
        "season": season,
        "crop_stage": stage,
        "area_m2": float(rng.choice([10.0, 20.0, 20.0, 50.0])),
        "current_soil_moisture_vwc_pct": round(float(moisture), 2),
        "soil_temperature_c": round(float(soil_temp), 2),
        "air_temperature_c": round(float(air_temp), 2),
        "relative_humidity_pct": round(float(humidity), 1),
        "rainfall_previous_24h_mm": round(float(rain_prev), 2),
        "forecast_rain_next_24h_mm": round(float(rain_forecast), 2),
        "irrigation_efficiency_pct": float(rng.choice([85.0, 90.0, 92.0, 95.0])),
        "soil_ph": round(float(rng.uniform(5.5, 8.5)), 2),
        "electrical_conductivity_ds_m": round(float(rng.uniform(0.2, 3.5)), 2),
    })

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {len(rows)} synthetic scenarios to {OUT}")
