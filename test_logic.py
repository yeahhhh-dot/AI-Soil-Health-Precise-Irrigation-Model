"""Small regression checks for the irrigation calculator (run: python -m unittest)."""
from pathlib import Path
import unittest

from irrigation_logic import evaluate_scenario, load_parameter_table

ROOT = Path(__file__).resolve().parent


class IrrigationLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_parameter_table(str(ROOT / "soil_stage_parameters.csv"))
        cls.base = {
            "scenario_id": "TEST",
            "soil_type": "Black soil",
            "season": "April-June",
            "crop_stage": "Flowering",
            "area_m2": 20.0,
            "current_soil_moisture_vwc_pct": 17.0,
            "soil_temperature_c": 28.0,
            "air_temperature_c": 32.0,
            "relative_humidity_pct": 40.0,
            "rainfall_previous_24h_mm": 0.0,
            "forecast_rain_next_24h_mm": 0.0,
            "irrigation_efficiency_pct": 90.0,
            "soil_ph": 7.0,
            "electrical_conductivity_ds_m": 1.0,
        }

    def test_target_is_inside_95_to_100_percent_fc(self):
        out = evaluate_scenario(self.base, self.params)
        self.assertGreaterEqual(out["target_pct_of_field_capacity"], 95.0)
        self.assertLessEqual(out["target_pct_of_field_capacity"], 100.0)

    def test_current_moisture_at_target_returns_zero_flow(self):
        item = dict(self.base)
        item["current_soil_moisture_vwc_pct"] = 35.0  # >= 95% of FC=36% (34.2)
        out = evaluate_scenario(item, self.params)
        self.assertFalse(out["irrigation_needed_now"])
        self.assertEqual(out["required_flow_for_one_hour_lph"], 0.0)

    def test_area_doubling_doubles_required_volume(self):
        first = evaluate_scenario(self.base, self.params)
        item = dict(self.base)
        item["area_m2"] = 40.0
        second = evaluate_scenario(item, self.params)
        self.assertAlmostEqual(second["gross_water_volume_l"], 2 * first["gross_water_volume_l"], places=8)

    def test_flow_is_gross_litres_for_one_hour(self):
        out = evaluate_scenario(self.base, self.params)
        self.assertAlmostEqual(out["required_flow_for_one_hour_lph"], out["gross_water_volume_l"], places=8)

    def test_rejects_moisture_above_saturation(self):
        item = dict(self.base)
        item["current_soil_moisture_vwc_pct"] = 54.0
        with self.assertRaises(ValueError):
            evaluate_scenario(item, self.params)


if __name__ == "__main__":
    unittest.main()
