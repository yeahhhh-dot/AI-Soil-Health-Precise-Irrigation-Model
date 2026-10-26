# Cotton Drip Irrigation Simulator — Khandesh, North Maharashtra

A Streamlit classroom prototype with:
- Fixed crop: cotton
- Fixed region: Khandesh, North Maharashtra
- Soil selector: sandy loam, clay loam, black soil, red-mixed-black soil, deep alluvial soil, black clayey soil, red soil, and red + red-mixed-black soil
- Seasonal selection: April–June, July–September, October–December, January–March
- Random Forest regression for irrigation depth (mm)
- A transparent water-balance calculation to convert a soil-moisture deficit into net/gross depth, litres for an area, required L/h over a one-hour delivery, and runtime at entered emitter capacity
- CSV training template and 100 simulated demonstration conditions

## Important scientific limits

The built-in 100 rows are synthetic scenarios whose irrigation-depth labels were calculated from the same simplified water-balance equation used by the app. They are for testing the interface and regression workflow only. Their test MAE/RMSE/R² are not evidence of real-world accuracy.

The default field-capacity and saturation moisture values attached to soil labels are editable placeholders, not measured hydraulic values for every Khandesh field. Replace them with lab values or verified local published data. Soil labels alone cannot determine exact hydraulic properties.

Moisture input must be volumetric water content (VWC, % by volume), or a calibrated conversion to VWC. A sensor output from 0–100 is not automatically VWC.

Routine irrigation toward 95–100% saturation is generally not recommended. Field capacity is the default. Near-saturation mode is included only as an educational option and can cause waterlogging or deep percolation. FAO describes the distinction between saturation and field capacity: https://www.fao.org/4/r4082e/r4082e03.htm

The volume calculation assumes the current moisture represents the entered root-zone depth and a uniform refill over the entered area. Real drip patterns depend on crop growth stage, root-zone depth, emitter spacing/discharge, wetted area, soil hydraulic conductivity, irrigation efficiency and field variability. Get agronomic/lab values before interpreting results as recommendations.

Previous-24-hour rainfall is available as a model feature but is not subtracted a second time from the transparent refill equation because current measured soil moisture should already reflect rain that reached the measured zone.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy with GitHub and Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, `README.md`, `data_template.csv`, and `demo_100_conditions.csv` to the root of a GitHub repository.
2. In Streamlit Community Cloud, create an app linked to that repository and branch `main`.
3. Set main file path to `app.py` and deploy.

## Training template for real data

`data_template.csv` contains the column names required for supervised Random Forest regression. The target `irrigation_depth_mm` should be the observed gross irrigation depth in mm for the corresponding irrigation event. The input values must be aligned to the same field/event and use consistent units. If a paper reports litres or m³/ha, document and convert the quantity to an equivalent depth (1 mm = 10 m³/ha; 1 mm over 1 m² = 1 L).

A paper containing only soil moisture readings, without a corresponding irrigation amount/decision, cannot by itself train a supervised irrigation-volume regressor. Cite source papers and record site, sampling dates, measurement depth, methods, missing-value handling and every conversion.

## Sources to begin a literature review

- Central Ground Water Board, Jalgaon district: https://www.cgwb.gov.in/sites/default/files/2022-11/jalgaon_f.pdf
- Maharashtra Bt cotton drip/fertigation field study (Rahuri): https://pub.isa-india.in/index.php/ija/article/view/5660
- FAO, soil water properties: https://www.fao.org/4/r4082e/r4082e03.htm
- FAO, irrigation water needs: https://www.fao.org/4/S2022E/s2022e08.htm
