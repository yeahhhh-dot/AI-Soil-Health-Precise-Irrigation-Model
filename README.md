# AI Soil Health & Precision Irrigation Simulator

A beginner-friendly Streamlit web app for an academic virtual-simulation project. It uses a Random Forest classifier and sliders to simulate sensor readings.

## Important scientific limitation

`sample_training_data_demo.csv` is **synthetic demonstration data**. Its `irrigation_class` labels are created using illustrative rules, not measured irrigation decisions from a field study. It is included only so the app can be launched and tested immediately.

For a research-backed model, replace the demo data with a labelled dataset and cite its source. The CSV must contain the following exact columns:

- `soil_moisture_pct` — soil moisture in percent
- `soil_temperature_c` — soil temperature in degrees Celsius
- `soil_ph` — soil pH
- `electrical_conductivity_ds_m` — EC in dS/m (mS/cm is numerically equivalent)
- `air_temperature_c` — air temperature in degrees Celsius
- `relative_humidity_pct` — relative humidity in percent
- `rainfall_24h_mm` — rainfall in the previous 24 hours, in millimetres
- `crop_type` — crop name/category
- `irrigation_class` — target label: `No irrigation`, `Moderate irrigation`, or `High irrigation`

The app cannot learn the three categories from sensor inputs alone: a supervised classifier needs a target label for each training row. If a source gives only numeric irrigation amounts, document defensible class thresholds based on a crop/soil agronomic reference and convert the amounts into classes before training. Do not invent labels and describe them as observed results.

## How to prepare real research data

1. Pick **one crop and a defined soil/site** for a manageable first model (for example, tomato in a specified soil type).
2. Download the published dataset and read its methods/data dictionary.
3. Convert measurements into the units listed above. Examples: Kelvin to °C = K − 273.15; volumetric water content in m³/m³ to percent = value × 100.
4. Align records by the same location and date/time before joining datasets. Do not join unrelated rows just because the columns have similar names.
5. Make sure each row has a defensible `irrigation_class` label based on recorded irrigation decisions/amounts or a clearly documented crop-specific threshold method.
6. Save the prepared data as CSV, upload it in the app, and save the test-set metrics for your report.

## Algorithm outline

- Numeric sensor inputs: median imputation for missing values.
- Crop type: one-hot encoding.
- Model: `RandomForestClassifier` (300 trees, class-weight balanced, fixed random seed).
- Evaluation: 25% held-out test split; accuracy, per-class precision/recall/F1, and confusion matrix.
- Prediction: class and model probabilities.

The model probabilities are not guaranteed to be calibrated probabilities. This is an educational prototype, not a certified agricultural decision-support system.
