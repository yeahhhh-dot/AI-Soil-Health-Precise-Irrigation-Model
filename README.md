# Cotton Drip Irrigation Simulator — Fuzzy Logic + Water Balance

A Streamlit classroom simulator for cotton in Khandesh, North Maharashtra. The selectable soil profiles are **Black soil** and **Red soil**. The user selects season and crop growth stage, enters the current readings, and receives a theoretical water volume and flow rate required to move the effective root zone toward a target of **95–100% of field capacity (FC)**.

## Important scientific limits

- `soil_stage_parameters.csv` currently contains **illustrative placeholders**, not validated hydraulic measurements for black and red soils across Khandesh. The field capacity, saturation, effective root-depth-by-stage, and stage-sensitivity settings must be replaced with measurements or relevant published data for the actual soil profile and crop stage before reporting a scientific result.
- `demo_100_conditions.csv` contains 100 **synthetic test scenarios**, not 100 observed field measurements. They test that the app processes combinations of inputs and calculates outputs. They are not a training set and do not demonstrate field accuracy.
- This is a hand-authored Mamdani-style fuzzy inference system, not a supervised machine-learning model trained on the 100 rows. The rules are transparent and should be reviewed/validated with an agronomist or suitable published irrigation guidance.
- The soil moisture input is assumed to be **volumetric water content (VWC, % by volume)** and to represent the **average water content across the selected effective root zone**. A single shallow moisture sensor reading may not represent an entire 20–80 cm root zone; such mismatch can make estimated water volume seriously wrong.
- The target is 95–100% of field capacity, **not 95–100% of saturation**. Field capacity and saturation are different conditions. Avoid prolonged soil saturation for cotton unless a qualified, locally appropriate irrigation plan specifically warrants it.
- The calculated L/h is the theoretical flow needed to deliver the gross refill volume in exactly one hour. It does not guarantee the soil can safely absorb that rate, and it is not an emitter/system design. This simplified model deliberately does not calculate infiltration rate, runoff, redistribution, deep drainage, reference evapotranspiration (ET0), or daily crop evapotranspiration (ETc).
- Soil temperature is represented by a qualitative fuzzy category; air temperature and relative humidity are used to estimate vapour-pressure deficit (VPD), a proxy for atmospheric drying demand. They are **not** a full ET calculation. Expected rainfall can nudge the target toward the lower end of the allowed band and lower fuzzy urgency; it is not treated as water already present in the soil or subtracted from the measured current moisture deficit. Previous rainfall is context only because its water contribution should already be reflected in a current soil-moisture reading.
- Soil pH and electrical conductivity are shown for soil-health context, but they are not given arbitrary weights in the water-volume calculation. Season is stored as context; it does not apply an unsupported seasonal multiplier. Current weather and the selected crop stage drive the immediate response.

## Project files

```text
app.py                       Streamlit user interface
irrigation_logic.py          Fuzzy inference and water-balance calculations
soil_stage_parameters.csv    Editable hydraulic/stage defaults (placeholders initially)
demo_100_conditions.csv      100 synthetic observations for batch testing
data_template.csv            Empty CSV template for creating more test scenarios
generate_demo_data.py         Recreates the deterministic 100-row synthetic dataset
test_logic.py                 Small automated checks for the main calculations
requirements.txt             Python dependencies
README.md                    This guide
```

## Run locally on Windows

1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ (enable **Add Python to PATH** during setup).
2. Extract this project ZIP to a folder.
3. Open Command Prompt inside that folder.
4. Run:

```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
streamlit run app.py
```

The app should open in your browser, normally at `http://localhost:8501`.

Optional logic checks (the app does not need this step):

```bat
py -m unittest -v
```

## Deploy on GitHub + Streamlit Community Cloud

Upload all project files to the root of your GitHub repository, including both `.py` files, `requirements.txt`, both CSV data files, and `soil_stage_parameters.csv`. In Streamlit Community Cloud, select the repository and branch, set the main file to `app.py`, and deploy.

## How the calculation works

1. A Mamdani-style fuzzy controller assesses (a) moisture deficit relative to field capacity, (b) an illustrative crop-stage sensitivity, (c) VPD estimated from air temperature and relative humidity, (d) soil-temperature category, and (e) forecast rain. It selects a target within 95–100% of FC and returns an urgency score.
2. If current VWC is already at or above 95% of FC, the model returns 0 L/h for additional refill.
3. Otherwise, the water-balance formula is:

   `net refill depth (mm) = max(0, (target VWC - current VWC) / 100 * effective root-zone depth (mm))`

   `net volume (L) = net refill depth (mm) * field area (m²)`

   `gross volume (L) = net volume (L) / (drip efficiency / 100)`

   `required flow (L/h) = gross volume (L) / 1 hour`

   The volume identity works because 1 mm of water over 1 m² equals 1 litre.

The formula is a root-zone refill calculation, not a full seasonal or daily cotton water-requirement model. A large result for a one-hour refill can arise from a large assumed root-zone deficit; it should trigger scrutiny of the sensor's representativeness and the assumptions, not be silently capped.

## Parameter file

`soil_stage_parameters.csv` has one row per soil/stage combination. The current values are deliberately flagged `ILLUSTRATIVE PLACEHOLDER`. Preserve the column names and one unique row for every `soil_type` × `crop_stage` combination when you replace the data. At minimum, record a citation or measurement description in `parameter_note` and change `parameter_status` to state where the value came from.

The stage-sensitivity numbers are fuzzy-controller design settings, not measured biological constants. They should be checked against agronomic guidance and adjusted transparently if better evidence becomes available.

## Input CSV format

Use the exact columns in `data_template.csv`. Batch upload accepts multiple rows and exports the computed results to `cotton_fuzzy_irrigation_results.csv`.

Allowed categorical values are:

- `soil_type`: `Black soil` or `Red soil`
- `season`: `January-March`, `April-June`, `July-September`, `October-December`
- `crop_stage`: `Establishment`, `Vegetative growth and squaring`, `Flowering`, `Boll development`, `Boll opening and maturity`

## References for background concepts

- FAO, *Crop Water Needs*: cotton seasonal crop-water requirement varies with climate and growth stage; cotton is listed in an indicative range of 700–1,300 mm for a total growing period. https://www.fao.org/4/s2022e/s2022e02.htm
- FAO, *Soil and Water*: explains saturation, field capacity, and available soil water as distinct concepts. https://www.fao.org/4/R4082E/r4082e03.htm
- FAO, *ET under soil-water-stress conditions*: root-zone water storage depends on field capacity, wilting point, and rooting depth. https://www.fao.org/4/x0490e/x0490e0e.htm
- FAO, *Technical handbook on pressurized irrigation techniques*: includes a section on irrigation scheduling and effective root depth. https://www.fao.org/4/a1336e/a1336e.pdf

These sources support general concepts; they do **not** validate the placeholder black-soil/red-soil parameter values shipped with this demo.
