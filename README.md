# Predictive Vehicle Maintenance

A machine-learning system that looks at a vehicle's sensors, usage and service history and predicts
**which component is likely to fail in the next 5,000 km**, **how many km until a repair is needed**, and
**whether the readings look abnormal** — then explains *why* and recommends what to do.

It works across 8 vehicle types (compact car → heavy-duty truck, hybrids and EVs), 37 input parameters,
and 8 failure modes, and it keeps working when sensors are missing. You can also train it on **any CSV**
with your own columns.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # dashboard
python -m pvm.train           # retrain the built-in model and print test metrics
pytest -q                     # run the tests
```

## What's in the dashboard

| Tab | What it does |
|---|---|
| **Live OBD** | Connect to a real car through an ELM327 adapter (or a built-in virtual car, or a phone-app log), watch live gauges and trouble codes, and get a diagnosis tuned to your exact make/model. |
| **Diagnostics** | Pick a vehicle, set its history, and read live (simulated or manual) sensors. See risk per component, remaining life, anomaly score, and a root-cause explanation for each alert. |
| **Forecast** | "What if I keep driving without maintenance?" — risk curves vs km and a suggested service schedule. |
| **Fleet analysis** | Score hundreds of vehicles at once (generated or uploaded CSV), rank them by priority, download results. If the CSV contains real outcomes, it grades the model. |
| **Train on your data** | Upload any table, choose the outcome columns, and train a new model on whatever columns you have. |
| **How the AI works** | Test metrics, what each metric means, feature importance, and a comparison against a perfect-knowledge "oracle". |

## Live OBD-II (your actual car)

**What you need:** an ELM327-compatible OBD-II adapter (USB, Bluetooth Classic, or Wi-Fi) and a laptop
running the app locally (`streamlit run app.py`). A cloud-hosted copy of the app can't reach hardware in your car.

1. Plug the adapter into the OBD-II port (under the dashboard, left of the steering column) and turn the
   ignition **on**.
2. Connect the laptop: USB just works; Bluetooth must be paired first (PIN is usually 1234 or 0000);
   for Wi-Fi, join the adapter's network (most use `192.168.0.10:35000`).
3. In the **Live OBD** tab choose *OBD-II adapter* → *Connect*. With the engine idling, wait for the
   coolant to pass 75 °C before trusting the diagnosis.

No adapter yet? Pick **Simulated car** — a virtual 2018 Kia Optima you can give faults
(overheating, weak battery, vacuum leak, misfire, bad alternator) — or upload a CSV from Torque Pro,
Car Scanner or OBD Fusion.

**What the car can and can't tell you.** Standard OBD-II (mode 01) covers emissions-related sensors:
rpm, speed, load, coolant temp, fuel trims, voltage, trouble codes, and sometimes oil temp and odometer.
Brake-pad thickness, tire tread, oil pressure and transmission temperature are not in the standard. They
need manufacturer-specific PIDs (add them per model in `config/vehicle_models.yaml`) or a manual check.
The app shows exactly how many of the model's inputs your car supplied, and the model treats the rest as unknown.

The diagnosis combines four layers:

| Layer | What it catches | Where |
|---|---|---|
| Trouble codes | Faults the car's computer already detected (e.g. P0217 overheating → cooling system) | `pvm/obd_live.py` |
| Range checks | Readings outside normal limits (low cranking voltage, fuel trim > ±10 %, …) | `config/vehicle_models.yaml` |
| Fleet AI model | Gradual wear, from mileage + service history + sensor patterns | `pvm/model.py` |
| Personal baseline | Drift from what *your* car normally does in the same conditions | `pvm/baseline.py` |

### Case by case: make/model profiles

The VIN is read from the car (or typed in) and decoded. It is checked with the VIN check-digit algorithm and
looked up with NHTSA's free vPIC service. If a profile exists for that make/model/year, the app uses it:

* **Manufacturer service intervals.** "Km since oil change" is rescaled to the interval the model was trained on.
  For example, 6,000 km on a Kia with a 12,070 km interval reads like 4,470 km on a 9,000 km-interval car.
* **Known issues.** For example, the 2011–2015 Optima's Theta II bearing recall and the KSDS code P1326,
  which is flagged instantly if stored.
* **Extra PIDs** for sensors the standard doesn't cover.

Included: Kia Optima 2011–2015 and 2016–2020, and Kia K5 2021+. Every value is cited in the YAML. To add a
car, copy a block, fill it in from the owner's manual, and restart. Cars without a profile fall back to the
closest generic type.

### Learning your car's "normal"

Save a few drives (at least 120 warm-engine samples), then click **Learn this car's normal**. For each signal,
a small model is trained on *your* car's data to predict what that signal should be at the current rpm,
speed, load and temperature. Later drives are compared against it, so a coolant temperature that is 6 °C higher
than usual for those conditions gets flagged, even if it's still inside the generic "normal" range. Logs and
baselines are stored privately in `data/vehicles/` (git-ignored).

## How it works

```
 vehicle config + service history ─┐
                                   ├─► hidden true health ─► noisy sensors ─┐   (pvm/simulate.py)
 driving style, climate, load ─────┘   (simulator only)     (+ gaps)        │
                                                                            ▼
                                     feature engineering (pvm/features.py) ─► gradient-boosted trees
                                                                              • 1 classifier per failure mode
                                                                              • quantile regressors for remaining life
                                                                              • isolation forest for anomalies
                                                                              (pvm/model.py)
```

**Why a simulator?** Real fleet failure data is private. The simulator generates vehicles whose components
wear according to physics-style rules (oil degrades with km and heat; pads wear faster on heavy, aggressively
driven vehicles and slower with regenerative braking; 12V batteries die sooner in hot climates; …). Each
component has a *hidden* health value; sensors are noisy functions of it, and some sensors are randomly
missing. **The model never sees the hidden health** — only what a real car would report — so it has to learn
the relationships from data.

**Why gradient-boosted trees?** They're the strongest general-purpose method for tabular data, handle
missing values and categories natively (so one model covers EVs, diesels and cars without pad sensors), and
train in about a minute on a laptop.

**Honest evaluation.** Data is split into train / validation / test. Alert thresholds are tuned on validation;
all reported metrics come from the untouched test set.

### Results (40,000 simulated vehicles, held-out test set)

| Failure mode | Prevalence | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|---|
| Cooling system | 5.8% | 0.957 | 0.753 | 68% | 75% |
| Engine oil | 9.3% | 0.966 | 0.858 | 80% | 80% |
| 12V battery | 7.6% | 0.962 | 0.819 | 77% | 78% |
| HV battery | 8.4% | 0.948 | 0.805 | 77% | 72% |
| Brakes | 8.2% | 0.959 | 0.834 | 77% | 83% |
| Tires | 14.5% | 0.968 | 0.859 | 82% | 83% |
| Transmission | 5.6% | 0.949 | 0.768 | 72% | 77% |
| Fuel & ignition | 7.0% | 0.975 | 0.821 | 79% | 76% |

Remaining life: median error ≈ 2,000 km, R² 0.86, 80% prediction interval covers 86% of cases.

Failures are random even for worn parts, so 100% is impossible. On a fresh fleet the model scores within
~0.01 ROC-AUC of an "oracle" that is allowed to see the hidden true health — it extracts nearly all the signal
that exists.

## Adding parameters

* **New sensor / input:** add the column to your data (or to `pvm/simulate.py`), add an entry to
  `config/vehicle_schema.yaml` so the dashboard gets a slider, and retrain. The model uses every column
  automatically.
* **New failure mode:** add it under `components:` in the schema and generate a `fail_<name>` column.
* **New vehicle type:** add it under `vehicle_types:` with its powertrain, mass and typical annual km.
* **Completely different data** (your own fleet, a factory machine, …):
  `python -m pvm.train --data my.csv --targets failed_soon --rul km_left --drop vehicle_id`

## Project layout

```
app.py                       Streamlit dashboard
config/vehicle_schema.yaml   every parameter, vehicle type and failure mode
pvm/simulate.py              physics-inspired fleet simulator
pvm/features.py              engineered features
pvm/model.py                 MaintenanceModel: train / predict / explain / save
pvm/train.py                 command-line training
pvm/obd_live.py              OBD-II sources (adapter, virtual car, phone logs), polling, DTC mapping
pvm/profiles.py              VIN decoding, make/model profiles, interval adaptation, range checks
pvm/baseline.py              per-vehicle "normal" model and storage
pvm/ui.py, pvm/ui_obd.py     dashboard components
config/vehicle_models.yaml   per-model service intervals, known issues, custom PIDs
models/                      trained model + metrics.json
tests/                       pytest suite
```

## Limitations

The bundled model is trained on simulated data, so its numbers describe the simulator, not real cars. The
pipeline is built to be retrained on real telematics data (OBD-II logs + repair records) without code changes.
