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
| **Diagnostics** | Pick a vehicle, set its history, and read live (simulated or manual) sensors. See risk per component, remaining life, anomaly score, and a root-cause explanation for each alert. |
| **Forecast** | "What if I keep driving without maintenance?" — risk curves vs km and a suggested service schedule. |
| **Fleet analysis** | Score hundreds of vehicles at once (generated or uploaded CSV), rank them by priority, download results. If the CSV contains real outcomes, it grades the model. |
| **Train on your data** | Upload any table, choose the outcome columns, and train a new model on whatever columns you have. |
| **How the AI works** | Test metrics, what each metric means, feature importance, and a comparison against a perfect-knowledge "oracle". |

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
models/                      trained model + metrics.json
tests/                       pytest suite
```

## Limitations

The bundled model is trained on simulated data, so its numbers describe the simulator, not real cars. The
pipeline is built to be retrained on real telematics data (OBD-II logs + repair records) without code changes.
