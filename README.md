# Civic & Municipality Complaint Classification System

A supervised machine-learning system that, at complaint **intake**, suggests a
complaint's **service category** (primary target) and **urgency level**
(secondary target) from the information available when the complaint is first
submitted. It is a **decision-support aid** for authorised municipal staff — not
a replacement for human assessment.

This repository contains the full, reproducible pipeline (data audit →
preprocessing → five models → evaluation → optimisation → ablations → McNemar's
test) plus a macOS-inspired Streamlit interface wired to the saved model
pipeline.

---

## 1. Key findings from the actual dataset

Dataset: `municipal_training_set_1100.csv` — **1,100 rows × 26 columns**
(1 duplicate `report_id` dropped → **1,099** usable records).

**Two supported targets:**
- **Service category (`issue_type`)** — primary. 9 classes after canonicalising
  inconsistent labels (`water-leak`→`water_leakage`, `illegal-dumping`→`garbage`).
- **Urgency (`priority_level`)** — secondary. `low / medium / high / critical`,
  reasonably balanced.

**Leakage columns excluded** (recorded after intake or encode the target):
`issue_status`, `municipal_department` (a near-deterministic function of
`issue_type`), `resolution_datetime`, `resolution_notes`, `priority_score` and
`severity_score` (triage-derived scores whose value ranges are nearly disjoint
per `priority_level`).

**Personal information excluded (POPIA):** `resident_id`, `resident_name`,
`resident_email`, `resident_phone`, `location_street_address`, `latitude`,
`longitude`, `photo_url`, `location_postal_code`, `location_country`.

**Features used at intake:** complaint text (`issue_description`, TF-IDF),
generalised location (`location_city`, `location_state`), `is_anonymous`,
`area_importance`, `citizen_reports_count`, and time features engineered from the
intake timestamp (`report_hour`, `report_dayofweek`, `report_month`).

---

## 2. Measured results (held-out 20% test set)

All numbers below are produced by the code in this repo — never invented.

### Service category (`issue_type`)
| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Dummy baseline (most-frequent) | 0.164 | 0.031 | 0.046 | 0.500 | 0.111 |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 0.986 | 0.979 | 0.987 | 1.000 | 0.999 |
| MLP (neural) | 0.986 | 0.941 | 0.985 | 1.000 | 0.991 |
| Tabular Neural Network | 0.995 | 0.988 | 0.996 | 1.000 | 1.000 |

### Urgency (`priority_level`)
| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC | Recall (critical) |
|---|---|---|---|---|---|---|
| Dummy baseline (most-frequent) | 0.368 | 0.135 | 0.198 | 0.500 | 0.250 | 0.000 |
| Logistic Regression | 0.791 | 0.799 | 0.793 | 0.948 | 0.869 | 0.884 |
| Random Forest | 0.786 | 0.792 | 0.790 | 0.935 | 0.845 | 0.884 |
| MLP (neural) | 0.809 | 0.808 | 0.809 | 0.950 | 0.876 | 0.884 |
| Tabular Neural Network | 0.809 | 0.815 | 0.812 | 0.953 | 0.876 | 0.884 |

**Provisional champions (by 5-fold CV macro-F1 on the training set):**
Random Forest for service category; the Tabular Neural Network for urgency.

### Optimisation, ablations, McNemar
- **RandomizedSearchCV** (5-fold, scoring = macro-F1) was run on each champion.
  On both targets the tuned configuration **did not beat** the default
  configuration on the locked test set, so the final pipeline keeps the simpler
  pre-optimisation model (recorded honestly in `final_pipeline_meta.json`).
- **Ablations** (one change each) for service category:
  removing the complaint **text** collapses macro-F1 from **0.979 → 0.189**,
  confirming the text is the dominant signal; removing location or class
  weighting has a minor effect. For urgency, removing text is a smaller drop
  (0.815 → 0.806) because urgency also correlates with numeric context.
- **McNemar's test** (champion vs strongest comparator, same test cases) shows
  **no statistically significant difference** on either target (service
  p ≈ 0.625, exact binomial; urgency p ≈ 0.424) — the top models are close.

> **Honest note on the perfect service-category score.** Logistic Regression
> reaches 1.000 on the service-category test set, and all substantive models
> score very high. This is a **property of the data**, not leakage: many
> complaints contain the category keyword directly (e.g. "pothole", "streetlight")
> and a share of records use templated phrasing, making the text highly
> separable. We verified no post-intake/target-encoding columns are used (see
> the leakage exclusions and `tests/test_prepare.py`). The urgency task is the
> more challenging, more informative benchmark.

---

## 3. Project structure

```
IDA_Project/
├── README.md
├── requirements.txt
├── data_card.md               # dataset provenance, targets, privacy
├── data_dictionary.csv        # field-by-field audit
├── municipal_training_set_1100.csv   # (not committed; see .gitignore / POPIA)
├── src/
│   ├── prepare.py             # load, clean, leakage/PII exclusion, pipeline, split
│   ├── models.py              # baseline + 4 substantive model factory
│   ├── keras_net.py           # Keras tabular NN + sklearn-compat wrappers
│   ├── eda.py                 # generates EDA figures
│   ├── train.py               # 5-fold CV + fit + save per-model pipelines
│   ├── evaluate.py            # test metrics, confusion, ROC/PR, learning curves
│   ├── optimise.py            # RandomizedSearch, ablations, McNemar, final pipeline
│   ├── predict.py             # inference + input validation for the app
│   ├── run_all.py             # one-command full pipeline
│   └── app.py                 # Streamlit interface
├── models/
│   ├── issue_type/            # baseline/logreg/rf/mlp/tabnet + final_pipeline.joblib
│   └── priority_level/
├── reports/
│   ├── figures/               # EDA + confusion + ROC/PR + learning curves (.png)
│   └── tables/                # cv_, test_metrics_, ablations_, mcnemar_, pre_post_ (.csv)
├── notebooks/
│   └── exploratory_analysis.ipynb
└── tests/                     # pytest: prepare, predict/validation, McNemar
```

---

## 4. Setup

- **Python:** 3.9+ (developed and tested on **3.9.6**).

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**TensorFlow is optional.** With TensorFlow installed, the second neural model is
a genuine Keras feed-forward tabular network (ReLU, dropout, L2, Adam,
cross-entropy, early stopping). Without it, the project automatically falls back
to a second, differently-configured scikit-learn MLP so everything still runs.
On Apple Silicon use the default `tensorflow` wheel (not `tensorflow-cpu`):

```bash
pip install "tensorflow>=2.12,<2.17"
```

---

## 5. How to run

```bash
# 1) Full pipeline in order (EDA -> train+CV -> evaluate -> optimise):
python -m src.run_all

# ...or run the stages individually:
python -m src.eda                       # EDA figures -> reports/figures/
python -m src.train --target both       # 5-fold CV + save model pipelines
python -m src.evaluate --target both    # held-out test metrics + figures
python -m src.optimise --target both    # search + ablations + McNemar + final pipeline

# 2) Launch the interface (loads the saved pipeline; does not retrain):
streamlit run src/app.py

# 3) Run the tests:
python -m pytest tests/ -q
```

Every stage uses **random seed 42**, a stratified **80/20** split, and learns all
preprocessing statistics from the training folds only.

---

## 6. The interface

A classic, macOS-inspired Streamlit app with six pages:

1. **Home / Overview** — project summary, real dataset statistics, selected
   models and their measured performance, responsible-use notice.
2. **Classify a Complaint** — form for the complaint text and approved intake
   fields; validates input (rejects empty / too-short / non-text like `12345`);
   shows the predicted **service category** and **urgency**, class
   probabilities, and a plain-language, review-required explanation.
3. **Dataset Explorer** — category/urgency distributions, missingness, category
   vs urgency, complaints over time, text length, plus a privacy summary.
4. **Model Performance** — test-set metric table, confusion matrices, ROC and
   precision-recall curves, and neural learning curves.
5. **Model Comparison** — CV table, before/after optimisation, the three
   ablations, and the McNemar result.
6. **About / Project Information** — method summary and responsible-use notes.

The app loads the **saved, complete** preprocessing-and-model pipeline from
`models/<target>/final_pipeline.joblib` — it never recreates a partial pipeline
and never retrains on launch.

---

## 7. Responsible use & privacy

- Predictions are **advisory** and may be wrong; staff must review them.
- Probabilities may not be calibrated.
- The model must **not** auto-reject complaints, deny residents services, or
  deprioritise a complaint.
- **Emergency procedures must never be replaced** by model predictions;
  high-urgency complaints always require human review.
- Personal identifiers are excluded from training, charts, and logs.

**Data handling:** do not commit the restricted raw dataset, personal data, or
credentials (see `.gitignore`). If the data licence forbids redistribution,
obtain the dataset lawfully from the supplying municipality / provider and place
`municipal_training_set_1100.csv` at the project root before running.

---

## 8. Reproducibility notes

- Fixed seed (`RANDOM_STATE = 42`) for splitting, model init, CV, and search.
- One shared preprocessing `ColumnTransformer`, fit inside each pipeline, so the
  test set never influences preprocessing.
- All reported figures/tables are regenerated from the dataset by the commands
  above; `reports/` is the source of truth for the numbers in this README.
