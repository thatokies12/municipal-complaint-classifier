# Civic & Municipality Complaint Classification System
### IDA Project — Team Presentation

> Each `---` is one slide. Speaker notes are under **Say:**. All numbers are the
> real measured results from this project (see `reports/tables/`).

---

## Slide 1 — Title

# CityDesk
## Civic & Municipality Complaint Classification System

An AI decision-support tool that suggests a complaint's **service category** and
**urgency** the moment it is submitted.

Team: Malaza TA (219147872) · Aphane TM (219699340) · Kabini AN (219976852)
Lecturer: Mothilal D

**Say:** "We built a system that helps municipal staff triage complaints faster
by suggesting what kind of problem it is and how urgent it is — at intake."

---

## Slide 2 — The problem

**Municipal complaints arrive through many channels** (call centre, web, email,
app, walk-in) and can be:
- misclassified or sent to the wrong department,
- left in a queue even when they signal an urgent hazard.

**Consequences:** delayed service delivery, poor use of limited staff/equipment,
and urgent problems sitting in routine queues.

**Our goal:** at the moment a complaint is received, suggest
1. the **service category** (water, roads, waste, lighting, …), and
2. the **urgency level** (low / medium / high / critical),
so staff can review and route it consistently.

**Say:** "This is a triage-support problem, not an automation problem. A person
still decides — we just give them a fast, consistent first suggestion."

---

## Slide 3 — Who it helps

- **Residents, businesses, visitors** — faster response to real hazards.
- **Customer-service agents & triage supervisors** — a consistent first-pass label.
- **Service departments** (water, roads, electricity, waste) — fewer misroutes.

**Prediction unit:** one complaint, as it exists when first submitted.

**Say:** "The benefit is reduced misrouting and faster human review of high-risk
complaints."

---

## Slide 4 — The dataset

- **Source:** approved municipal complaints training set (`municipal_training_set_1100.csv`).
- **Size:** 1,100 records × 26 columns.
- **Two usable targets:**
  - **Service category** (`issue_type`) — 9 classes (primary).
  - **Urgency** (`priority_level`) — low / medium / high / critical (secondary).
- Contains complaint text, timestamps, location, and context fields.

**Say:** "We chose this dataset because it directly matches the task: it has real
complaint text plus the two labels we need to predict."

---

## Slide 5 — Data audit (what we actually found)

| Check | Finding |
|---|---|
| Rows / columns | 1,100 × 26 (→ 1,099 after 1 duplicate id removed) |
| Duplicates | 1 duplicate `report_id` |
| Missing values | concentrated in PII + post-outcome fields |
| Label issues | `water-leak`/`water_leakage`, `illegal-dumping`, `CA`/`California` |
| Class balance | service moderately imbalanced; urgency fairly balanced |

**Say:** "We didn't assume anything — we inspected the file first and found real
label inconsistencies and leakage risks we had to handle."

---

## Slide 6 — Preventing data leakage (critical)

We **excluded** anything recorded *after* intake or that encodes the answer:
- `issue_status`, `resolution_datetime`, `resolution_notes` — post-outcome.
- `municipal_department` — assigned after routing; almost perfectly predicts `issue_type`.
- `priority_score`, `severity_score` — triage scores whose ranges almost
  perfectly separate the urgency labels.

**Why it matters:** using these would give fake-high scores that collapse in the
real world.

**Say:** "A model that sees the department or the priority score is cheating —
those only exist after a human has already triaged the complaint."

---

## Slide 7 — Privacy (POPIA)

Removed / anonymised personal information:
- names, emails, phone numbers, resident IDs,
- exact street address, latitude/longitude, postal code, photo URLs.

Personal identifiers are excluded from **training, charts, and logs**.

**Say:** "Under POPIA we only keep what's needed for the task — generalised city
and state, never the person's identity or exact location."

---

## Slide 8 — Features used at intake

- **Complaint text** → TF-IDF (the main signal).
- **Generalised location** → city, state (one-hot).
- **Context** → area importance, related reports count (scaled).
- **Time** → hour / day-of-week / month from the intake timestamp.
- **Anonymous flag.**

All preprocessing lives in one reproducible scikit-learn pipeline, fit on
**training data only**.

**Say:** "Every model gets the same information, through the same pipeline, so the
comparison is fair."

---

## Slide 9 — Models compared

**Baseline:** DummyClassifier (most-frequent) — the bar to beat.

**Four substantive models:**
1. Logistic Regression (traditional, interpretable)
2. Random Forest (traditional, non-linear)
3. MLP (neural)
4. Tabular Neural Network — Keras feed-forward (ReLU, dropout, L2, Adam, early stopping)

**Say:** "Two traditional, two neural — exactly what the brief asked for, all on
identical splits."

---

## Slide 10 — Training & validation protocol

- **80/20 split**, stratified, fixed **seed 42**.
- **5-fold cross-validation** on the training set for model selection.
- **Test set untouched** until the final, single evaluation.
- Class imbalance handled with **class weights on training data only**.

**Say:** "We locked the test set away and only used it once, at the very end."

---

## Slide 11 — Results: Service category

| Model | Accuracy | Macro-F1 | PR-AUC |
|---|---|---|---|
| Dummy baseline | 0.164 | 0.031 | 0.111 |
| Logistic Regression | 1.000 | 1.000 | 1.000 |
| Random Forest | 0.986 | 0.979 | 0.999 |
| MLP | 0.986 | 0.941 | 0.991 |
| Tabular Neural Net | 0.995 | 0.988 | 1.000 |

**Honest note:** scores are very high because complaint text often contains the
category word (e.g. "pothole"). We verified this is **text separability, not
leakage**. Removing text collapses macro-F1 to ~0.19 (our ablation).

**Say:** "The service task is easy because the words give it away — and we proved
that with an ablation."

---

## Slide 12 — Results: Urgency (the harder, informative task)

| Model | Accuracy | Macro-F1 | PR-AUC |
|---|---|---|---|
| Dummy baseline | 0.368 | 0.135 | 0.250 |
| Logistic Regression | 0.791 | 0.799 | 0.869 |
| Random Forest | 0.786 | 0.792 | 0.845 |
| MLP | 0.809 | 0.808 | 0.876 |
| **Tabular Neural Net** | **0.809** | **0.815** | **0.876** |

Recall on the **critical** class: **0.88** — important for not missing hazards.

**Say:** "Urgency is where the models actually have to work, and every one beats
the baseline by a wide margin."

---

## Slide 13 — How we measured performance

- **Macro-F1 (primary)** — treats every class equally, so small classes count.
- Accuracy, per-class precision/recall, weighted-F1.
- **Confusion matrices** — where each model makes mistakes.
- **ROC-AUC and PR-AUC** (one-vs-rest) — probability quality on imbalanced data.
- **Recall on the urgent class** — we must not miss hazards.

**Say:** "Accuracy alone can hide poor performance on rare but important classes,
so macro-F1 and PR-AUC are our headline metrics."

---

## Slide 14 — Optimisation & statistical rigour

- **RandomizedSearchCV** (5-fold, scoring macro-F1) on the strongest model.
  - It did **not** beat the defaults → we kept the simpler model (honest result).
- **3 ablations:** removing text, location, and class weighting — one change each.
  - Text is the dominant signal; other components matter less.
- **McNemar's test:** champion vs strongest comparator on the same test cases.
  - No statistically significant difference (p ≈ 0.42–0.63) — the top models are close.

**Say:** "We tuned, we tested statistically, and we reported the truth — even when
tuning didn't help."

---

## Slide 15 — The interface (live demo)

**CityDesk** — a web dashboard for authorised staff:
- **Dashboard** — live stats and selected models.
- **Classify Complaint** — type a complaint → get category + urgency + confidence.
- **Dataset Explorer / Model Performance / Model Comparison** — evidence pages.

Loads the **saved pipeline** (no retraining) and validates input (rejects empty /
"12345" / too short).

**Demo cases:**
1. "Main water pipe has burst and flooding is occurring." → Water-related
2. "Streetlight has not been working for several days." → Streetlight
3. "Garbage collection was missed this week." → Waste
4. "12345" → rejected with a clear message

**Say:** *(switch to the running app and classify each example live)*

---

## Slide 16 — Responsible use

- Predictions are **advisory** and may be wrong.
- Staff **must review** every suggestion.
- The model **must not** auto-reject complaints or deny services.
- **Emergency procedures are never replaced** by the model.
- High-urgency complaints always get human review.

**Say:** "This is a support tool. A human always stays in the loop, especially for
urgent cases."

---

## Slide 17 — Limitations & recommendation

**Limitations:** possible label noise, class imbalance, limited geographic
coverage, and service patterns that change over time. Performance should be
monitored and checked for fairness across wards/groups.

**Recommendation:** deploy as a **triage aid** with human oversight; monitor
accuracy and drift after deployment; prefer the simpler model where it's
practically as good.

**Say:** "We recommend it as a decision-support pilot, with monitoring — not a
replacement for staff judgement."

---

## Slide 18 — Reproducibility

- Fixed seed (42), one preprocessing pipeline, saved model artefacts.
- `requirements.txt`, `README.md`, `data_card.md`, `data_dictionary.csv`.
- One-command run: `python -m src.run_all` → launch `streamlit run src/app.py`.
- Automated tests (26 passing) for preprocessing, prediction and validation.

**Say:** "Anyone can reproduce every number in this talk from the repo."

---

## Slide 19 — Summary

- Real municipal data, carefully audited and de-identified.
- Leakage and privacy handled deliberately.
- Baseline + 4 models, fair comparison, honest statistics.
- Urgency macro-F1 **0.82** vs baseline **0.14**; service **~0.99**.
- A working, validated interface wired to the saved model.

**Thank you — questions?**

---

## Slide 20 — Contribution statement

| Member | Contribution |
|---|---|
| Malaza TA | *(fill in)* |
| Aphane TM | *(fill in)* |
| Kabini AN | *(fill in)* |

**Say:** "Each of us can defend our part of the modelling, validation and
evaluation choices."
