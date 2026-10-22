# Q&A Preparation — Defence Answers

Short, confident answers for the four expected questions, plus likely follow-ups.
Everything here is backed by the real artefacts in `reports/` and `models/`.

---

## 1. Why did you choose the dataset you used?

**30-second answer:**
"We chose the municipal complaints dataset because it directly matches our task.
It contains real complaint **text** plus the two labels we need to predict — the
**service category** and the **urgency level** — so it's a genuine supervised
classification problem, not something we had to invent. It also has enough
labelled examples (1,100 records across 9 service categories and 4 urgency
levels) to train and fairly evaluate a baseline plus four models."

**Supporting points if pushed:**
- It reflects how complaints really arrive: free-text descriptions, a timestamp,
  a location, and context fields — exactly the information available **at intake**.
- Both target options the brief allows (service type and urgency) are present and
  sufficiently populated, so we could do both tasks and report them separately.
- It let us demonstrate the full pipeline: text features (TF-IDF) + tabular
  features, class imbalance, and real data-quality issues to clean.

**Follow-up — "Is it big enough?"**
"For this scope, yes. 1,100 records with stratified 5-fold CV gives stable
estimates; the baseline-to-model gap is large and consistent across folds. We'd
want more data before production, especially for the smallest classes
(`graffiti`, `other`) and for fairness checks across wards."

---

## 2. How did you clean the dataset?

**30-second answer:**
"Four steps: fix structure, fix labels, remove leakage, remove personal data.
We dropped a duplicate record, canonicalised inconsistent labels, engineered
intake-only time features, then **excluded** any column that leaks the answer or
contains personal information — all inside a reproducible pipeline so the test
set never influences cleaning."

**The concrete steps:**
1. **Structure:** dropped 1 duplicate `report_id`; parsed the mixed-format
   timestamp with `format='mixed'`.
2. **Label canonicalisation:** merged `water-leak`→`water_leakage`,
   `illegal-dumping`→`garbage`, and `California`→`CA` so the same thing isn't
   counted as two classes.
3. **Leakage removal (key point):** excluded `issue_status`,
   `municipal_department`, `resolution_datetime`, `resolution_notes`,
   `priority_score`, and `severity_score`.
   - `municipal_department` is assigned *after* routing and almost perfectly
     predicts the service type.
   - `priority_score` / `severity_score` are triage scores whose value ranges
     almost perfectly separate the urgency labels — using them is cheating.
4. **Privacy (POPIA):** removed names, emails, phones, resident IDs, exact
   address, latitude/longitude, postal code, photo URLs.
5. **Imputation/scaling inside the pipeline:** median impute + scale numerics,
   most-frequent impute + one-hot for categoricals, TF-IDF for text — all fit on
   training folds only.

**Follow-up — "How do you know the department column leaks?"**
"We cross-tabulated `issue_type` against `municipal_department` and each category
maps to essentially one department. And it's logically post-intake — it only
exists after a human routes the complaint, which is the thing we're trying to
predict."

---

## 3. How did you measure the performance of your models?

**30-second answer:**
"We evaluated the baseline and all four models on the **same untouched 20% test
set**, using **macro-F1 as the primary metric** because the classes are
imbalanced and macro-F1 weights every class equally. We also reported accuracy,
per-class precision/recall, weighted-F1, confusion matrices, and ROC-AUC / PR-AUC,
plus recall on the critical-urgency class."

**Why those metrics:**
- **Macro-F1:** stops the majority class from hiding poor performance on rare but
  important classes (e.g. `critical`).
- **PR-AUC & ROC-AUC (one-vs-rest):** assess probability quality on imbalanced
  data.
- **Confusion matrices:** show *which* categories get confused.
- **Recall on `critical`:** we must not leave hazards in a routine queue — it's
  0.88 for our urgency model.

**Protocol that makes it trustworthy:**
- Stratified 80/20 split, fixed seed 42.
- 5-fold cross-validation on the **training** set for model selection/tuning.
- Test set used **once**, only at the end.
- Statistical check: **McNemar's test** between the champion and the next-best
  model on the same test cases (no significant difference — they're close).
- **Overfitting check:** compared train vs validation scores and plotted neural
  learning curves.

**Headline results (real):**
- **Urgency:** Tabular NN macro-F1 **0.815** vs baseline **0.135**.
- **Service:** ~**0.98–1.00** macro-F1 — and we proved via ablation this is text
  separability, not leakage (removing text drops it to ~0.19).

**Follow-up — "Why is accuracy not enough?"**
"With imbalanced classes, a model can get high accuracy by favouring the common
class while failing the rare ones. The dummy baseline shows this: 37% accuracy on
urgency but only 0.14 macro-F1. Macro-F1 exposes that gap."

---

## 4. What problem are you solving?

**30-second answer:**
"Municipalities receive large volumes of complaints through many channels, and
they can be misrouted or left in a queue even when they signal an urgent hazard.
We're solving the **triage problem**: at the moment a complaint is submitted, we
suggest the **service category** and the **urgency level** so staff can review and
route it faster and more consistently."

**Frame it clearly:**
- **Prediction unit:** one complaint at intake, using only information available
  at that point.
- **Decision supported:** which department it likely belongs to, and how urgent
  it likely is.
- **Benefit:** fewer misroutes, faster human review of high-risk complaints,
  better use of limited staff and equipment.
- **What it is NOT:** it does not resolve, close, deny, or deprioritise
  complaints, and it never replaces emergency procedures — a human stays in the
  loop, especially for urgent cases.

**Follow-up — "Why not just let staff do it?"**
"Staff still do it — this gives them a fast, consistent first suggestion so the
easy cases move quickly and the urgent ones get surfaced for review sooner. It
supports the decision; it doesn't make it."

---

## Quick-fire extras they might ask

- **"Which model did you pick and why?"** — Random Forest for service category and
  the Tabular Neural Network for urgency, chosen by cross-validated macro-F1.
  Optimisation didn't beat the defaults, so we kept the simpler configuration.
- **"Did you tune hyperparameters?"** — Yes, RandomizedSearchCV (5-fold, macro-F1)
  on the strongest model; it didn't improve the held-out score, which we report
  honestly.
- **"How do you handle a bad input in the app?"** — Validation rejects empty,
  too-short, or non-text inputs like "12345" with a clear message.
- **"Is it fair across groups?"** — A known limitation: performance could be worse
  for under-represented wards. We recommend monitoring and fairness checks before
  any real deployment.
- **"Can we trust the probabilities?"** — We show them but caution they may not be
  calibrated; they indicate relative confidence, not guaranteed likelihood.
