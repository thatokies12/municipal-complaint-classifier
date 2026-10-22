# Data Card — Municipal Complaints Training Set

## Overview
- **File:** `municipal_training_set_1100.csv`
- **Rows:** 1,100 complaint records
- **Columns:** 26
- **Unit of observation:** One municipal complaint as it exists at the time it is first submitted.
- **Purpose:** Supervised classification of a complaint's **service category** (primary target) and **urgency level** (secondary target) from information available at intake.

## Provenance & licence
The dataset was supplied as part of the IDA group assignment as an approved,
lawfully reusable training set. The exact originating municipality and collection
period are not embedded in the file. Treat it as sample/educational municipal data.
Do **not** redistribute if the supplying licence prohibits it; the cleaned,
de-identified pipeline inputs are what the application consumes.

## Targets
| Target | Column | Classes | Notes |
|---|---|---|---|
| Service category (primary) | `issue_type` | garbage, pothole, water_leakage, noise, streetlight, drainage, illegal_parking, graffiti, other | Raw labels canonicalised: `water-leak`→`water_leakage`, `illegal-dumping`→`garbage`. |
| Urgency (secondary) | `priority_level` | low, medium, high, critical | Reasonably balanced (207–405 each). |

## Class distribution (raw, before canonicalisation)
- **issue_type:** garbage 159, pothole 150, water_leakage 145, noise 144, streetlight 140, drainage 135, illegal_parking 133, graffiti 26, illegal-dumping 23, water-leak 23, other 22.
- **priority_level:** medium 405, high 273, critical 215, low 207.

## Features used at intake
- `issue_description` — free text (primary signal, TF-IDF).
- `report_datetime` — intake timestamp; engineered into hour / day-of-week / month.
- `location_city`, `location_state` — generalised location (state values canonicalised).
- `is_anonymous` — boolean.
- `area_importance`, `citizen_reports_count` — numeric context available at intake.

## Excluded features and why
**Data leakage (recorded after intake / encode the target):**
- `issue_status`, `resolution_datetime`, `resolution_notes` — post-outcome.
- `municipal_department` — assigned after routing; a near-deterministic function of `issue_type` (confirmed by crosstab).
- `priority_score`, `severity_score` — triage-derived scores whose value ranges are nearly disjoint per `priority_level` (e.g. low 1.06–3.98, critical 8.5–10.0); using them to predict urgency is leakage.

**Personal information (POPIA — removed/anonymised):**
- `resident_id`, `resident_name`, `resident_email`, `resident_phone`
- `location_street_address`, `latitude`, `longitude`, `photo_url`
- `location_postal_code` (near-identifying, high cardinality).

## Known data-quality issues
- Mixed datetime formats in `report_datetime` (parsed with `format='mixed'`).
- Inconsistent categorical spellings: `location_state` (`CA`/`California`), and `issue_type`/`municipal_department` variants — canonicalised in `prepare.py`.
- 1 duplicate `report_id` (dropped on load).
- Missingness concentrated in PII and post-outcome fields (all excluded) and `photo_url`.
- Small `issue_type` classes (`other`, `graffiti`) → macro-F1 emphasised to avoid majority-class masking.

## Responsible use
Predictions are **advisory decision-support** for authorised municipal staff.
They must not deny service, deprioritise a resident, or replace emergency
procedures. High-urgency complaints must always receive human review.
