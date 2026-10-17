# DATASET CARD - MedRelay follow-up datasets (mode: full, seed 20261017)

**Purpose.** Reliable, leakage-checked training/evaluation data for MedRelay: clinical-action extraction, unresolved follow-up risk, a temporal evidence graph and a
real-time replay stream. **Not for clinical use. Not clinically validated.**

## What is real, synthetic, annotated, generated
| dataset | nature | used for |
|---|---|---|
| Synthetic Hospital v1.3 | synthetic (LLM-assisted) | notes, imaging orders, documented labs/imaging, official patient splits |
| Synthea sample | synthetic (rule-based) | timestamped encounters/labs/meds/procedures/care plans; FHIR notes |
| MTS-Dialog | annotated (human summaries of dialogues) | action-language variety (silver labels only) |
| ACI-Bench | annotated (clinician notes of simulated visits) | action-language variety (silver labels only), CC BY 4.0 |
| Kaggle No-Show | **real**, de-identified | appointment non-attendance baseline (independent of everything else) |
| follow-up scenarios (obligations, events, outcomes) | **generated** (seeded simulator) | risk prediction, graph, replay |

## Sizes (measured)
* Source records: {"synthetic_hospital": {"patients": 1268, "encounters": 5602, "notes": 5602, "imaging_orders": 1865}, "synthea_csv": {"patients": 108, "encounters": 5571, "observations": 68648, "medications": 3850, "procedures": 15884, "careplans": 349, "fhir_document_reference_notes": 5635}, "mts_dialog": {"MTS-Dialog-TrainingSet": 1201, "MTS-Dialog-ValidationSet": 100, "MTS-Dialog-TestSet-1-MEDIQA-Chat-2023": 200, "MTS-Dialog-TestSet-2-MEDIQA-Sum-2023": 200}, "aci_bench": {"train": 67, "valid": 20, "clinicalnlp_taskB_test1": 40, "clinicalnlp_taskC_test2": 40, "clef_taskC_test3": 40}, "noshow": {"rows": 110527}}
* Canonical: 3284 patients (1908 are document-level pseudo-patients), 11173 encounters, 14989 notes.
* Action classification: {'train': 5634, 'test': 3382, 'validation': 986} examples after duplicate/near-duplicate exclusion; label counts {'appointment_referral': 258, 'laboratory_test': 248, 'imaging': 170, 'procedure': 171, 'medication_action': 1072, 'patient_communication': 251, 'followup_review': 316, 'other_action': 196, 'no_action': 7539}; tiers {'silver': 10002}; **gold = 0**.
* Obligations: 24296 in 9037 chains; non-root outcomes {'completed': 8640, 'overdue_unresolved': 6312, 'unverified': 2069, 'censored_or_unknown': 454}; verified-evidence share 61.21% (excluding documented roots).
* Dependency depth distribution {1: 8902, 2: 8869, 3: 5864, 4: 441, 5: 195, 6: 25}; 135 cross-chain dependency edges.
* Risk dataset: 12590 assessments; by split {'train': 7468, 'test': 3244, 'validation': 1878}; outcomes {'overdue_unresolved': 6312, 'completed': 4040, 'unverified': 1784, 'censored_or_unknown': 454}; binary {'train': {'1': 3732, '0': 2384}, 'validation': {'1': 1002, '0': 558}, 'test': {'1': 1578, '0': 1098}}.
* No-show: 110521 rows, no-show rate 0.2019; patient-grouped split {'test': {'n': 16471, 'no_show_rate': 0.2027}, 'train': {'n': 77554, 'no_show_rate': 0.2017}, 'validation': {'n': 16496, 'no_show_rate': 0.2022}}; chronological {'test': {'n': 16579, 'no_show_rate': 0.1836}, 'train': {'n': 77364, 'no_show_rate': 0.2091}, 'validation': {'n': 16578, 'no_show_rate': 0.1865}}.
* Graph: 103627 nodes / 223925 edges. Event stream: 90904 deliveries.

## Splits
Official splits are preserved (Synthetic Hospital: train->train, public->validation, heldout->test, **never trained on**; MTS/ACI: official partitions, test sets -> `test`).
Synthea (no official split) and No-Show use deterministic patient-level hash assignment (70/15/15). Splits are assigned **before** scenario generation and reused by every
artefact for the same patient. Exact/near-duplicate action texts are removed from lower-precedence splits (train > validation > test).

## Intended use
Train/evaluate (1) a multi-label action classifier, (2) a follow-up failure risk model, (3) appointment attendance risk (separate), (4) graph/replay consumers.

## Limitations & assumptions (read before use)
1. **No human-reviewed (gold) action labels exist**; silver labels are rule-based and noisy (an informal developer read-through of samples found false positives such as past-tense statements and some multi-label ASR run-ons; precision has NOT been measured).
2. Synthetic Hospital provides only month-granular dates (all on the 15th); sub-day times for it are imputed. Plan sections exist in ~17% of its notes and are mostly narrative.
3. Synthea/SH contain **no** appointment, task, result-review or patient-notification events. Every follow-up event after the seed is **simulated**; outcome rates (e.g. overdue share) are
   simulation parameters, not epidemiology. Model scores on this data show pipeline soundness, not real-world effectiveness.
4. Behaviour profiles (scenario types) are generator-specific templates; they are kept in a hidden ground-truth file, but a model can still learn the generator's causal structure.
5. MTS/ACI-derived obligations are anchored to simulated dates and are single-document pseudo-patients (history features are empty for them).
6. Category skew: result-review/notification chains dominate the risk rows (see `reports/dataset_quality.md`, Q12).
7. Synthea timelines span decades (earliest 1942); replays should time-compress or filter by date.
8. No-show data: Brazilian public-hospital appointments from 2016; neighbourhood/age/gender are demographic proxies - subgroup rates in `reports/dataset_statistics.json`; do not use to deny or deprioritise care.
9. Rule extraction thresholds, scenario probabilities and censoring rates are in `src/` and `config.yaml`.

## No-show subgroup distribution & bias screen (real data; descriptive, not causal)

| subgroup | level | n | no-show rate |
|---|---|---|---|
| gender | F | 71836 | 0.2031 |
| gender | M | 38685 | 0.1996 |
| age_band | 0-17 | 27378 | 0.219 |
| age_band | 18-34 | 24244 | 0.2397 |
| age_band | 35-49 | 21864 | 0.2053 |
| age_band | 50-64 | 22634 | 0.1672 |
| age_band | 65+ | 14401 | 0.155 |
| scholarship | 0 | 99660 | 0.198 |
| scholarship | 1 | 10861 | 0.2374 |
| handicap_level | 0 | 108282 | 0.2023 |
| handicap_level | 1 | 2040 | 0.1784 |
| handicap_level | 2 | 183 | 0.2022 |
| handicap_level | 3 | 13 | 0.2308 |
| handicap_level | 4 | 3 | 0.3333 |
| hypertension | 0 | 88720 | 0.209 |
| hypertension | 1 | 21801 | 0.173 |
| diabetes | 0 | 102578 | 0.2036 |
| diabetes | 1 | 7943 | 0.18 |
| wait_band | (-1, 0] | 38562 | 0.0465 |
| wait_band | (0, 3] | 14675 | 0.2289 |
| wait_band | (14, 30] | 17371 | 0.3259 |
| wait_band | (3, 7] | 17510 | 0.252 |
| wait_band | (30, 400] | 10378 | 0.33 |
| wait_band | (7, 14] | 12025 | 0.3047 |

Neighbourhoods: 81 (7 with <100 rows); neighbourhood no-show rate ranges 0.1456-0.2892 among neighbourhoods with >=100 rows. Neighbourhood, age, gender and the welfare flag (`scholarship`) are demographic/geographic proxies; the baseline model is **not** equalised across groups. Baseline subgroup behaviour on the held-out patient-grouped test split (XGBoost sanity model):

| subgroup | level | n (test) | prevalence | mean predicted | share flagged at top-20% risk | PR-AUC |
|---|---|---|---|---|---|---|
| gender | F | 10758 | 0.2096 | 0.2047 | 0.1998 | 0.3749 |
| gender | M | 5713 | 0.1896 | 0.1996 | 0.2006 | 0.3474 |
| age_band | 0-17 | 4028 | 0.2108 | 0.2146 | 0.2607 | 0.3706 |
| age_band | 18-34 | 3492 | 0.2405 | 0.2424 | 0.3749 | 0.4083 |
| age_band | 50-64 | 3571 | 0.1652 | 0.1703 | 0.0616 | 0.2861 |
| age_band | 35-49 | 3362 | 0.2234 | 0.2104 | 0.2002 | 0.3712 |
| age_band | 65+ | 2018 | 0.1526 | 0.1564 | 0.0213 | 0.2686 |
| scholarship | 0 | 14853 | 0.1981 | 0.199 | 0.1823 | 0.354 |
| scholarship | 1 | 1618 | 0.2447 | 0.2387 | 0.3628 | 0.4317 |
| handicap_level>0 | 0 | 16162 | 0.2033 | 0.2032 | 0.2014 | 0.3672 |
| handicap_level>0 | 1 | 309 | 0.1715 | 0.1893 | 0.1294 | 0.2469 |

Differences in the share flagged at a fixed top-20% cut-off show the model concentrates flags in some age groups. **Do not use a model trained on this data to deny, delay or deprioritise care**; any deployment needs fairness evaluation on local data and clinical governance.

## Loading
See README.md ("Loading datasets"). Hidden fields live only in `followup_risk/ground_truth_hidden.parquet` and `event_stream/ground_truth_*.jsonl`.
