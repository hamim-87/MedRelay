# MedRelay Leakage Report

Categories: repeated notes, paraphrased/near-duplicate text, future events, final status/generator columns, graph edges after assessment time, patient identity, source-document overlap, generator-specific templates, test-set use during tuning.

| id | check | status | detail |
|---|---|---|---|
| Q09a | action examples: exact/near duplicates across splits | **PASS** | exact-duplicate texts spanning splits: 0; eval rows with cosine>=0.92 to a train row: 0 (excluded rows were removed before writing) |
| Q09b | notes: identical note text across splits | **WARN** | 43 identical normalized note texts appear in more than one split (by source: {'aci_bench': 0, 'mts_dialog': 43, 'synthea': 0, 'synthetic_hospital': 0}); duplicated-text groups by source: {'aci_bench': 0, 'mts_dialog': 68, 'synthea': 0, 'synthetic_hospital': 0}. Synthea/MTS/ACI short or templated notes are expected to repeat; they are in the source as-is and splits are patient/official-level. |
| Q09c | no-show: duplicate rows spanning splits | **WARN** | 2 duplicate feature-vectors span splits (patient-grouped: same patient never in 2 splits) |
| Q09d | cross-source sentence overlap (ACI/MTS/SH/Synthea) | **WARN** | 22 normalized sentences occur in >1 source dataset: {('aci_bench', 'mts_dialog'): 22} |
| L01 | future-information leakage (features vs events <= assessment time) | **PASS** | features recomputed independently for 400 sampled rows match; no forbidden/generator columns; all rows open at assessment_time and assessment_time < deadline |
| L02 | no-show history features use only earlier appointments | **PASS** | 0/300 sampled rows mismatch brute-force recomputation |
| L03 | no-show: timing-uncertain column kept out of primary features | **PASS** | ['sms_received__timing_uncertain'] present only as clearly-labelled optional column |
| L04 | graph edges created after assessment time are excluded from snapshots | **PASS** | 0 sampled obligations have a COMPLETED_BY edge at/before assessment_time; 0 have an overdue-trigger review edge before assessment |
| L05 | patient-level split independence & reuse across all artefacts | **PASS** | every patient has one split reused in notes/obligations/events/action/risk/graph/stream; no-show patients never span splits |
| L06 | SH held-out patients never appear in training files | **PASS** | 0 rows |
| L07 | single-feature AUC scan for generator/template leakage | **PASS** | max single-feature AUC (train+val) = 0.665 (elapsed_fraction); top5 [('elapsed_fraction', 0.665), ('parent_state', 0.616), ('n_incomplete_dependencies', 0.615), ('n_incomplete_ancestors', 0.614), ('hours_since_last_event', 0.604)]. Thresholds: >0.90 FAIL, >0.75 WARN. Note: events/features are produced by a seeded generator, so even a sub-threshold model score reflects the generator's causal structure, NOT real-world predictability. |
| L08 | test split not used for tuning | **PASS** | baseline_check recorded test_used_for_model_selection=False; hyper-parameters fixed a priori, validation only for early stopping |
