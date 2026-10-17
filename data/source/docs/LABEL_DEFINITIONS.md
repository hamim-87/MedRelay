# LABEL DEFINITIONS

## Dataset 1 - action categories (multi-label; `labels` list and `y_<category>` columns)
| category | meaning (instruction/order language directed at a future action) | NOT this |
|---|---|---|
| appointment_referral | refer/consult a specialist, schedule or arrange an appointment/visit | merely mentioning a past consult |
| laboratory_test | order/repeat/check blood, urine, culture, A1c ... | lab values already resulted/reported |
| imaging | order/schedule x-ray, CT, MRI, ultrasound, echo ... | imaging result narrative |
| procedure | schedule/recommend a surgery, biopsy, endoscopy, injection, ECG ... | risks of a procedure |
| medication_action | start/stop/adjust/continue/refill a drug, medication reconciliation | home-medication lists, past exposure |
| patient_communication | notify/call/instruct/counsel the patient, return precautions | questions to the patient |
| followup_review | follow up/recheck/re-evaluate/review results in a timeframe | past visits |
| other_action | admit/discharge, diet/exercise/lifestyle directive, supportive care | |
| no_action | none of the above (includes completed-result records, history, questions, risks) | |

`tense` values: `directive` (forward-looking/orders), `historical`, `record` (documented completion or result), `question`, `none`.
Silver labels come from `src/action_rules.py` (rule ids in `rules`, per-category evidence spans in `evidence_spans`, offsets in `char_start/char_end` relative to
`span_basis`). Section headers of MTS-Dialog (PLAN, DISPOSITION ...) are note-section labels and are **not** used as action labels.
Tiers: `gold` = human-reviewed with `reviewer_id`+`reviewed_at`; `silver` = rule-proposed; `synthetic` = generated text (none in this release).

## Dataset 2 - outcome labels (per obligation; defined by the event history, see `generate_scenarios.label_node`)
| label | definition |
|---|---|
| completed | a verified completion-evidence event for this step exists at or before the deadline, with no conflicting secondary record |
| overdue_unresolved | no completion evidence by the deadline AND the deadline miss is documented (task-tracker snapshot `open` at deadline+1h, or documented no-show/cancellation), OR completion evidence only appears after the deadline (`resolved_late=True`) |
| unverified | completion evidence conflicts with a secondary record, or no completion and no status evidence at all (documentation gap) despite full observation |
| censored_or_unknown | the observation window ended before the deadline could be resolved |

Binary target `label_binary`: 1 = overdue_unresolved, 0 = completed; **unverified and censored_or_unknown are excluded** (never treated as failures).
`assessment_time` = created + U(0.2, 0.8) x (deadline - created), seeded per obligation; only obligations without completion evidence at that time are kept.
The risk question: *given only events <= assessment_time, will the obligation be unresolved at its deadline?* Deterministic overdue rules (flag only after the deadline) are
reported separately in the baselines and cannot provide early warning.

## Dataset 3 - `no_show`
1 = patient did not attend the scheduled appointment (`No-show = Yes` in the source). It is an attendance label, **not** a follow-up-completion label.

## Evidence verification states
`verified` (documented evidence), `unverified` (weak/secondary), `inferred` (derived relation, e.g. a later documented encounter inside the window, confidence 0.3),
`contradicted` (conflicting secondary record).
