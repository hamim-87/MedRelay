# ANNOTATION GUIDE & REVIEW PROCESS (action classification)

Purpose: turn rule-proposed **silver** labels into **gold** labels through genuine human review. LLM-generated labels must NOT be written as gold unless an
independent human review step is recorded.

1. Open `data/processed/action_classification/review_queue.csv` (679 rows: uncertain rule outputs + a stratified random audit sample).
2. For each row read `text` (and its `source_dataset`/`section`) and decide the label set using `LABEL_DEFINITIONS.md`:
   * label **forward-looking instructions/orders only**; completed results, history, questions and risk statements are `no_action`;
   * multi-label is allowed; use `|` to separate (e.g. `laboratory_test|followup_review`);
   * one sentence = one decision; do not use neighbouring context to invent an instruction.
3. Fill `labels` (required), `reviewer_id` (a stable id or initials), `reviewed_at` (ISO date) and optionally `reviewer_comment`.
4. Save the completed file as `data/annotations/reviewed/<name>.csv` (keep columns `example_id,labels,reviewer_id,reviewed_at`).
5. Re-run `python -m src.pipeline --config config.yaml --steps extract_actions,baseline_check,validate`. Rows with all three fields filled become `label_tier=gold`
   (`label_method=human_review:<file>`); the baseline then reports rule/keyword/TF-IDF precision, recall and F1 **against gold**.
6. Quality: double-annotate >= 10% of rows and report agreement (Cohen's kappa) before using gold for model selection. The pipeline requires at least
   30 gold rows to compute gold metrics.

Current state: **0 gold rows** (no human review has been performed in this release).
