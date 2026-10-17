"""Data-schema compatibility, missing records, patient identity separation, train/test leakage."""
import json

import pandas as pd
import pytest

from src.data_loader import SPLITS, load_action, load_interim, load_risk, path_of, read_source, validate_datasets


def test_dataset_validation_has_no_hard_failures(cfg):
    rep = validate_datasets(cfg, write=False)
    assert rep["hard_failures"] == [], rep["hard_failures"]
    assert rep["label_provenance"]["gold_human_reviewed_action_labels"] == 0          # honesty: no human-reviewed gold exists
    assert "SIMULATED" in rep["label_provenance"]["risk_labels"]


def test_risk_feature_schema_matches_spec(cfg):
    spec = json.load(open(path_of(cfg, "source", "processed/followup_risk/feature_spec.json")))
    R = load_risk(cfg)
    for sp in SPLITS:
        assert set(spec["feature_columns"]) <= set(R[sp].columns)
    assert not (set(spec["forbidden_in_model_input"]) & set(spec["feature_columns"]))


def test_missing_source_file_is_reported_not_invented(cfg):
    with pytest.raises(FileNotFoundError):
        read_source(cfg, "processed/does_not_exist.csv.gz")


def test_namespaces_keep_sources_and_patients_separate(cfg):
    P = load_interim(cfg)["patients"]
    prefixes = P.patient_id.str.split(":").str[0]
    assert set(prefixes) <= {"sh", "sy", "mts", "aci"}
    assert P.patient_id.is_unique
    assert P[P.source_dataset.isin(["mts_dialog", "aci_bench"])].is_pseudo_patient.all()        # document-level pseudo patients are flagged, never merged with real timelines


def test_no_patient_in_two_splits_and_heldout_never_trains(cfg):
    P = load_interim(cfg)["patients"]
    R = load_risk(cfg)
    A = load_action(cfg)
    held = set(P[P.official_split == "heldout"].patient_id)
    allr = pd.concat([R[s].assign(_s=s) for s in SPLITS])
    assert (allr.groupby("patient_id")._s.nunique() == 1).all()
    assert not (set(R["train"].patient_id) & held) and not (set(R["validation"].patient_id) & held)
    assert not (set(A["train"].patient_id) & held)


def test_exact_duplicate_sentences_do_not_cross_splits(cfg):
    A = load_action(cfg)
    norm = lambda s: " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())  # noqa: E731
    sets = {s: {norm(t) for t in A[s].text} for s in SPLITS}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["validation"]) and not (sets["validation"] & sets["test"])


def test_synthetic_template_examples_are_train_only_and_marked(cfg):
    aug = load_action(cfg)["synthetic_train_augmentation"]
    assert (aug.label_tier == "synthetic").all() and (aug.split == "train").all()
