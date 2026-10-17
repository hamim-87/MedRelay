"""Dataset loading, integrity validation and replay-stream construction.

Inputs are the gzip-CSV exports of the MedRelay dataset-engineering pipeline in data/source/. The pipeline's Parquet outputs
cannot be read on s390x (no pyarrow wheel), so the same tables were exported once to portable CSV and their origin checksums
are kept in data/source/reports/SOURCE_PARQUET_SHA256.json.

This module never fabricates data. `build_stream` re-derives the replay stream from the generator's event tables so that every
payload carries what a live consumer needs (chain metadata, all patient encounters) and applies deterministic delivery
imperfections (late, duplicate, out-of-order).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "validation", "test")


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg = yaml.safe_load(open(Path(path) if path else ROOT / "config.yaml"))
    cfg["mode"] = os.environ.get("MEDRELAY_MODE", cfg["mode"])
    cfg["_root"] = str(ROOT)
    return cfg


def path_of(cfg: dict, key: str, *parts: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p.joinpath(*parts)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def stable_hash(s: str, salt: str = "") -> int:
    return int(hashlib.sha256((salt + s).encode()).hexdigest()[:16], 16)


def _json_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: json.loads(x) if isinstance(x, str) and x[:1] in "[{" else x)
    return df


# --------------------------------------------------------------------------- tables
def read_source(cfg: dict, rel: str, json_cols=()) -> pd.DataFrame:
    p = path_of(cfg, "source", rel)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - upload the dataset-pipeline export to data/source/ (see docs/linuxone_setup.md)")
    return _json_cols(pd.read_csv(p, low_memory=False), json_cols)


def load_action(cfg: dict) -> dict[str, pd.DataFrame]:
    out = {sp: read_source(cfg, f"processed/action_classification/{sp}.csv.gz") for sp in SPLITS}
    out["synthetic_train_augmentation"] = read_source(cfg, "processed/action_classification/synthetic_train_augmentation.csv.gz")
    return out


def load_risk(cfg: dict, binary_only: bool = False) -> dict[str, pd.DataFrame]:
    out = {sp: read_source(cfg, f"processed/followup_risk/features_{sp}.csv.gz") for sp in SPLITS}
    if binary_only:
        out = {k: v[v.label_binary.notna()].assign(label_binary=lambda d: d.label_binary.astype(int)) for k, v in out.items()}
    return out


def load_risk_hidden(cfg: dict) -> pd.DataFrame:
    return read_source(cfg, "processed/followup_risk/ground_truth_hidden.csv.gz")


def load_noshow(cfg: dict) -> pd.DataFrame:
    return read_source(cfg, "processed/appointment_noshow/all.csv.gz")


def load_interim(cfg: dict) -> dict[str, pd.DataFrame]:
    d = {
        "patients": read_source(cfg, "interim/patients.csv.gz"), "encounters": read_source(cfg, "interim/encounters.csv.gz"),
        "obligations": read_source(cfg, "interim/obligations.csv.gz", json_cols=["depends_on", "evidence_span"]),
        "events": read_source(cfg, "interim/events.csv.gz"), "links": read_source(cfg, "interim/evidence_links.csv.gz"),
        "ground_truth": read_source(cfg, "interim/ground_truth.csv.gz"),
    }
    return d


def load_graph_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return read_source(cfg, "processed/evidence_graph/nodes.csv.gz"), read_source(cfg, "processed/evidence_graph/edges.csv.gz")


def subset_patients(cfg: dict, patients: pd.DataFrame) -> set[str] | None:
    """Fast mode: deterministic per-split patient subset (hash order). Full mode: None (= everyone)."""
    spec = cfg["data"][cfg["mode"]]["patients_per_split"]
    if not spec:
        return None
    keep: set[str] = set()
    for sp, n in spec.items():
        ids = patients.loc[patients.split == sp, "patient_id"].tolist()
        ids.sort(key=lambda i: stable_hash(i, "fast_subset"))
        keep.update(ids[:n])
    return keep


# --------------------------------------------------------------------------- replay stream
def _fmt(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_stream(cfg: dict, force: bool = False) -> dict[str, Path]:
    """Derive the replay stream (+ separate ground truth) from the generator's tables. Deterministic for a fixed seed."""
    out_dir = path_of(cfg, "processed")
    sp = out_dir / "event_stream.jsonl.gz"
    gp = out_dir / "stream_ground_truth.csv.gz"
    ap = out_dir / "stream_delivery_anomalies.csv.gz"
    if sp.exists() and gp.exists() and not force:
        return {"stream": sp, "ground_truth": gp, "anomalies": ap}
    T = load_interim(cfg)
    obl, ev, gt, enc = T["obligations"], T["events"], T["ground_truth"], T["encounters"]
    seed = cfg["seed"]
    sc = cfg["stream"]
    recs: list[dict] = []
    pat_split = dict(zip(obl.patient_id, obl.split))
    chain_len = obl.groupby("chain_id").size().to_dict()

    def add(event_id, etype, ts, patient_id, encounter_id, obligation_id, payload, provenance, simulated, split, source, tick=False):
        r = np.random.default_rng(stable_hash(event_id, f"stream{seed}"))
        t = pd.Timestamp(ts)
        anomalies = []
        if tick:
            ing = t
        else:
            late_p = sc["late_obligation_p"] if etype == "obligation_created" else sc["late_p"]
            if r.random() < late_p:
                ing = t + pd.Timedelta(hours=float(r.uniform(*sc["late_hours"])))
                anomalies.append("late")
            else:
                ing = t + pd.Timedelta(seconds=float(np.exp(np.log(20) + 0.8 * r.standard_normal())))
        base = dict(event_id=event_id, patient_id=patient_id, encounter_id=encounter_id, obligation_id=obligation_id, event_time_utc=_fmt(t),
                    event_type=etype, payload=payload, provenance=provenance, is_simulated=bool(simulated), split=split, source_dataset=source)
        recs.append(dict(base, ingest_time_utc=_fmt(ing), delivery_attempt=1, _anom=anomalies))
        if not tick and r.random() < sc["dup_p"]:
            recs.append(dict(base, ingest_time_utc=_fmt(ing + pd.Timedelta(seconds=float(r.uniform(1, 600)))), delivery_attempt=2, _anom=["duplicate"]))

    # every encounter of every patient with an obligation (encounter-based features need the complete list)
    for r in enc[enc.patient_id.isin(set(obl.patient_id))].itertuples():
        add(f"{r.encounter_id}:start", "encounter_start", r.event_time_utc, r.patient_id, r.encounter_id, None,
            {"encounter_type": None if pd.isna(r.encounter_type) else r.encounter_type, "department": None if pd.isna(r.department) else r.department,
             "time_precision": r.time_precision, "time_imputed": bool(r.time_imputed)}, "documented_source", False, pat_split[r.patient_id], r.source_dataset)
    for r in obl.itertuples():
        documented0 = r.chain_position == 0 and r.seed_kind in ("documented_lab_result", "structured_order", "documented_careplan", "text_extracted")
        add(f"{r.obligation_id}:created", "obligation_created", r.created_time_utc, r.patient_id, r.encounter_id if isinstance(r.encounter_id, str) else None,
            r.obligation_id,
            {"category": r.category, "step_type": r.step_type, "description": r.description, "responsible_role": r.responsible_role,
             "due_time_utc": r.due_time_utc, "depends_on": list(r.depends_on), "extraction_confidence": float(r.extraction_confidence),
             "label_tier": r.label_tier, "chain_id": r.chain_id, "chain_position": int(r.chain_position), "chain_length": int(chain_len[r.chain_id]),
             "source_dataset": r.source_dataset},
            "extracted_rule" if r.seed_kind == "text_extracted" and r.chain_position == 0 else ("documented_source" if documented0 else "simulated"),
            not documented0, r.split, r.source_dataset)
        add(f"{r.obligation_id}:deadline", "deadline_reached", r.due_time_utc, r.patient_id, r.encounter_id if isinstance(r.encounter_id, str) else None,
            r.obligation_id, {"due_time_utc": r.due_time_utc}, "simulated", True, r.split, r.source_dataset, tick=True)
    split_of = dict(zip(obl.obligation_id, obl.split))
    src_of = dict(zip(obl.obligation_id, obl.source_dataset))
    for r in ev.itertuples():
        add(r.event_id, r.event_type, r.timestamp_utc, r.patient_id, r.encounter_id if isinstance(r.encounter_id, str) else None, r.obligation_id,
            {"status": r.status, "evidence_source": r.evidence_source, "verification": r.verification,
             "details": json.loads(r.payload) if isinstance(r.payload, str) else {}, "source_event_id": None if pd.isna(r.source_event_id) else r.source_event_id},
            r.provenance, r.provenance == "simulated", split_of[r.obligation_id], src_of[r.obligation_id])
    df = pd.DataFrame(recs)
    df["_t"] = pd.to_datetime(df.ingest_time_utc, utc=True)
    df = df.sort_values(["_t", "event_id", "delivery_attempt"], kind="mergesort").reset_index(drop=True)
    df["stream_seq"] = np.arange(len(df))
    df["delivery_id"] = df.event_id + "#" + df.delivery_attempt.astype(str)
    cols = ["stream_seq", "delivery_id", "event_id", "delivery_attempt", "patient_id", "encounter_id", "obligation_id", "event_time_utc",
            "ingest_time_utc", "event_type", "payload", "provenance", "is_simulated", "split", "source_dataset"]
    with gzip.open(sp, "wt") as f:
        for rec in df[cols].to_dict("records"):
            rec = {k: (None if (not isinstance(v, (dict, list)) and pd.isna(v)) else v) for k, v in rec.items()}
            f.write(json.dumps(rec, default=str) + "\n")
    g = gt.merge(obl[["obligation_id", "category", "step_type"]], on="obligation_id")
    g.to_csv(gp, index=False, compression="gzip")
    df[df._anom.map(len) > 0][["delivery_id", "event_id", "event_type", "obligation_id", "event_time_utc", "ingest_time_utc"]].assign(
        anomalies=df[df._anom.map(len) > 0]._anom.map("|".join)).to_csv(ap, index=False, compression="gzip")
    return {"stream": sp, "ground_truth": gp, "anomalies": ap}


def iter_stream(cfg: dict, patients: set[str] | None = None, splits: tuple[str, ...] | None = None, limit: int | None = None):
    """Yield stream deliveries in ingest order (the order a live service receives them)."""
    p = build_stream(cfg)["stream"]
    n = 0
    with gzip.open(p, "rt") as f:
        for line in f:
            ev = json.loads(line)
            if patients is not None and ev["patient_id"] not in patients:
                continue
            if splits is not None and ev["split"] not in splits:
                continue
            yield ev
            n += 1
            if limit and n >= limit:
                return


def load_stream_df(cfg: dict, patients: set[str] | None = None, splits=None) -> pd.DataFrame:
    return pd.DataFrame(list(iter_stream(cfg, patients, splits)))


def load_stream_truth(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(build_stream(cfg)["ground_truth"])


# --------------------------------------------------------------------------- validation
def validate_datasets(cfg: dict, write: bool = True) -> dict:
    """Schema / split / leakage / reference checks on the loaded datasets. Returns a report; `hard_failures` must be empty."""
    rep: dict = {"mode": cfg["mode"], "checks": {}, "hard_failures": [], "warnings": []}

    def chk(name, ok, detail, hard=True):
        rep["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            (rep["hard_failures"] if hard else rep["warnings"]).append(f"{name}: {detail}")

    T = load_interim(cfg)
    A = load_action(cfg)
    R = load_risk(cfg)
    NS = load_noshow(cfg)
    nodes, edges = load_graph_tables(cfg)
    P, E, O, V, G = T["patients"], T["encounters"], T["obligations"], T["events"], T["ground_truth"]
    # -- schemas
    need = {"obligations": ["obligation_id", "patient_id", "category", "step_type", "responsible_role", "created_time_utc", "due_time_utc", "depends_on",
                            "chain_id", "chain_position", "split", "label_tier"],
            "events": ["event_id", "patient_id", "obligation_id", "event_type", "timestamp_utc", "provenance", "verification"],
            "patients": ["patient_id", "source_dataset", "split", "official_split", "is_pseudo_patient"],
            "encounters": ["encounter_id", "patient_id", "event_time_utc", "time_precision", "time_imputed"]}
    miss = {t: [c for c in cols if c not in T[t].columns] for t, cols in need.items()}
    chk("schema_interim", not any(miss.values()), {k: v for k, v in miss.items() if v} or "all required columns present")
    spec = json.load(open(path_of(cfg, "source", "processed/followup_risk/feature_spec.json")))
    missf = [c for c in spec["feature_columns"] + spec["meta_columns"] + spec["label_columns"] if c not in R["train"].columns]
    chk("schema_risk_features", not missf, missf or f"{len(spec['feature_columns'])} features + meta/labels present")
    chk("schema_action", all(c in A["train"].columns for c in ["text", "labels_str", "label_tier", "label_method", "group_id", "split"]), "text/labels/tier/method/group present")
    chk("schema_noshow", all(c in NS.columns for c in ["appointment_id", "patient_key", "no_show", "wait_days", "split"]), "appointment columns present")
    # -- missing values & duplicates & keys
    nulls = {t: [c for c in need[t] if T[t][c].isna().any() and c not in ("official_split",)] for t in need}
    chk("missing_required", not any(nulls.values()), {k: v for k, v in nulls.items() if v} or "no nulls in required columns")
    dups = {"obligations": int(O.obligation_id.duplicated().sum()), "events": int(V.event_id.duplicated().sum()), "patients": int(P.patient_id.duplicated().sum()),
            "action": int(pd.concat([A[s] for s in SPLITS]).example_id.duplicated().sum()), "noshow": int(NS.appointment_id.duplicated().sum()),
            "nodes": int(nodes.node_id.duplicated().sum()), "edges": int(edges.edge_id.duplicated().sum())}
    chk("unique_primary_keys", not any(dups.values()), dups)
    # -- foreign keys
    dep = {d for ds in O.depends_on for d in ds}
    fk = {"obligations->patients": int((~O.patient_id.isin(P.patient_id)).sum()), "events->obligations": int((~V.obligation_id.isin(O.obligation_id)).sum()),
          "depends_on->obligations": int(len(dep - set(O.obligation_id))), "encounters->patients": int((~E.patient_id.isin(P.patient_id)).sum()),
          "edges->nodes": int((~(edges.src_id.isin(nodes.node_id) & edges.dst_id.isin(nodes.node_id))).sum()),
          "risk->obligations": int(sum((~R[s].obligation_id.isin(O.obligation_id)).sum() for s in SPLITS))}
    chk("foreign_keys", not any(fk.values()), fk)
    # -- splits & patient grouping
    sp = dict(zip(P.patient_id, P.split))
    bad_split = {"obligations": int((O.patient_id.map(sp) != O.split).sum())}
    for s in SPLITS:
        bad_split[f"risk_{s}"] = int((R[s].patient_id.map(sp) != s).sum())
        bad_split[f"action_{s}"] = int((A[s].patient_id.map(sp) != s).sum()) if A[s].patient_id.isin(P.patient_id).all() else 0
    chk("split_assignment_consistent", not any(bad_split.values()), bad_split)
    multi = {"action": int(pd.concat([A[s] for s in SPLITS]).groupby("group_id").split.nunique().gt(1).sum()), "noshow_patients": int(NS.groupby("patient_key").split.nunique().gt(1).sum()),
             "risk": int(pd.concat([R[s] for s in SPLITS]).groupby("patient_id").split.nunique().gt(1).sum())}
    chk("patient_level_independence", not any(multi.values()), multi)
    sh_ho = set(P[P.official_split == "heldout"].patient_id)
    chk("sh_heldout_only_in_test", not any(R[s].patient_id.isin(sh_ho).any() for s in ("train", "validation")) and not A["train"].patient_id.isin(sh_ho).any(),
        f"{len(sh_ho)} held-out Synthetic Hospital patients are test-only")
    act_all = pd.concat([A[s].assign(_s=s) for s in SPLITS])
    act_all["_n"] = act_all.text.str.lower().str.replace(r"[^a-z0-9 ]+", "", regex=True).str.split().str.join(" ")
    chk("action_exact_duplicates_across_splits", int(act_all.groupby("_n")._s.nunique().gt(1).sum()) == 0, "no normalized sentence appears in >1 split")
    # -- label provenance / balance / real-vs-synthetic distinction
    tiers = pd.concat([A[s] for s in SPLITS]).label_tier.value_counts().to_dict()
    rep["label_provenance"] = {
        "action_label_tiers": tiers, "action_label_methods": pd.concat([A[s] for s in SPLITS]).label_method.value_counts().to_dict(),
        "action_sources": pd.concat([A[s] for s in SPLITS]).source_dataset.value_counts().to_dict(),
        "synthetic_template_examples(train-only)": int(len(A["synthetic_train_augmentation"])),
        "risk_labels": "SIMULATED follow-up outcomes (generated scenarios); not clinically verified",
        "noshow_labels": "REAL appointment attendance (Kaggle, de-identified); not follow-up completion",
        "gold_human_reviewed_action_labels": int(tiers.get("gold", 0))}
    if tiers.get("gold", 0) == 0:
        rep["warnings"].append("no human-reviewed (gold) action labels: extraction/classification metrics against silver or developer-written sets are NOT clinical accuracy")
    rep["class_balance"] = {
        "action": {s: {c: int(A[s].get(f"y_{c}", pd.Series(dtype=int)).sum()) for c in ["appointment_referral", "laboratory_test", "imaging", "procedure", "medication_action",
                                                                                  "patient_communication", "followup_review", "other_action", "no_action"]} for s in SPLITS},
        "risk_binary": {s: {str(k): int(v) for k, v in R[s].label_binary.dropna().astype(int).value_counts().items()} for s in SPLITS},
        "risk_outcomes": {s: R[s].outcome_label.value_counts().to_dict() for s in SPLITS},
        "noshow": {s: {"n": int((NS.split == s).sum()), "rate": round(float(NS[NS.split == s].no_show.mean()), 4)} for s in SPLITS}}
    # -- temporal leakage
    allr = pd.concat([R[s] for s in SPLITS])
    chk("assessment_before_deadline", bool((pd.to_datetime(allr.assessment_time_utc) < pd.to_datetime(allr.deadline_utc)).all()), "assessment_time < deadline for every risk row")
    comp = V.merge(O[["obligation_id", "step_type"]], on="obligation_id")
    from .ontology import COMPLETION_EVENT
    comp = comp[comp.event_type == comp.step_type.map(COMPLETION_EVENT)].groupby("obligation_id").timestamp_utc.min()
    a = allr.set_index("obligation_id").assessment_time_utc
    leaked = int(sum((o in comp.index) and (pd.Timestamp(comp[o]) <= pd.Timestamp(t)) for o, t in a.items()))
    chk("no_completed_obligation_at_assessment", leaked == 0, f"{leaked} risk rows already had completion evidence at assessment time")
    forb = sorted(set(spec["forbidden_in_model_input"]) & set(spec["feature_columns"]))
    chk("no_forbidden_feature_columns", not forb, forb or "no generator flags / final status / future timestamps among features")
    # -- graph structure
    import networkx as nx
    dg = nx.DiGraph()
    dg.add_edges_from(edges[edges.edge_type == "DEPENDS_ON"][["src_id", "dst_id"]].itertuples(index=False, name=None))
    chk("dependency_graph_acyclic", nx.is_directed_acyclic_graph(dg), f"{dg.number_of_nodes()} obligations / {dg.number_of_edges()} dependency edges")
    # -- source data hashes
    srcsha = path_of(cfg, "source", "reports/SOURCE_PARQUET_SHA256.json")
    rep["source_parquet_checksums_recorded"] = len(json.load(open(srcsha))) if srcsha.exists() else 0
    rep["counts"] = {"patients": len(P), "encounters": len(E), "obligations": len(O), "events": len(V), "risk_rows": len(allr),
                     "action_examples": int(sum(len(A[s]) for s in SPLITS)), "noshow_rows": len(NS), "graph_nodes": len(nodes), "graph_edges": len(edges)}
    rep["data_files_sha256"] = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(path_of(cfg, "source").rglob("*.gz"))}
    if write:
        (path_of(cfg, "results") / "data_validation.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep
