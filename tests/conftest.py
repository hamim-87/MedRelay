import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEDRELAY_MODE", "fast")


@pytest.fixture(scope="session")
def cfg():
    from src.data_loader import load_config
    return load_config()


def mk_ob(oid, step, created, deadline, deps=(), chain="c1", pos=0, n=1, pid="p1", enc=None):
    from src.evidence_graph import iso
    return {"event_id": f"{oid}:created", "patient_id": pid, "encounter_id": enc, "obligation_id": oid, "event_time_utc": iso(created), "ingest_time_utc": iso(created + 5),
            "event_type": "obligation_created", "provenance": "simulated", "is_simulated": True,
            "payload": {"category": "laboratory_test", "step_type": step, "description": step, "responsible_role": "patient", "due_time_utc": iso(deadline), "depends_on": list(deps),
                        "extraction_confidence": 0.9, "label_tier": "silver", "chain_id": chain, "chain_position": pos, "chain_length": n, "source_dataset": "test"}}


def mk_ev(eid, oid, etype, t_event, t_ingest=None, ver="verified", pid="p1", details=None, prov="simulated"):
    from src.evidence_graph import iso
    return {"event_id": eid, "patient_id": pid, "encounter_id": None, "obligation_id": oid, "event_time_utc": iso(t_event), "ingest_time_utc": iso(t_ingest if t_ingest is not None else t_event + 30),
            "event_type": etype, "payload": {"status": "ok", "verification": ver, "details": details or {}, "evidence_source": "t", "source_event_id": None}, "provenance": prov, "is_simulated": True}


@pytest.fixture()
def T0():
    from src.evidence_graph import to_epoch
    return to_epoch("2024-03-04T09:00:00Z")
