"""Build the MedRelay pitch deck (PDF + editable PPTX) from the archived LinuxONE results.

    python build_deck.py            # writes ../IBM_Z_Datathon_2026_Pitch_Deck.{pdf,pptx}
Team names are optional: put {"name": "...", "members": ["...", "..."]} in team.json (nothing is invented when it is absent).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import charts
import facts
from deckkit import Deck, W, H

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent
TEAM = json.loads((HERE / "team.json").read_text()) if (HERE / "team.json").exists() else {"name": None, "members": []}
REPO = "github.com/hamim-87/MedRelay"

# palette
NAVY0, NAVY1 = "#060B17", "#0E1C3A"
PANEL, PANEL_LN = "#10203F", "#2A4A86"
BLUE, LBLUE, CYAN, TEAL, GREEN, AMBER, RED = "#0F62FE", "#4589FF", "#33B1FF", "#08BDBA", "#42BE65", "#F1C21B", "#FA4D56"
WHITE, MUTED, DIM = "#FFFFFF", "#B4C2DE", "#7C8FB5"

F = charts.build_all()
D = Deck()
N_SLIDES = 10


def fmt(n):
    return f"{int(n):,}"


# ------------------------------------------------------------------------------------------------ shared pieces
def panel(s, x, y, w, h, accent=None, fill=PANEL, alpha=0.95):
    D.rect(s, x, y, w, h, fill=fill, line=PANEL_LN, lw=0.8, r=9, alpha=alpha, line_alpha=0.75)
    if accent:
        D.rect(s, x + 14, y, 44, 3, fill=accent, r=1.5)


def header(s, idx, tag, title, title_size=28):
    D.bg(s, NAVY0, NAVY1)
    D.ellipse(s, 730, 6, 200, 200, fill=BLUE, alpha=0.08)
    D.ellipse(s, 790, 36, 110, 110, fill=CYAN, alpha=0.07)
    D.rect(s, 40, 31, 4, 13, fill=CYAN)
    D.text(s, 52, 29, 500, 18, tag, size=11.5, color=CYAN, bold=True)
    D.text(s, 40, 52, 880, 74, title, size=title_size, bold=True, lh=1.16)
    D.line(s, 40, 508, 920, 508, color=PANEL_LN, lw=0.7, alpha=0.7)
    D.text(s, 40, 514, 500, 14, "MedRelay  |  IBM Z Datathon 2026  |  synthetic data only, decision support for qualified human review", size=8.5, color=DIM)
    D.text(s, 840, 514, 80, 14, f"{idx} / {N_SLIDES}", size=9, color=DIM, align="r")


def chip(s, x, y, w, text, color=CYAN, size=10.5, h=22, fill=None):
    D.rect(s, x, y, w, h, fill=fill or color, line=color, lw=0.8, r=h / 2, alpha=0.14 if not fill else 1.0)
    D.text(s, x, y + (h - size * 1.2) / 2 - 0.5, w, size * 1.3, text, size=size, color=color if not fill else WHITE, bold=True, align="c")


def numdot(s, x, y, n, color=CYAN, d=26):
    D.ellipse(s, x, y, d, d, fill=color, alpha=1.0)
    D.text(s, x, y + (d - 14.5) / 2 - 0.5, d, 16, str(n), size=13, bold=True, color=NAVY0, align="c")


# ------------------------------------------------------------------------------------------------ slide 1: cover
def slide_cover():
    s = D.new(); D.bg(s, NAVY0, "#0B2250")
    # network motif (decorative geometry)
    nodes = [(700, 120), (800, 200), (880, 110), (760, 300), (860, 330), (930, 240), (690, 400), (810, 430), (900, 450), (610, 250), (640, 90), (930, 60)]
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (1, 5), (3, 6), (6, 7), (7, 8), (4, 8), (0, 9), (9, 3), (0, 10), (2, 11), (5, 8), (7, 4)]
    D.ellipse(s, 590, 50, 360, 360, fill=None, line=CYAN, lw=0.8, line_alpha=0.18)
    D.ellipse(s, 650, 110, 240, 240, fill=None, line=LBLUE, lw=0.8, line_alpha=0.22)
    for a, b in edges:
        D.line(s, nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], color=CYAN, lw=1.1, alpha=0.38)
    cols = [CYAN, BLUE, GREEN, LBLUE, CYAN, AMBER, BLUE, CYAN, GREEN, LBLUE, BLUE, CYAN]
    for (x, y), c in zip(nodes, cols):
        D.ellipse(s, x - 11, y - 11, 22, 22, fill=c, alpha=0.18); D.ellipse(s, x - 5.5, y - 5.5, 11, 11, fill=c, alpha=1.0)
    D.rect(s, 60, 52, 4, 16, fill=CYAN)
    D.text(s, 72, 51, 500, 20, "IBM Z DATATHON 2026", size=14, color=CYAN, bold=True)
    D.text(s, 72, 72, 520, 18, "Theme: Real-Time AI for Critical Decisions  |  Detect. Decide. Respond.", size=11, color=MUTED)
    D.text(s, 60, 128, 560, 100, "MedRelay", size=88, bold=True, lh=1.05)
    D.text(s, 60, 238, 540, 84, "Catching the clinical follow-ups that fall through the cracks, as events unfold.", size=25, bold=True, color="#DCE8FF", lh=1.2)
    D.text(s, 60, 332, 520, 48, "Evidence-aware tracking of follow-up obligations: a local LLM, a temporal evidence graph and calibrated ML, running end to end on IBM LinuxONE.", size=14, color=MUTED, lh=1.3)
    chip(s, 60, 388, 120, "s390x big-endian", CYAN, 10); chip(s, 188, 388, 100, "z/VM 7.3", CYAN, 10); chip(s, 296, 388, 120, "NNPA verified", GREEN, 10); chip(s, 424, 388, 120, "Granite on-box", LBLUE, 10)
    y = 428
    if TEAM.get("name"):
        D.text(s, 60, y, 560, 20, [[("Team ", {"color": MUTED}), (TEAM["name"], {"color": WHITE})]], size=16, bold=True); y += 24
    if TEAM.get("members"):
        D.text(s, 60, y, 700, 18, "  |  ".join(TEAM["members"]), size=11.5, color=MUTED); y += 24
    D.text(s, 60, y, 560, 16, REPO, size=12, color=CYAN); y += 20
    D.text(s, 60, y, 700, 14, "Prototype on synthetic data. Alerts are decision support for qualified human review.", size=10, color=DIM)
    s.notes = ("MedRelay finds clinical follow-up obligations in notes, tracks them against real evidence of completion, and warns early about the ones likely to be missed. "
               "It matches the 'Real-Time AI for Critical Decisions' theme: streaming events in, risk and alerts out. Everything was built and executed on an IBM LinuxONE VM.")


# ------------------------------------------------------------------------------------------------ slide 2: problem
def slide_problem():
    s = D.new(); header(s, 2, "THE PROBLEM", "Follow-ups fall through the cracks, and “no record” is not “failed”")
    cards = [("Hidden in free text", "“Repeat the CBC in 48 hours. The physician must review the result. Arrange a visit in 2 weeks.” Obligations live in notes, not in structured fields."),
             ("Chained and fragile", "A result must be reviewed before the patient is told. One missed step silently blocks every task after it."),
             ("Documentation lags reality", "Tasks get done but are recorded late, or never. Tools that read “no record” as “failed” bury staff in false alarms.")]
    for i, (t, b) in enumerate(cards):
        x = 40 + i * 300
        panel(s, x, 122, 280, 176, accent=CYAN)
        numdot(s, x + 16, 140, i + 1)
        D.text(s, x + 52, 142, 215, 24, t, size=16.5, bold=True)
        D.text(s, x + 16, 180, 252, 114, b, size=14, color=MUTED, lh=1.3)
    panel(s, 40, 312, 880, 144, accent=RED)
    D.text(s, 58, 326, 400, 18, "MEASURED IN OUR SIMULATED TEST REPLAY", size=10.5, bold=True, color=MUTED)
    D.text(s, 58, 346, 220, 58, f"{F['missing_n']} / {F['missing_n']}", size=46, bold=True, color=RED, lh=1.0)
    D.text(s, 58, 404, 225, 44, "done-but-undocumented tasks flagged overdue by a naive “no record = failed” rule", size=11, color=MUTED, lh=1.22)
    D.text(s, 300, 346, 220, 58, f"{int(F['missing_engine_confirmed_overdue'] * F['missing_n'])} / {F['missing_n']}", size=46, bold=True, color=CYAN, lh=1.0)
    D.text(s, 300, 404, 215, 44, "confirmed overdue by MedRelay (flagged unverified instead)", size=11, color=MUTED, lh=1.22)
    D.line(s, 536, 334, 536, 442, color=PANEL_LN, lw=0.8, alpha=0.8)
    D.text(s, 554, 326, 360, 18, "WHO IS AFFECTED", size=10.5, bold=True, color=MUTED)
    D.text(s, 554, 348, 356, 100, [[("Patients", {"bold": True, "color": WHITE}), " awaiting results, referrals and repeat tests"],
                                   [("Clinicians and care coordinators", {"bold": True, "color": WHITE}), " drowning in alerts they cannot trust"],
                                   [("Patient-safety teams", {"bold": True, "color": WHITE}), " who need an audit trail, not a guess"]], size=12.5, color=MUTED, lh=1.25, space=4)
    D.text(s, 40, 466, 880, 30, "Why it matters: a missed result or referral can delay diagnosis, and a flood of false alarms makes real ones easier to miss. Outcomes above are simulated from explicit event timelines; no real patient data was used for follow-up outcomes.", size=10.5, color=DIM, lh=1.25)
    s.notes = ("Three facts make follow-up tracking hard: obligations are in free text, they depend on each other, and documentation lags behind what actually happened. "
               "Naive tools treat 'no record' as 'failed'. In our injected test, 53 of 53 tasks that were done but undocumented were flagged overdue by the naive rule; MedRelay confirmed none and flagged them as unverified. Simulated data.")


# ------------------------------------------------------------------------------------------------ slide 3: solution
def slide_solution():
    s = D.new(); header(s, 3, "OUR SOLUTION", "A live, evidence-checked tracker for every follow-up")
    steps = [("EXTRACT", CYAN, "Notes become structured obligations", "A local LLM plus rules propose tasks, deadlines, roles and dependencies. Every field must cite its source text, or it is rejected."),
             ("TRACK", LBLUE, "Events become verified states", "A temporal evidence graph links tasks, dependencies and events. Deterministic rules decide: pending, completed, overdue, unverified, contradictory."),
             ("WARN", GREEN, "Forecasts become explained alerts", "Calibrated ML predicts which tasks will miss their deadline. Alerts say why, state their uncertainty, and go to a human.")]
    for i, (k, c, t, b) in enumerate(steps):
        x = 40 + i * 300
        panel(s, x, 124, 280, 214, accent=c)
        D.text(s, x + 16, 140, 200, 16, f"STEP {i + 1}  |  {k}", size=11.5, bold=True, color=c)
        D.text(s, x + 16, 162, 250, 46, t, size=17.5, bold=True, lh=1.15)
        D.text(s, x + 16, 214, 252, 120, b, size=13.5, color=MUTED, lh=1.3)
        if i < 2:
            D.line(s, x + 282, 231, x + 298, 231, color=CYAN, lw=2.2, arrow=True)
    D.text(s, 40, 356, 600, 16, "WHAT MAKES IT DIFFERENT", size=10.5, bold=True, color=MUTED)
    diffs = [("The LLM never sets a state", "It only proposes; rules decide."), ("No evidence is not failure", "Unverified and censored tasks stay visible, never “failed”."), ("Dependency and time aware", "A blocked review is flagged before the deadline.")]
    for i, (t, b) in enumerate(diffs):
        x = 40 + i * 300
        D.rect(s, x, 376, 280, 62, fill=BLUE, alpha=0.16, line=LBLUE, lw=0.8, r=8, line_alpha=0.7)
        D.text(s, x + 14, 384, 255, 20, t, size=13.5, bold=True)
        D.text(s, x + 14, 406, 255, 28, b, size=11, color=MUTED, lh=1.2)
    D.text(s, 40, 452, 880, 40, [[("Benefit for users: ", {"bold": True, "color": WHITE}), "earlier warnings (median lead 72 h in our test), fewer false alarms, a full audit trail for every state change, and notes that never leave the server."]],
           size=12.5, color=MUTED, lh=1.3)
    s.notes = ("Extract, track, warn. The point judges should remember: the LLM only proposes evidence-grounded records; deterministic rules decide the state; absence of evidence is 'unverified', never 'failed'.")


# ------------------------------------------------------------------------------------------------ slide 4: architecture
def box(s, x, y, w, h, title, sub, color=CYAN, tsize=12, ssize=10.2, mod=None):
    D.rect(s, x, y, w, h, fill="#14294F", line=color, lw=1.1, r=8, alpha=1.0, line_alpha=0.9)
    D.rect(s, x, y + 8, 3.5, h - 16, fill=color)
    D.text(s, x + 11, y + 7, w - 16, 32, title, size=tsize, bold=True, lh=1.12)
    D.text(s, x + 11, y + 40, w - 16, h - 44, sub, size=ssize, color=MUTED, lh=1.22)
    if mod:
        D.text(s, x + 11, y + h - 15, w - 14, 12, mod, size=8, color=DIM, family="mono")


def slide_arch():
    s = D.new(); header(s, 4, "SYSTEM ARCHITECTURE", "Everything runs inside one LinuxONE VM")
    # LinuxONE boundary
    D.rect(s, 176, 116, 640, 356, fill="#0A1630", line=CYAN, lw=1.6, r=14, alpha=0.9, dash=[6, 4], line_alpha=0.95)
    D.rect(s, 192, 106, 330, 20, fill=CYAN, r=10)
    D.text(s, 192, 109, 330, 16, "IBM LinuxONE VM  |  s390x  |  z/VM 7.3  |  2 vCPU", size=10.2, bold=True, color=NAVY0, align="c")
    cw, gap = 128, 32
    cx = [30, 190, 350, 510, 670, 830]
    r1, r2, bh = 150, 306, 112
    # inputs
    D.text(s, 30, 128, 130, 14, "INPUTS", size=9.5, bold=True, color=MUTED)
    box(s, cx[0], r1, cw, bh, "Clinical notes", "synthetic discharge and visit notes", color=LBLUE)
    box(s, cx[0], r2, cw, bh, "Event stream", "results, acknowledgements, visits; late, duplicate, out-of-order", color=LBLUE)
    # row 1
    box(s, cx[1], r1, cw, bh, "Extractor", "rules + Granite 4.0 350M (llama.cpp, VXE); Pydantic validator", color=CYAN, mod="clinical_extractor")
    box(s, cx[2], r1, cw, bh, "Evidence graph", "tasks, dependencies, events; state rules R1 to R7", color=CYAN, mod="evidence_graph")
    box(s, cx[3], r1, cw, bh, "Risk model", "49 features, XGBoost, calibrated probability", color=CYAN, mod="risk_predictor")
    box(s, cx[4], r1, cw, bh, "Alert engine", "grounded text, stated uncertainty, priority", color=CYAN, mod="alert_engine")
    # row 2
    box(s, cx[1], r2, cw, bh, "Event processor", "idempotent, event vs ingest time, timers, micro-batches", color=CYAN, mod="event_processor")
    box(s, cx[2], r2, cw, bh, "SQLite (WAL)", "events, transitions, alerts; rebuild from log", color=CYAN)
    box(s, cx[3], r2, cw, bh, "Priority score", "risk x urgency x unresolved dependencies", color=CYAN)
    box(s, cx[4], r2, cw, bh, "Audit trail", "every state change with rule and evidence", color=CYAN)
    # outputs
    D.text(s, 830, 128, 100, 14, "OUTPUTS", size=9.5, bold=True, color=MUTED)
    box(s, cx[5], r1, 100, bh, "Alert feed", "for human review", color=GREEN, tsize=11.5)
    box(s, cx[5], r2, 100, bh, "Notebook demo", "live storyboard", color=GREEN, tsize=11.5)
    # arrows row 1
    A = dict(color=CYAN, lw=1.8, arrow=True)
    D.line(s, cx[0] + cw, r1 + 40, cx[1], r1 + 40, **A)
    for i in (1, 2, 3):
        D.line(s, cx[i] + cw, r1 + 40, cx[i + 1], r1 + 40, **A)
    D.line(s, cx[4] + cw, r1 + 40, cx[5], r1 + 40, **A)
    # event stream -> event processor -> graph / sqlite
    D.line(s, cx[0] + cw, r2 + 40, cx[1], r2 + 40, **A)
    D.line(s, cx[1] + cw, r2 + 80, cx[2], r2 + 80, **A)
    D.line(s, cx[1] + cw, r2 + 30, 334, r2 + 30, color=CYAN, lw=1.8)
    D.line(s, 334, r2 + 30, 334, r1 + 84, color=CYAN, lw=1.8)
    D.line(s, 334, r1 + 84, cx[2], r1 + 84, **A)
    # risk -> priority -> alert
    D.line(s, cx[3] + 40, r1 + bh, cx[3] + 40, r2, **A)
    D.line(s, cx[3] + cw, r2 + 30, 654, r2 + 30, color=CYAN, lw=1.8)
    D.line(s, 654, r2 + 30, 654, r1 + 84, color=CYAN, lw=1.8)
    D.line(s, 654, r1 + 84, cx[4], r1 + 84, **A)
    # alert -> audit ; outputs
    D.line(s, cx[4] + 64, r1 + bh, cx[4] + 64, r2, **A)
    D.line(s, cx[4] + cw, r2 + 40, cx[5], r2 + 40, **A)
    D.text(s, 30, 432, 140, 36, "LLM output is validated and grounded; it never sets a state.", size=9.6, color=AMBER, bold=True, lh=1.2)
    D.text(s, 40, 480, 880, 22, "Every box is code in the repository (src/) and was executed on the VM: clean-kernel run of the notebook, 56 code cells, 0 errors.", size=11, color=MUTED, align="c")
    s.notes = ("Left to right: notes and the event stream enter the LinuxONE VM. The extractor proposes validated obligations; the event processor ingests events idempotently and keeps SQLite as the record of truth. "
               "The evidence graph derives states with deterministic rules; the risk model forecasts failure probability; the alert engine produces grounded text. Priority combines risk, urgency and unresolved dependencies.")


# ------------------------------------------------------------------------------------------------ slide 5: data & ML
def slide_data_ml():
    s = D.new(); header(s, 5, "DATASET & MACHINE LEARNING", "Leakage-safe by construction, honest about labels")
    c = F["counts"]
    tiles = [(fmt(c["patients"]), "patients"), (fmt(c["obligations"]), "follow-up obligations"), (fmt(c["events"]), "source events"), (fmt(c["action_examples"]), "action examples (weak labels)"), (fmt(c["noshow_rows"]), "real appointments (no-show)")]
    tw = 168
    for i, (n, l) in enumerate(tiles):
        x = 40 + i * (tw + 10)
        panel(s, x, 118, tw, 68)
        D.text(s, x + 10, 124, tw - 16, 34, n, size=24, bold=True, color=CYAN, lh=1.05)
        D.text(s, x + 10, 158, tw - 14, 26, l, size=10, color=MUTED, lh=1.15)
    D.text(s, 40, 192, 880, 16, "Sources: Synthea, synthetic_hospital, MTS-Dialog, ACI-Bench (all synthetic or simulated) + Kaggle appointment no-shows (real, de-identified). Follow-up outcomes are simulated from event timelines.", size=9.6, color=DIM)
    # pipeline rows
    def step(x, y, w, t, sub, color):
        D.rect(s, x, y, w, 70, fill="#14294F", line=color, lw=1.0, r=8, line_alpha=0.9)
        D.text(s, x + 9, y + 7, w - 14, 18, t, size=11.5, bold=True, lh=1.1)
        D.text(s, x + 9, y + 27, w - 14, 42, sub, size=10, color=MUTED, lh=1.2)
    D.text(s, 40, 218, 300, 14, "DATA PIPELINE", size=10, bold=True, color=MUTED)
    sw, sg = 202, 24
    row1 = [("Canonical tables", "patients, encounters, notes, source events; SHA-256 checked"), ("Scenario simulator", "event-level timelines with documented gaps, late and missing evidence"),
            ("Event stream", "late, duplicate and out-of-order deliveries; outcomes kept in a hidden file"), ("Evidence graph", "103,627 nodes, 223,925 edges; dependency graph is acyclic")]
    for i, (t, b) in enumerate(row1):
        x = 40 + i * (sw + sg)
        step(x, 236, sw, t, b, LBLUE)
        if i < 3: D.line(s, x + sw + 2, 271, x + sw + sg - 2, 271, color=LBLUE, lw=1.8, arrow=True)
    D.text(s, 40, 318, 300, 14, "ML PIPELINE", size=10, bold=True, color=MUTED)
    row2 = [("49 features, 6 groups", "task, own events, workload, graph dependency, graph temporal, evidence; strict event-time cutoff"), ("Dense snapshots", "8 assessment points per task, only while the task is still open"),
            ("XGBoost + calibration", "train, validation and test split by patient; calibrated probabilities"), ("Validation thresholds", "alert threshold fixed on validation; applied once to test")]
    for i, (t, b) in enumerate(row2):
        x = 40 + i * (sw + sg)
        step(x, 336, sw, t, b, CYAN)
        if i < 3: D.line(s, x + sw + 2, 371, x + sw + sg - 2, 371, color=CYAN, lw=1.8, arrow=True)
    pts = [("No leakage", "features see only events ingested before the assessment time"), ("No tuning on test", "thresholds and hyper-parameters chosen on validation only"), ("Strong baselines", "deadline-only rule, independent-task ML, patient-grouped bootstrap CIs")]
    for i, (t, b) in enumerate(pts):
        x = 40 + i * 300
        D.text(s, x, 418, 285, 18, t, size=12.5, bold=True, color=WHITE)
        D.text(s, x, 436, 280, 34, b, size=10.8, color=MUTED, lh=1.22)
    chip(s, 40, 474, 330, "0 human-reviewed gold labels | outcomes simulated", AMBER, 10, h=22)
    D.text(s, 384, 477, 536, 16, "Scores describe the simulator, not clinical accuracy. Real validation needs clinician-reviewed labels.", size=10, color=DIM)
    s.notes = ("The data are synthetic or simulated, plus one real de-identified no-show dataset used as a secondary proof of concept. Key engineering decisions: strict event-time cutoffs, dense snapshot training, thresholds chosen on validation only, and patient-grouped bootstrap intervals. "
               "We state clearly that there are no human-reviewed gold labels.")


# ------------------------------------------------------------------------------------------------ slide 6: LinuxONE
def slide_linuxone():
    s = D.new(); header(s, 6, "WHY IBM Z / LINUXONE", "The whole stack, executed on s390x")
    hw = F["hw"]; mm = F["mm"][F["mm"].M == 1024].iloc[0]; thr = F["thr"]; q = F["queue"].set_index("scenario")
    panel(s, 40, 118, 420, 218, accent=CYAN)
    D.text(s, 56, 130, 380, 16, "VERIFIED ON THE VM", size=10.5, bold=True, color=CYAN)
    rows = [("Platform", f"{hw['uname_m']}, {hw['byteorder']}-endian, {hw['hypervisor']}"), ("Resources", f"{hw['nproc']} vCPU, {hw['memory_total_gb']:.1f} GB RAM"),
            ("CPU facilities", "vector (VXE2) and NNPA (facility 165)"), ("Native stack", "native Python 3.12 environment; no container runtime"),
            ("LLM runtime", "llama.cpp (VXE) + big-endian Granite 4.0 GGUF"), ("Local inference", f"{F['llm_model'].replace('.be.gguf', '')}, {F['llm_tps']:.1f} tokens/s")]
    for i, (k, v) in enumerate(rows):
        y = 154 + i * 30
        D.text(s, 56, y, 100, 28, k, size=10.5, color=MUTED, bold=True, lh=1.15)
        D.text(s, 150, y, 304, 28, v, size=11.2, color=WHITE, lh=1.18)
    panel(s, 476, 118, 444, 218, accent=GREEN)
    D.text(s, 492, 130, 410, 16, f"NNPA DEVICE TEST (zDNN {F['zdnn']})", size=10.5, bold=True, color=GREEN)
    D.image(s, str(HERE / "assets" / "chart_nnpa.png"), 486, 148, 424, 152)
    D.text(s, 492, 300, 420, 32, f"{mm['speedup_total_vs_numpy']:.0f}x faster than NumPy BLAS on this VM incl. data conversion; max error {mm['max_abs_err']:.3f} on values up to {mm['max_abs_ref']:.1f} (16-bit arithmetic).", size=10, color=MUTED, lh=1.2)
    kp = [(f"{thr[thr.config.str.startswith('graph+states')].events_per_s.max() / 1000:.1f} k", "events/s", "graph + state engine, in-memory"),
          (f"{thr[thr.config.str.contains('ML risk')].events_per_s.min():.0f} to {thr[thr.config.str.contains('ML risk')].events_per_s.max():.0f}", "events/s", "states + alerts + ML risk, SQLite WAL / in-memory"),
          (f"{q.loc['2x capacity, block (backpressure)', 'queue_latency_ms_p99']:.0f} ms", "p99 queue latency", "at 2x measured capacity, backpressure, 0 lost"),
          (f"{q.loc['2x capacity, drop (bounded queue 200)', 'loss_rate'] * 100:.0f}% / {q.loc['4x capacity, drop (bounded queue 200)', 'loss_rate'] * 100:.0f}%", "events shed", "bounded queue at 2x / 4x measured capacity")]
    for i, (n, u, l) in enumerate(kp):
        x = 40 + i * 223
        panel(s, x, 350, 211, 78)
        D.text(s, x + 12, 356, 190, 34, n, size=22, bold=True, color=CYAN, lh=1.05)
        D.text(s, x + 12, 386, 190, 14, u, size=10.5, bold=True, color=WHITE)
        D.text(s, x + 12, 401, 190, 26, l, size=9.2, color=MUTED, lh=1.15)
    D.rect(s, 40, 440, 880, 56, fill=AMBER, alpha=0.1, line=AMBER, lw=0.8, r=8, line_alpha=0.7)
    D.text(s, 54, 446, 852, 48, [[("What we do not claim: ", {"bold": True, "color": AMBER}), "the tree and linear risk models run on the CPU, not on the NNPA; only the zDNN microbenchmark proves NNPA execution. "
                                  "Throughput was measured on a small 2 vCPU VM and varies run to run. No security, cost or availability claims are made."],
                                 [("Why LinuxONE fits: ", {"bold": True, "color": WHITE}), "notes are processed by a local model on the same server that holds the data, and the event engine needs sustained, low-latency streaming."]], size=10.5, color=MUTED, lh=1.24, space=3)
    s.notes = ("What was actually run on LinuxONE: the full pipeline on an s390x big-endian VM under z/VM, a native Python environment, llama.cpp compiled for the z vector facility, Granite GGUF models converted to big-endian, and a direct zDNN matrix-multiply test on the NNPA. "
               "We are explicit that tree models do not use NNPA. Throughput numbers come from a 2 vCPU VM and varied between runs.")


# ------------------------------------------------------------------------------------------------ slide 7: results
def slide_results():
    s = D.new(); header(s, 7, "EXPERIMENTS & RESULTS", "Fewer false alarms, earlier warnings, honest limits")
    ew, ab = F["ew"], F["ab"]
    pw, gap = 285, 8.5
    panels = [("EARLY WARNING, >= 24 h BEFORE DEADLINE", "chart_early_warning.png",
               f"Graph model C raised {int(ew.loc['C_graph_ml', 'alerts'])} alerts at {ew.loc['C_graph_ml', 'precision'] * 100:.0f}% precision, with a {ew.loc['C_graph_ml', 'median_lead_h']:.0f} h median lead. Deadline-only A reaches 100% recall but alerts on {int(ew.loc['A_deadline_only', 'alerts'])} of {int(ew.loc['A_deadline_only', 'n_pos'] + ew.loc['A_deadline_only', 'n_neg'])} tasks."),
              ("EVIDENCE-AWARE STATES vs NAIVE RULE", "chart_engine_vs_naive.png",
               f"State accuracy at the assessment time {F['state_acc_engine']:.3f} vs {F['state_acc_naive']:.3f}; {F['state_acc_final']:.3f} once late evidence reconciles. The naive rule never recognises an unverified task."),
              ("WHAT DRIVES THE FORECAST", "chart_ablation.png",
               "Temporal features carry the signal; dependency features add a small, significant gain. Evidence features and the graph-vs-independent gap are not significant at this sample size.")]
    for i, (t, img, cap) in enumerate(panels):
        x = 40 + i * (pw + gap + 4)
        panel(s, x, 112, pw, 286)
        D.text(s, x + 12, 122, pw - 20, 14, t, size=9.4, bold=True, color=CYAN)
        D.image(s, str(HERE / "assets" / img), x + 8, 140, pw - 16, 178)
        D.text(s, x + 12, 322, pw - 24, 72, cap, size=9.6, color=MUTED, lh=1.22)
    pair = F["e3"]["paired"]
    cards = [(f"{F['c_pr_auc']:.3f}", "PR-AUC, graph model", f"95% CI {F['c_ci'][0]:.3f} to {F['c_ci'][1]:.3f}; simulated outcomes", CYAN),
             (f"{F['noshow_best']['test_pr_auc']:.3f}", "PR-AUC, real no-show data", f"vs {F['noshow_prev']:.3f} base rate; secondary proof of concept", GREEN),
             (f"{pair['C_minus_B']['diff']:+.3f}", "graph vs independent ML", f"PR-AUC, 95% CI {pair['C_minus_B']['lo']:+.3f} to {pair['C_minus_B']['hi']:+.3f}: not significant", AMBER),
             (f"{F['genai']['failure_recall'].loc['E_rules_plus_llm']:.2f}", "failures surfaced, rules + LLM", f"vs {F['genai']['failure_recall'].loc['A_rules']:.2f} rules only; {int(F['genai']['failures_total'].iloc[0])} failures, {int(F['genai']['episodes'].iloc[0])} episodes", LBLUE)]
    for i, (n, l1, l2, col) in enumerate(cards):
        x = 40 + i * 223
        D.rect(s, x, 406, 211, 66, fill=PANEL, line=col, lw=0.9, r=8, line_alpha=0.8)
        D.text(s, x + 10, 411, 90, 30, n, size=21, bold=True, color=col, lh=1.05)
        D.text(s, x + 100, 414, 108, 28, l1, size=9.8, bold=True, lh=1.15)
        D.text(s, x + 10, 443, 196, 28, l2, size=8.8, color=MUTED, lh=1.15)
    D.text(s, 40, 476, 880, 28, f"Fast-mode run on the VM: {F['n_test_patients']} test patients, {F['n_test_obl']} test obligations, simulated outcomes, patient-grouped bootstrap (100 resamples), thresholds chosen on validation. "
                                 "Close to the deadline a deadline-only rule is hard to beat (max-score PR-AUC 0.981 vs 0.969).", size=9.4, color=DIM, lh=1.22)
    s.notes = ("Only measured results. The strongest finding is behavioural: the evidence-aware engine does not call undocumented work failed, and the graph model warns earlier with fewer false alerts at 24 hours of lead. "
               "We also show where we are not better: graph versus independent ML is within noise here, and a deadline-only rule is strong near the deadline. Fast-mode sample is small; full-mode confidence intervals will be tighter.")


# ------------------------------------------------------------------------------------------------ slide 8: demo
def slide_demo():
    s = D.new(); header(s, 8, "LIVE PROTOTYPE", "A delayed acknowledgement, handled correctly")
    panel(s, 40, 112, 300, 150, accent=LBLUE)
    D.text(s, 54, 124, 270, 14, "1  INPUT NOTE", size=9.6, bold=True, color=LBLUE)
    D.text(s, 54, 142, 274, 116, "Patient discharged after pneumonia admission. Repeat the CBC within 48 hours. The treating physician must review the result. Contact the patient if the result is abnormal. Arrange a follow-up appointment in 2 weeks.",
           size=10.4, color=WHITE, lh=1.3, family="mono")
    panel(s, 40, 270, 300, 200, accent=CYAN)
    D.text(s, 54, 282, 270, 14, "2  EXTRACTED OBLIGATIONS (RULES + GRANITE 350M)", size=9.2, bold=True, color=CYAN)
    ob = [("ob0", "laboratory_test", "due PT48H"), ("ob1", "followup_review", "role: ordering_clinician; after ob0"), ("ob2", "patient_communication", "if the result is abnormal; after ob1"), ("ob3", "appointment_referral", "due P2W; after ob1")]
    for i, (a, b, c) in enumerate(ob):
        y = 304 + i * 40
        D.text(s, 54, y, 30, 14, a, size=10, bold=True, color=CYAN, family="mono")
        D.text(s, 88, y, 244, 14, b, size=10.4, bold=True, family="mono")
        D.text(s, 88, y + 15, 244, 22, c, size=9.4, color=MUTED, lh=1.15)
    # timeline
    panel(s, 354, 112, 566, 166, accent=GREEN)
    D.text(s, 368, 124, 540, 14, "3  AUDIT TRAIL OF THE RESULT-REVIEW TASK (VERBATIM FROM THE EXECUTED NOTEBOOK)", size=9.2, bold=True, color=GREEN)
    ev = [("Mar 04  09:00", "pending", "R4  task created", BLUE), ("Mar 09  09:00", "censored_not_yet_assessable", "R6  deadline passed, awaiting late evidence", AMBER),
          ("Mar 09  10:00", "overdue_without_verified_completion", "R5  tracker confirms task still open", RED), ("Mar 10  15:00", "completed_with_evidence", "R1  delayed acknowledgement arrives", GREEN)]
    D.line(s, 392, 160, 392, 254, color=PANEL_LN, lw=2, alpha=0.9)
    for i, (t, st, why, col) in enumerate(ev):
        y = 146 + i * 31
        D.ellipse(s, 384, y + 6, 16, 16, fill=col, alpha=1.0)
        D.text(s, 412, y + 2, 98, 14, t, size=10, color=MUTED, family="mono")
        D.text(s, 514, y, 400, 15, st, size=11.2, bold=True, color=col, family="mono")
        D.text(s, 514, y + 14, 400, 13, why, size=9.4, color=MUTED)
    panel(s, 354, 286, 566, 184)
    D.text(s, 368, 296, 540, 14, "4  RISK OF EACH TASK AS EVENTS ARRIVE (GRAPH ML, REAL OUTPUT)", size=9.2, bold=True, color=CYAN)
    D.image(s, str(HERE / "evidence" / "figures" / "demo_risk_monitor.png"), 366, 314, 542, 152)
    D.text(s, 40, 476, 880, 28, "Interim “overdue” was a correct state at that time and was reconciled when the late evidence arrived (verified in the notebook: overdue_without_verified_completion to completed_with_evidence). "
                                 "Output of notebook section 17, executed on the LinuxONE VM; the scenario events are scripted.", size=9.6, color=DIM, lh=1.22)
    s.notes = ("Walk through the story: a discharge note produces four obligations. The review task passes its deadline, so it first becomes 'censored', then 'overdue' once the tracker confirms it is still open. "
               "When the delayed acknowledgement is documented, the engine reconciles the state to completed, with the evidence id in the audit trail. Real output of the executed notebook; the events are scripted.")


# ------------------------------------------------------------------------------------------------ slide 9: innovation / impact / scale
def slide_impact():
    s = D.new(); header(s, 9, "INNOVATION, IMPACT & SCALABILITY", "What is proven, and what comes next")
    e4 = F["e4"]
    panel(s, 40, 112, 430, 300, accent=GREEN)
    D.text(s, 56, 124, 400, 16, "MEASURED (FAST-MODE RUN ON LINUXONE)", size=10.5, bold=True, color=GREEN)
    meas = [(f"0 of {F['missing_n']}", "undocumented-but-done tasks confirmed overdue (naive rule: all)"),
            (f"{F['dup_ignored']}", f"duplicate deliveries ignored; final states identical to a clean run ({F['dup_identical']:.0%})"),
            ("1.000", "stored states restored identically when rebuilt from the SQLite event log"),
            (f"{ew_far('C_graph_ml'):.1f}% vs {ew_far('A_deadline_only'):.1f}%", "false-alert rate, graph model vs deadline-only rule, alerts >= 24 h ahead"),
            ("56 / 56", "notebook code cells executed without error in a clean kernel")]
    for i, (n, t) in enumerate(meas):
        y = 148 + i * 50
        D.text(s, 56, y, 120, 30, n, size=17, bold=True, color=CYAN, lh=1.05)
        D.text(s, 178, y + 1, 276, 44, t, size=10.8, color=MUTED, lh=1.22)
    panel(s, 490, 112, 430, 300, accent=AMBER)
    D.text(s, 506, 124, 400, 16, "ANTICIPATED (NOT YET MEASURED)", size=10.5, bold=True, color=AMBER)
    ant = [("Deployment", "discharge follow-up monitoring fed by EHR event streams, with a human review queue"),
           ("Scale-out", "larger LinuxONE partitions for more patients; NNPA for neural components (current models are tree and linear)"),
           ("Validation", "de-identified real records and clinician-reviewed labels (there are 0 gold labels today)"),
           ("Privacy", "on-premises inference keeps notes on the server; a formal security evaluation is future work")]
    for i, (k, t) in enumerate(ant):
        y = 148 + i * 62
        D.text(s, 506, y, 90, 16, k, size=11.5, bold=True, color=WHITE)
        D.text(s, 600, y, 308, 58, t, size=10.8, color=MUTED, lh=1.22)
    cmp = [("vs reminder systems", "evidence-aware: absence is unverified, not failed"), ("vs LLM-only pipelines", "validated, grounded fields; deterministic states"), ("vs risk scores alone", "dependency and time aware, with an audit trail")]
    for i, (t, b) in enumerate(cmp):
        x = 40 + i * 300
        D.rect(s, x, 424, 280, 66, fill=BLUE, alpha=0.15, line=LBLUE, lw=0.8, r=8, line_alpha=0.7)
        D.text(s, x + 12, 432, 258, 18, t, size=12.5, bold=True)
        D.text(s, x + 12, 454, 258, 34, b, size=10.6, color=MUTED, lh=1.2)
    s.notes = ("Left: what we measured. Right: what we only anticipate, clearly separated. The social value is fewer missed follow-ups with fewer false alarms; the honest next step is validation on de-identified real records with clinician-reviewed labels.")


def ew_far(k):
    return float(F["ew"].loc[k, "false_alert_rate"]) * 100


# ------------------------------------------------------------------------------------------------ slide 10: conclusion
def slide_conclusion():
    s = D.new(); header(s, 10, "CONCLUSION", "No follow-up left behind", title_size=34)
    ach = [(f"{F['missing_engine_confirmed_overdue'] * F['missing_n']:.0f} / {F['missing_n']}", "false failures", "Evidence-aware states never call undocumented work “failed”; the naive rule did, every time."),
           (f"{ew_far('C_graph_ml'):.1f}%", "false-alert rate", f"Graph forecasting warns {F['ew'].loc['C_graph_ml', 'median_lead_h']:.0f} h ahead with fewer false alerts than a deadline-only rule ({ew_far('A_deadline_only'):.1f}%)."),
           ("s390x", "end to end", "Local Granite LLM, streaming engine and ML, plus a verified NNPA device test, all on one LinuxONE VM.")]
    for i, (n, u, t) in enumerate(ach):
        x = 40 + i * 300
        panel(s, x, 108, 280, 142, accent=CYAN)
        D.text(s, x + 16, 120, 250, 44, n, size=32, bold=True, color=CYAN, lh=1.05)
        D.text(s, x + 16, 160, 250, 16, u, size=11.5, bold=True)
        D.text(s, x + 16, 180, 252, 68, t, size=10.8, color=MUTED, lh=1.24)
    panel(s, 40, 262, 430, 140)
    D.text(s, 56, 272, 400, 14, "WHY LINUXONE MATTERS HERE", size=10, bold=True, color=CYAN)
    D.text(s, 56, 292, 404, 108, ["Sensitive notes are processed by a local model on the server that holds the data.", "The same VM runs extraction, graph, ML and the stream engine, with recoverable state.",
                                  "NNPA execution is verified, a path for future neural models."], size=11, color=MUTED, lh=1.24, space=4)
    panel(s, 490, 262, 430, 140)
    D.text(s, 506, 272, 400, 14, "NEXT STEPS", size=10, bold=True, color=AMBER)
    D.text(s, 506, 292, 404, 108, ["Clinician-reviewed labels to replace weak and simulated ones.", "Validate on de-identified real records under governance.",
                                   "Run the full-size experiments and neural components on the NNPA."], size=11, color=MUTED, lh=1.24, space=4)
    D.text(s, 40, 414, 880, 40, [[("Detect. Decide. Respond.", {"color": CYAN}), "  And never mistake silence for failure."]], size=22, bold=True, lh=1.2)
    D.text(s, 40, 453, 560, 16, f"Code and notebook: {REPO}", size=11.5, color=CYAN, bold=True)
    tm = ("Team " + TEAM["name"] if TEAM.get("name") else "") + (": " if TEAM.get("name") and TEAM.get("members") else "") + ", ".join(TEAM.get("members", []))
    D.text(s, 40, 471, 880, 16, tm if tm else "Prototype on synthetic data. Alerts are decision support for qualified human review.", size=11, color=WHITE if tm else DIM, bold=bool(tm))
    if tm:
        D.text(s, 40, 488, 880, 14, "Prototype on synthetic data. Alerts are decision support for qualified human review.", size=9.5, color=DIM)
    s.notes = ("Three achievements: no false failures from missing documentation, earlier warnings with fewer false alerts, and an end-to-end system on LinuxONE. "
               "Close on the limits: synthetic data, weak labels, small fast-mode sample. The repository has the notebook, tests and instructions to reproduce everything.")


def main():
    for fn in (slide_cover, slide_problem, slide_solution, slide_arch, slide_data_ml, slide_linuxone, slide_results, slide_demo, slide_impact, slide_conclusion):
        fn()
    pdf = OUTDIR / "IBM_Z_Datathon_2026_Pitch_Deck.pdf"; pptx = OUTDIR / "IBM_Z_Datathon_2026_Pitch_Deck.pptx"
    D.save_pdf(str(pdf)); D.save_pptx(str(pptx))
    print("wrote", pdf, pptx, "| slides:", len(D.slides))
    if D.warnings:
        print("TEXT FIT WARNINGS:"); [print("  ", w) for w in D.warnings]
    issues = D.check_pptx(str(pptx))
    print("PPTX geometry issues:", issues or "none")


if __name__ == "__main__":
    main()
