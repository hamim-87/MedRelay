"""Demonstration views: patient journey, temporal evidence graph, risk monitor, alert feed, performance panel, and an interactive replay widget
(with a static storyboard fallback when ipywidgets is not usable). All figures are rendered from the live processor/graph state."""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .alert_engine import STEP_LABEL
from .evidence_graph import iso, to_epoch
from .ontology import COMPLETION_EVENT, State
from .visualization import NODE_COLORS, STATE_COLORS, STATE_SHORT

H = 3600.0
EVENT_MARK = {"lab_collected": ("o", "#2e9e5b"), "result_posted": ("s", "#16a085"), "result_opened": ("^", "#f39c12"), "result_acknowledged": ("D", "#27ae60"),
              "patient_notified": ("P", "#8e44ad"), "appointment_scheduled": ("o", "#2980b9"), "visit_attended": ("*", "#2e9e5b"), "visit_no_show": ("X", "#d64545"),
              "reminder_sent": (".", "#7f8c8d"), "contact_attempt": (".", "#7f8c8d"), "task_status_snapshot": ("v", "#d64545"), "conflicting_record": ("X", "#9b59b6")}


def _hrs(t, t0):
    return (t - t0) / H


def state_spans(proc, oid: str, t_end: float) -> list[tuple[float, float, str, bool]]:
    tr = [r for r in proc.transitions if r["oid"] == oid]
    spans = []
    for i, r in enumerate(tr):
        t0 = to_epoch(r["t_ingest"])
        t1 = to_epoch(tr[i + 1]["t_ingest"]) if i + 1 < len(tr) else t_end
        spans.append((t0, t1, r["new_state"], r["late"]))
    return spans


def plot_patient_journey(proc, patient_id: str, title: str | None = None, t_end: float | None = None, figsize=(11, 4.2)):
    g = proc.graph
    oids = sorted(g.patient_obs[patient_id], key=lambda o: (g.obs[o].chain_position, o))
    t0 = min(g.obs[o].created for o in oids)
    t_end = t_end or max(proc.clock, max(g.obs[o].deadline for o in oids))
    fig, ax = plt.subplots(figsize=figsize)
    for r, oid in enumerate(oids):
        ob = g.obs[oid]
        for a, b, st, late in state_spans(proc, oid, t_end):
            ax.barh(r, _hrs(min(b, t_end), t0) - _hrs(a, t0), left=_hrs(a, t0), height=0.5, color=STATE_COLORS[st], alpha=0.85, edgecolor="none")
        ax.plot([_hrs(ob.deadline, t0)] * 2, [r - 0.38, r + 0.38], color="black", lw=2)
        for e in g.ob_events.get(oid, ()):
            if e.t_event > t_end or e.etype in ("deadline_reached",):
                continue
            m, c = EVENT_MARK.get(e.etype, ("o", "#555"))
            ax.scatter(_hrs(e.t_event, t0), r, marker=m, s=55, color="white", edgecolor=c, linewidth=1.6, zorder=5)
            if e.t_ingest - e.t_event > 6 * H:
                ax.annotate("", xy=(_hrs(e.t_ingest, t0), r + 0.32), xytext=(_hrs(e.t_event, t0), r + 0.32), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    ax.set_yticks(range(len(oids)))
    ax.set_yticklabels([f"{STEP_LABEL.get(g.obs[o].step_type, g.obs[o].step_type)}" for o in oids], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("hours since discharge / obligation creation"); ax.set_title(title or f"Patient journey {patient_id}", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS.values()]
    ax.legend(handles, [STATE_SHORT[k] for k in STATE_COLORS], fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    return fig


def plot_evidence_graph(proc, patient_id: str, now: float | None = None, figsize=(10, 5.2), title: str | None = None):
    """Obligation chain (rounded by state) with its evidence events; delayed evidence (ingested late) is drawn dashed."""
    g = proc.graph
    now = now or proc.clock
    oids = sorted(g.patient_obs[patient_id], key=lambda o: (g.obs[o].chain_position, o))
    D = nx.DiGraph()
    pos, y = {}, 0
    for i, oid in enumerate(oids):
        dec = g.derive_state(oid, now, live=True)
        D.add_node(oid, kind="Obligation", color=STATE_COLORS[dec.state.value], label=f"{STEP_LABEL.get(g.obs[oid].step_type, g.obs[oid].step_type)}\n[{STATE_SHORT[dec.state.value]}{' late' if dec.late else ''}]")
        pos[oid] = (i * 2.6, 0)
        for j, e in enumerate([e for e in g.ob_events.get(oid, ()) if e.etype not in ("deadline_reached", "reminder_sent", "contact_attempt") and e.t_ingest <= now]):
            from .ontology import EVENT_NODE_TYPE
            nt = EVENT_NODE_TYPE.get(e.etype, "EvidenceEvent")
            D.add_node(e.eid, kind=nt, color=NODE_COLORS[nt], label=e.etype.replace("_", "\n"))
            pos[e.eid] = (i * 2.6 + 0.0, -1.5 - 0.9 * j)
            comp = e.etype == COMPLETION_EVENT[g.obs[oid].step_type]
            D.add_edge(oid, e.eid, etype="COMPLETED_BY" if comp else ("CONTRADICTS" if e.etype == "conflicting_record" else "EVIDENCED_BY"), late=e.t_ingest - e.t_event > 6 * H)
        for par in g.obs[oid].depends_on:
            if par in oids:
                D.add_edge(oid, par, etype="DEPENDS_ON", late=False)
    fig, ax = plt.subplots(figsize=figsize)
    style = {"DEPENDS_ON": dict(edge_color="#34495e", width=2.2), "EVIDENCED_BY": dict(edge_color="#95a5a6", width=1.0), "COMPLETED_BY": dict(edge_color="#27ae60", width=2.4),
             "CONTRADICTS": dict(edge_color="#9b59b6", width=2.0)}
    for et, st in style.items():
        es = [(u, v) for u, v, d in D.edges(data=True) if d["etype"] == et]
        if es:
            nx.draw_networkx_edges(D, pos, edgelist=es, ax=ax, arrows=True, arrowsize=12, style="dashed" if et == "CONTRADICTS" else "solid", connectionstyle="arc3,rad=0.12" if et == "DEPENDS_ON" else "arc3,rad=0", **st)
    late_edges = [(u, v) for u, v, d in D.edges(data=True) if d.get("late")]
    if late_edges:
        nx.draw_networkx_edges(D, pos, edgelist=late_edges, ax=ax, edge_color="#c0392b", style="dotted", width=2.2, arrows=True)
    for kind, size, fs in (("Obligation", 3600, 7), (None, 900, 6)):
        nodes = [n for n, d in D.nodes(data=True) if (d["kind"] == "Obligation") == (kind == "Obligation")]
        nx.draw_networkx_nodes(D, pos, nodelist=nodes, node_color=[D.nodes[n]["color"] for n in nodes], node_size=size, ax=ax, edgecolors="white")
        nx.draw_networkx_labels(D, {n: pos[n] for n in nodes}, labels={n: D.nodes[n]["label"] for n in nodes}, font_size=fs, font_color="white" if kind == "Obligation" else "black", ax=ax)
    ax.set_title(title or f"Temporal evidence graph at {iso(now)} (dotted red = evidence documented late)", fontsize=10)
    ax.axis("off")
    ax.legend([plt.Line2D([0], [0], color="#34495e", lw=2), plt.Line2D([0], [0], color="#27ae60", lw=2), plt.Line2D([0], [0], color="#95a5a6", lw=1),
               plt.Line2D([0], [0], color="#c0392b", lw=2, ls=":")], ["DEPENDS_ON", "COMPLETED_BY", "EVIDENCED_BY", "late documentation"], fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def plot_risk_monitor(proc, patient_id: str, scorer: str = "C_graph_ml", figsize=(10, 3.4)):
    rl = proc.risk_log_df() if proc.log_risk else pd.DataFrame()
    g = proc.graph
    oids = sorted(g.patient_obs[patient_id], key=lambda o: (g.obs[o].chain_position, o))
    t0 = min(g.obs[o].created for o in oids)
    fig, ax = plt.subplots(figsize=figsize)
    for oid in oids:
        d = rl[(rl.oid == oid) & (rl.scorer == scorer)].sort_values("t_clock")
        if len(d):
            ax.step([_hrs(t, t0) for t in d.t_clock], d.risk, where="post", marker="o", ms=3, label=STEP_LABEL.get(g.obs[oid].step_type, g.obs[oid].step_type))
        ax.axvline(_hrs(g.obs[oid].deadline, t0), ls=":", lw=0.8, color="grey")
    ax.axhline(proc.alerts.risk_thr, color="#d64545", ls="--", lw=1, label=f"alert threshold {proc.alerts.risk_thr}")
    ax.set_ylim(0, 1.02); ax.set_xlabel("hours since discharge"); ax.set_ylabel("P(unresolved at deadline)"); ax.set_title(f"Real-time risk ({scorer}) as events arrive - patient {patient_id}", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def alert_feed_html(proc, patient_id: str | None = None, last: int = 8) -> str:
    al = [a for a in proc.alert_log if patient_id is None or a.patient_id == patient_id][-last:]
    col = {"high": "#d64545", "medium": "#e0a800", "low": "#5b8def"}
    parts = []
    for a in reversed(al):
        parts.append(f"<div style='border-left:5px solid {col.get(a.severity, '#999')};margin:6px 0;padding:4px 8px;background:#f7f7f7;font-size:12px'>"
                     f"<b>{html.escape(a.rule)}</b> <span style='color:#666'>{html.escape(iso(a.t_ingest))} - {a.status}</span><br>{html.escape(a.explanation)}</div>")
    return "".join(parts) or "<i>no alerts yet</i>"


def audit_table(proc, patient_id: str) -> pd.DataFrame:
    oids = set(proc.graph.patient_obs[patient_id])
    df = pd.DataFrame([r for r in proc.transitions if r["oid"] in oids])
    if df.empty:
        return df
    df["step"] = df.oid.map(lambda o: proc.graph.obs[o].step_type)
    return df[["t_ingest", "step", "old_state", "new_state", "rule", "trigger_event_id", "evidence"]]


class LiveDemo:
    """Step through a scripted episode with the real processor (streaming inference at every event). Works with or without ipywidgets."""

    def __init__(self, cfg: dict, scorers: dict, events: list[dict], patient_id: str, primary: str = "C_graph_ml"):
        self.cfg, self.scorers, self.events, self.patient, self.primary = cfg, scorers, events, patient_id, primary
        self.proc = None

    def run_to(self, k: int):
        from .event_processor import EventProcessor
        p = EventProcessor(self.cfg, ":memory:", scorers=self.scorers, primary=self.primary, rescore_batch=1, log_risk=True, build_nx=True, commit_every=10 ** 9)
        for ev in self.events[:k]:
            p.process(ev)
        p.flush()
        self.proc = p
        return p

    def finish(self):
        self.proc.finalize()
        return self.proc

    def storyboard(self, steps: list[int], with_graph: bool = True):
        figs = []
        for k in steps:
            p = self.run_to(k)
            p.flush()
            t = to_epoch(self.events[min(k, len(self.events)) - 1]["ingest_time_utc"])
            p._advance(t + 1)
            p.flush()
            figs.append((k, p, plot_patient_journey(p, self.patient, f"After {k} deliveries (clock {iso(p.clock)})", t_end=p.clock), plot_evidence_graph(p, self.patient, now=p.clock) if with_graph else None))
        return figs

    def widget(self):
        import ipywidgets as w
        from IPython.display import HTML, clear_output, display
        out = w.Output()
        slider = w.IntSlider(value=len(self.events), min=1, max=len(self.events), step=1, description="events", continuous_update=False, layout=w.Layout(width="520px"))
        play = w.Play(value=1, min=1, max=len(self.events), step=1, interval=700)
        w.jslink((play, "value"), (slider, "value"))

        def draw(_=None):
            with out:
                clear_output(wait=True)
                p = self.run_to(slider.value)
                p._advance(to_epoch(self.events[slider.value - 1]["ingest_time_utc"]) + 1)
                p.flush()
                display(plot_patient_journey(p, self.patient, f"Event {slider.value}/{len(self.events)}: {self.events[slider.value-1]['event_type']} (event {self.events[slider.value-1]['event_time_utc']}, ingested {self.events[slider.value-1]['ingest_time_utc']})", t_end=p.clock))
                display(plot_evidence_graph(p, self.patient, now=p.clock))
                display(HTML(alert_feed_html(p, self.patient)))
                plt.close("all")
        slider.observe(draw, names="value")
        draw()
        display(w.VBox([w.HBox([play, slider]), out]))
