"""Charts for the deck, drawn from the archived result files only (see facts.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import facts

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)
TXT, MUT, GRID = "#E8EEF9", "#A8B8D8", "#2B3F66"
CYAN, BLUE, GRAY, RED, GREEN, AMBER = "#33B1FF", "#0F62FE", "#6F83A8", "#FA4D56", "#42BE65", "#F1C21B"
plt.rcParams.update({"font.family": "Liberation Sans", "text.color": TXT, "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": TXT,
                     "axes.edgecolor": GRID, "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none", "font.size": 12})


def _style(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7, alpha=0.8); ax.set_axisbelow(True)


def chart_early_warning(F, size=(4.15, 2.9)):
    ew = F["ew"]; sy = [("A_deadline_only", "A  deadline-only", GRAY), ("A_reminder_24h", "A  reminder at 24 h", "#8FA3C8"), ("B_independent_ml", "B  independent ML", BLUE), ("C_graph_ml", "C  graph ML (ours)", CYAN)]
    fig, ax = plt.subplots(figsize=size)
    ys = np.arange(len(sy))[::-1]
    for y, (k, lab, col) in zip(ys, sy):
        far = ew.loc[k, "false_alert_rate"] * 100
        ax.barh(y, far, color=col, height=0.55)
        ax.text(far + 0.8, y + 0.13, f"{far:.1f}%", va="center", fontsize=15, fontweight="bold", color=TXT)
        ax.text(far + 0.8, y - 0.24, f"recall {ew.loc[k, 'recall'] * 100:.0f}%, lead {ew.loc[k, 'median_lead_h']:.0f} h", va="center", fontsize=10.2, color=MUT)
    ax.set_yticks(ys); ax.set_yticklabels([s[1] for s in sy], fontsize=11.5)
    ax.set_xlim(0, 44); ax.set_xlabel("false-alert rate % (lower is better)", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=10.5); _style(ax, "x")
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "chart_early_warning.png", dpi=240); plt.close(fig)


def chart_engine_vs_naive(F, size=(4.15, 2.75)):
    metrics = [("Done but undocumented\n(n=%d): called overdue" % F["missing_n"], F["missing_naive_overdue"] * 100, F["missing_engine_confirmed_overdue"] * 100),
               ("Truly unverified tasks:\nrecognised as not confirmed", F["nc_recall_naive"] * 100, F["nc_recall_engine"] * 100)]
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(2); w = 0.34
    for i, (lab, nv, en) in enumerate(metrics):
        ax.bar(i - w / 2 - 0.01, nv, w, color=GRAY); ax.bar(i + w / 2 + 0.01, en, w, color=CYAN)
        for xx, v in ((i - w / 2 - 0.01, nv), (i + w / 2 + 0.01, en)):
            ax.text(xx, v + 3, f"{v:.0f}%", ha="center", fontsize=15, fontweight="bold", color=TXT)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], fontsize=10.2)
    ax.set_ylim(0, 160); ax.set_yticks([0, 50, 100]); ax.set_yticklabels(["0", "50", "100 %"], fontsize=10.5); _style(ax, "y")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRAY, label="naive: no record = failed"), Patch(color=CYAN, label="MedRelay evidence-aware engine")], loc="upper center", fontsize=10, frameon=False,
              ncol=1, bbox_to_anchor=(0.5, 1.03), labelcolor=TXT)
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "chart_engine_vs_naive.png", dpi=240); plt.close(fig)


def chart_ablation(F, size=(4.15, 2.75)):
    ab = F["ab"]
    rows = [("remove temporal features", "C_no_temporal"), ("remove graph-dependency features", "C_no_graph_dependency"), ("independent tasks (no graph)", "B_independent_ml"), ("remove evidence features", "C_no_evidence")]
    fig, ax = plt.subplots(figsize=size)
    ys = np.arange(len(rows))[::-1]
    for y, (lab, k) in zip(ys, rows):
        d, lo, hi = ab.loc[k, "pr_auc_diff_vs_C"], ab.loc[k, "diff_ci_lo"], ab.loc[k, "diff_ci_hi"]
        sig = (hi < 0) or (lo > 0)
        col = CYAN if sig else GRAY
        ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round"); ax.plot([d], [y], "o", color=col, ms=9)
        ax.text(0.05, y, f"{d:+.3f}", va="center", ha="left", fontsize=13, fontweight="bold", color=TXT)
    ax.axvline(0, color=MUT, lw=1, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels(["remove temporal\nfeatures", "remove graph-\ndependency features", "independent tasks\n(no graph)", "remove evidence\nfeatures"], fontsize=10.2)
    ax.set_xlim(-0.24, 0.16); ax.set_xticks([-0.2, -0.1, 0.0]); ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlabel("change in PR-AUC vs full model", fontsize=10.5); _style(ax, "x")
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "chart_ablation.png", dpi=240); plt.close(fig)


def chart_nnpa(F, size=(4.3, 2.35)):
    r = F["mm"][F["mm"].M == 1024].iloc[0]
    bars = [("C loop (CPU)", r["cpu_naive_c_ms"], GRAY), ("NumPy BLAS (CPU)", r["numpy_blas_ms"], GRAY), ("NNPA, incl. conversion", r["nnpa_total_ms"], CYAN), ("NNPA, operation only", r["nnpa_op_ms"], BLUE)]
    fig, ax = plt.subplots(figsize=size)
    ys = np.arange(len(bars))[::-1]
    for y, (lab, v, col) in zip(ys, bars):
        ax.barh(y, v, color=col, height=0.56)
        ax.text(v * 1.25, y, f"{v:,.2f} ms" if v < 10 else f"{v:,.0f} ms", va="center", fontsize=13, fontweight="bold", color=TXT)
    ax.set_xscale("log"); ax.set_xlim(0.25, 40000)
    ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=11)
    ax.set_xlabel("1024^3 matmul time, ms (log scale)", fontsize=10.5); ax.tick_params(axis="x", labelsize=10); _style(ax, "x")
    fig.tight_layout(pad=0.4); fig.savefig(OUT / "chart_nnpa.png", dpi=240); plt.close(fig)


def build_all():
    F = facts.load()
    chart_early_warning(F); chart_engine_vs_naive(F); chart_ablation(F); chart_nnpa(F)
    return F


if __name__ == "__main__":
    build_all(); print("charts written to", OUT)
