"""Plot helpers (matplotlib, headless-safe) and notebook display helpers. Every plot is saved to figures/ and returned."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg") if "ipykernel" not in __import__("sys").modules and "inline" not in os.environ.get("MPLBACKEND", "") else (matplotlib.get_backend() if "ipykernel" in __import__("sys").modules else "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATE_COLORS = {"pending": "#5b8def", "completed_with_evidence": "#2e9e5b", "overdue_without_verified_completion": "#d64545", "unverified_insufficient_evidence": "#e0a800",
                "contradictory_evidence": "#9b59b6", "censored_not_yet_assessable": "#8c8c8c"}
STATE_SHORT = {"pending": "pending", "completed_with_evidence": "completed (evidence)", "overdue_without_verified_completion": "overdue (confirmed)",
               "unverified_insufficient_evidence": "unverified", "contradictory_evidence": "contradictory", "censored_not_yet_assessable": "not yet assessable"}
NODE_COLORS = {"Patient": "#444444", "Encounter": "#7f8c8d", "ClinicalNote": "#95a5a6", "Obligation": "#5b8def", "Task": "#2e9e5b", "Result": "#16a085", "Review": "#f39c12",
               "Acknowledgment": "#27ae60", "Notification": "#8e44ad", "EvidenceEvent": "#bdc3c7", "ResponsibleRole": "#34495e"}


def save(fig, cfg_or_dir, name: str):
    d = Path(cfg_or_dir) if not isinstance(cfg_or_dir, dict) else Path(cfg_or_dir["_root"]) / cfg_or_dir["paths"]["figures"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    return p


def plot_calibration(curves: dict[str, pd.DataFrame], title="Reliability (calibration) curve", cfg=None, name="calibration"):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for k, c in curves.items():
        ax.plot(c.mean_pred, c.frac_pos, "o-", ms=4, label=k)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction positive"); ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=.3)
    if cfg:
        save(fig, cfg, name)
    return fig


def plot_confusion(df: pd.DataFrame, title: str, cfg=None, name="confusion", normalize=True):
    m = df.values.astype(float)
    if normalize:
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.62 * len(df.columns) + 2.2, 0.55 * len(df.index) + 1.6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels([c.replace("pred:", "") for c in df.columns], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels([c.replace("true:", "") for c in df.index], fontsize=7)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if df.values[i, j]:
                ax.text(j, i, int(df.values[i, j]), ha="center", va="center", fontsize=6, color="white" if m[i, j] > .5 else "black")
    ax.set_title(title, fontsize=9)
    if cfg:
        save(fig, cfg, name)
    return fig


def plot_bars(df: pd.DataFrame, x: str, y: str, hue: str | None = None, title="", cfg=None, name="bars", err: tuple[str, str] | None = None, ylabel=None, figsize=(7, 3.6)):
    fig, ax = plt.subplots(figsize=figsize)
    if hue:
        cats, hues = list(dict.fromkeys(df[x])), list(dict.fromkeys(df[hue]))
        w = 0.8 / len(hues)
        for i, h in enumerate(hues):
            d = df[df[hue] == h].set_index(x).reindex(cats)
            xs = np.arange(len(cats)) + i * w
            e = None if err is None else [d[y] - d[err[0]], d[err[1]] - d[y]]
            ax.bar(xs, d[y], w, label=str(h), yerr=e, capsize=2)
        ax.set_xticks(np.arange(len(cats)) + 0.4 - w / 2); ax.set_xticklabels(cats, rotation=25, ha="right", fontsize=8); ax.legend(fontsize=7)
    else:
        e = None if err is None else [df[y] - df[err[0]], df[err[1]] - df[y]]
        ax.bar(df[x], df[y], yerr=e, capsize=2)
        ax.set_xticklabels(df[x], rotation=25, ha="right", fontsize=8)
    ax.set_title(title); ax.set_ylabel(ylabel or y); ax.grid(axis="y", alpha=.3)
    if cfg:
        save(fig, cfg, name)
    return fig


def plot_lines(series: dict[str, tuple], title="", xlabel="", ylabel="", cfg=None, name="lines", figsize=(5.4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for k, (x, y) in series.items():
        ax.plot(x, y, marker="o", ms=3, label=k)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.grid(alpha=.3)
    if cfg:
        save(fig, cfg, name)
    return fig
