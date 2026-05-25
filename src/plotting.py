"""Plotting helpers for the agent-memory paper. CPU only.

Centralized so figures used in to_human/ progress reports and paper/ stay
visually consistent. matplotlib only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

LABELS = ["single-hop", "temporal", "multi-hop", "open-domain"]
LABEL_COLORS = {
    "single-hop":  "#2566c2",
    "temporal":    "#d4a017",
    "multi-hop":   "#1a6f3a",
    "open-domain": "#c2255c",
}


def style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    })


def save_fig(fig, out_path: Path | str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out_path} (+ .pdf)")
