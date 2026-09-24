"""Shared matplotlib style: validated categorical palette, recessive chrome.

The palette is a colour-blind-validated categorical order (blue, orange, aqua...)
and a reserved status palette. Charts use one y-axis only (no dual axes), thin
marks and hairline grids.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SYNTHETIC_NOTE = "Synthetic data (prototype) - not SOMELEC data"


def apply_style() -> None:
    """Apply the project style to matplotlib globally."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "figure.figsize": (11, 4),
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def add_source_note(fig: plt.Figure, note: str = SYNTHETIC_NOTE) -> None:
    """Stamp every figure with its data provenance."""
    fig.text(0.99, 0.005, note, ha="right", va="bottom", fontsize=8, color=INK_MUTED)
