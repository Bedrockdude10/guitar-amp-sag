"""Figure styling for publication plots.

Follows the data-visualisation method used across the project: a **fixed-order,
colourblind-safe categorical palette** (the Okabe-Ito set, the canonical
CVD-safe palette for scientific figures), thin marks, a recessive grid, and no
dual-axis charts (callers use stacked subplots that share the time axis instead
of two y-scales).  Colours are assigned to entities in a fixed order and never
cycled or re-assigned by rank.
"""

from __future__ import annotations

from typing import Dict, List

# Okabe & Ito (2008) colourblind-safe qualitative palette, in fixed order.
OKABE_ITO: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Semantic roles used consistently across figures.  Defined by palette *index*
# so they cannot drift from ``OKABE_ITO`` if the palette is ever edited.
ROLE_COLORS: Dict[str, str] = {
    "target": OKABE_ITO[7],      # black -- ground truth / measured amp
    "prediction": OKABE_ITO[0],  # blue -- our model
    "reference": OKABE_ITO[1],   # vermillion -- a second reference (true V_B+, baseline)
    "fit": OKABE_ITO[2],         # bluish green -- a fitted curve
}

# Maximum points drawn per line: long audio signals are stride-decimated to this
# many samples so figures stay light without changing their shape.
MAX_PLOT_POINTS: int = 20000

# Named figure sizes (inches) shared across the plotting functions.
FIGSIZE_WIDE = (8, 3)          # single wide panel
FIGSIZE_WIDE_STACKED = (8, 4.5)  # two stacked wide panels
FIGSIZE_SMALL = (6, 3.2)       # single compact panel
FIGSIZE_SMALL_STACKED = (6, 4.5)  # two stacked compact panels


def color_for_index(i: int) -> str:
    """Categorical colour for series ``i`` (fixed order, folds into black)."""
    return OKABE_ITO[i % len(OKABE_ITO)]


def apply_style() -> None:
    """Apply the project's recessive matplotlib rcParams (idempotent)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })
