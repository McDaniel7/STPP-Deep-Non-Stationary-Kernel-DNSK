"""Shared matplotlib styling for the example figures."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
AXIS = "#c9c8c3"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (all-pairs safe)
EVENT = "#eb6834"

# Single-hue sequential ramp (near zero recedes to the surface) and an ordinal sub-ramp.
_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("dnsk_blue", [SURFACE] + _BLUE)
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
# Warm single-hue ramp for influence kernels (white -> deep orange-red).
WARM = LinearSegmentedColormap.from_list(
    "dnsk_warm", [SURFACE, "#fde5d6", "#fbc4a5", "#f7a072", "#f07b45", "#e05a2b", "#bf3f1c", "#8f2a12", "#5e1a0b"])
# Diverging blue <-> red with a neutral gray midpoint (for kernels that take negative values).
DIVERGING = LinearSegmentedColormap.from_list(
    "dnsk_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a7a6", "#e34948", "#9c2423"])


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "text.color": TEXT, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
        "axes.edgecolor": AXIS, "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "image.interpolation": "bilinear",
    })


def colormap_for(values, kind="intensity", style="nature"):
    """Colour map and normalisation for a group of heatmaps sharing one scale.

    ``kind`` is ``"kernel"`` or ``"intensity"``. ``style="nature"`` uses a warm ramp for kernels and a
    blue ramp for intensities (both from 0), switching kernels to a diverging map if they take clearly
    negative values. ``style="iclr"`` mimics the paper: viridis for kernels, ``bwr`` for intensities,
    each spanning the data range.
    """
    lo, hi = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
    if style == "iclr":
        return ("viridis" if kind == "kernel" else "bwr"), mpl.colors.Normalize(vmin=lo, vmax=hi)
    if kind == "kernel" and lo < -0.02 * hi:
        bound = max(hi, -lo)
        return DIVERGING, TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    return (WARM if kind == "kernel" else SEQUENTIAL), mpl.colors.Normalize(vmin=0.0, vmax=hi)


def event_color(style="nature"):
    """Colour of event markers drawn on top of intensity maps."""
    return "#0b0b0b" if style == "iclr" else EVENT


__all__ = ["apply", "colormap_for", "event_color", "TwoSlopeNorm", "SEQUENTIAL", "WARM", "DIVERGING",
           "ORDINAL_BLUE", "CATEGORICAL", "SURFACE", "TEXT", "TEXT_SECONDARY", "EVENT"]
