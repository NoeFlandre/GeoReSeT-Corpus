"""Matplotlib rendering of the card figures. Styling only: logic lives in ``geo``.

Excluded from mutation testing; the figures are checked by looking at them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import shapely
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

from georeset.normalize.languages import display_name
from georeset.publish.geo import binned_rings, land_rings

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
LAND = "#e9e8e4"
BAR = "#2a78d6"
# One-hue sequential ramp (light -> dark), one step per decade of sentences.
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
EDGES = [10**k for k in range(len(RAMP) + 1)]


def _figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def world_map(counts: pl.DataFrame, land: list[shapely.Geometry], path: Path) -> Path:
    """Equal Earth map of sentences per H3 cell, log-binned on one blue ramp."""
    fig, ax = _figure(12, 6)
    ax.add_collection(PolyCollection(land_rings(land), facecolors=LAND, edgecolors="none"))
    drawn = binned_rings(counts, len(RAMP))
    rings = [ring for ring, _ in drawn]
    ax.add_collection(PolyCollection(rings, facecolors=[RAMP[b] for _, b in drawn], edgecolors="none"))
    ax.set_xlim(-17_000_000, 17_000_000)
    ax.set_ylim(-7_500_000, 8_700_000)
    ax.set_aspect("equal")
    ax.axis("off")
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=BoundaryNorm(EDGES, len(RAMP)), cmap=ListedColormap(RAMP)),
        ax=ax,
        orientation="horizontal",
        fraction=0.04,
        pad=0.02,
        aspect=40,
    )
    bar.set_ticks(EDGES, labels=[f"{e:,}" for e in EDGES])
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)
    bar.set_label("Sentences per H3 cell (resolution 3)", color=TEXT_SECONDARY, fontsize=9)
    return _save(fig, path)


def language_bars(top: pl.DataFrame, path: Path) -> Path:
    """Horizontal bars of sentences per language, largest on top."""
    fig, ax = _figure(8, 0.32 * top.height + 0.6)
    labels = [display_name(c) for c in top["language"]][::-1]
    values = top["sentences"].to_list()[::-1]
    ax.barh(labels, values, color=BAR, height=0.7)
    for y, value in enumerate(values):
        ax.text(value, y, f" {value:,}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.tick_params(axis="y", colors=TEXT, labelsize=9, length=0)
    ax.xaxis.set_visible(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, max(values) * 1.18)
    return _save(fig, path)
