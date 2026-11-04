"""Matplotlib rendering of the card figures. Styling only: logic lives in ``geo``.

Excluded from mutation testing; the figures are checked by looking at them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

from georeset.normalize.languages import display_name
from georeset.publish.geo import cell_centres, edge_labels, log_edges

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
EMPTY = "#e3e2de"  # hexagons with data in the full dataset only
BAR = "#2a78d6"
# One-hue sequential ramp (light -> dark), resampled to the number of bins.
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d2b5e"]
LEVELS = 11
HEX_DEGREES = 2
# Rows of the hexagon grid that make regular hexagons over 180° of latitude.
_GRID = (360 // HEX_DEGREES, round(180 / (HEX_DEGREES * np.sqrt(3))))
_EXTENT = (-180, 180, -90, 90)


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


def _hexbin(ax, centres: pl.DataFrame, **kwargs):
    return ax.hexbin(
        centres["lon"], centres["lat"], gridsize=_GRID, extent=_EXTENT, linewidths=0.15, **kwargs
    )


def _colour(hexes) -> list[int]:
    """Bin the hexagons' means on log-spaced edges; return the edges."""
    values = hexes.get_array()
    edges = log_edges(float(values.max()) if values.size else 1, LEVELS)
    bins = len(edges) - 1
    hexes.set_cmap(ListedColormap(LinearSegmentedColormap.from_list("ramp", RAMP)(np.linspace(0, 1, bins))))
    hexes.set_norm(BoundaryNorm(edges, bins))
    return edges


def _colorbar(fig, ax, hexes, edges: list[int], label: str) -> None:
    bar = fig.colorbar(hexes, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40)
    bar.set_ticks([(lo + hi) / 2 for lo, hi in itertools.pairwise(edges)], labels=edge_labels(edges))
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)
    bar.set_label(label, color=TEXT_SECONDARY, fontsize=9)


def world_map(
    counts: pl.DataFrame, path: Path, unit: str, resolution: int, full: pl.DataFrame | None = None
) -> Path:
    """Plate carrée map of the mean ``unit`` per H3 cell in each 2° hexagon, log-binned on one blue ramp.

    ``full``: H3 counts of the full dataset, drawn grey where ``counts`` has nothing.
    """
    fig, ax = _figure(12, 6.4)
    label = f"{unit} per H3 cell (resolution {resolution}), mean per {HEX_DEGREES}° hexagon"
    if full is not None:
        _hexbin(ax, cell_centres(full), mincnt=1, cmap=ListedColormap([EMPTY]), edgecolors=EMPTY)
        label += " (grey: the full dataset)"
    centres = cell_centres(counts)
    hexes = _hexbin(ax, centres, C=centres["n"], reduce_C_function=np.mean, edgecolors="face")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal")
    ax.axis("off")
    _colorbar(fig, ax, hexes, _colour(hexes), label)
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
