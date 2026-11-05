"""Matplotlib rendering of the card figures, on the Equal Earth projection. Styling only: logic in ``geo``.

Excluded from mutation testing; the figures are checked by looking at them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, to_rgba
from matplotlib.patches import Polygon

from georeset.normalize.languages import display_name
from georeset.publish.geo import (
    CELL_RESOLUTION,
    DISPLAY_RESOLUTION,
    LATITUDES,
    cell_rings,
    edge_labels,
    equal_earth,
    log_edges,
    outline,
)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
EMPTY = "#e3e2de"  # cells with data in the full dataset only, or land without enough
BAR = "#2a78d6"
# One-hue sequential ramp (light -> dark), resampled to the number of bins.
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d2b5e"]
# Two-sided ramp for principal component scores: blue below zero, orange above.
DIVERGING = ["#0d2b5e", "#2a78d6", "#b7d3f6", "#f2f1ed", "#f6c9a8", "#d9702a", "#7a3410"]
LEVELS = 11
CAPTION_WIDTH = 100  # characters per line of a caption or bar label: the text stays within the map's width


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


def _cells(ax, cells: np.ndarray, values=None, colours=None, **kwargs) -> PolyCollection:
    """``cells`` as their projected outlines, coloured by ``values`` (``cmap``, ``norm``) or ``colours``."""
    rings, owners = cell_rings(cells)
    drawn = PolyCollection(rings, linewidths=0.1, **kwargs)
    if colours is None:
        drawn.set_array(np.asarray(values)[owners])
        drawn.set_edgecolor("face")
    else:
        drawn.set_facecolor(np.asarray(colours)[owners])
        drawn.set_edgecolor(np.asarray(colours)[owners])
    drawn.set_clip_path(Polygon(outline(), transform=ax.transData))
    ax.add_collection(drawn)
    return drawn


def _grey(ax, cells: np.ndarray) -> None:
    """``cells`` in light grey, under the data."""
    _cells(ax, cells, colours=np.tile(to_rgba(EMPTY), (len(cells), 1)))


def _colour(hexes) -> list[int]:
    """Bin the cells' means on log-spaced edges; return the edges."""
    values = hexes.get_array()
    edges = log_edges(float(values.max()) if values.size else 1, LEVELS)
    bins = len(edges) - 1
    hexes.set_cmap(ListedColormap(LinearSegmentedColormap.from_list("ramp", RAMP)(np.linspace(0, 1, bins))))
    hexes.set_norm(BoundaryNorm(edges, bins))
    return edges


def _colorbar(fig, ax, hexes, edges: list[int], label: str) -> None:
    bar = _bar(fig, ax, hexes, label)
    bar.set_ticks([(lo + hi) / 2 for lo, hi in itertools.pairwise(edges)], labels=edge_labels(edges))


def _wrapped(text: str) -> str:
    """``text`` with each of its lines wrapped to ``CAPTION_WIDTH``."""
    return "\n".join(textwrap.fill(line, CAPTION_WIDTH) for line in text.split("\n"))


def _bar(fig, ax, hexes, label: str, **kwargs):
    """A horizontal colour bar under the map, in the card's quiet style."""
    bar = fig.colorbar(hexes, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40, **kwargs)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)
    bar.set_label(_wrapped(label), color=TEXT_SECONDARY, fontsize=9)
    return bar


def world_map(
    cells: np.ndarray, values: np.ndarray, path: Path, unit: str, grey: np.ndarray | None = None
) -> Path:
    """``values`` per display cell (the mean ``unit`` of its child cells), log-binned on one blue ramp.

    ``grey``: cells drawn grey under them (the full dataset).
    """
    fig, ax = _figure(12, 6.4)
    label = (
        f"{unit} per H3 cell (resolution {CELL_RESOLUTION}), mean per resolution-{DISPLAY_RESOLUTION} cell"
    )
    if grey is not None:
        _grey(ax, grey)
        label += " (grey: the full dataset)"
    drawn = _cells(ax, cells, values)
    _world(ax)
    _colorbar(fig, ax, drawn, _colour(drawn), label)
    return _save(fig, path)


def _world(ax) -> None:
    """Equal Earth extent over the maps' latitudes, no axes."""
    x = outline()[:, 0]
    _, y = equal_earth(np.zeros(2), np.array(LATITUDES))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(*y)
    ax.set_aspect("equal")
    ax.axis("off")


def rgb_map(cells: np.ndarray, rgb: np.ndarray, path: Path, land: np.ndarray, caption: str) -> Path:
    """``cells`` coloured by ``rgb`` (three values in [0, 1] each), over grey ``land``."""
    fig, ax = _figure(12, 6.4)
    _grey(ax, land)
    _cells(ax, cells, colours=np.asarray(rgb))
    _world(ax)
    fig.text(0.5, 0.06, _wrapped(caption), ha="center", va="top", color=TEXT_SECONDARY, fontsize=9)
    return _save(fig, path)


def diverging_map(
    cells: np.ndarray, values: np.ndarray, limit: float, label: str, path: Path, land: np.ndarray
) -> Path:
    """``cells`` coloured by ``values`` on a two-sided ramp from ``-limit`` to ``limit``, clipped beyond."""
    fig, ax = _figure(12, 6.4)
    _grey(ax, land)
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    drawn = _cells(ax, cells, values, cmap=cmap, norm=Normalize(-limit, limit))
    _world(ax)
    _bar(fig, ax, drawn, label, extend="both")
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
