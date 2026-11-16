"""Figures. Matplotlib is imported lazily so the core package works without a display."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .diagnose import isotonic_decreasing
from .report import Analysis

__all__ = ["THEMES", "style_axes", "plain_log_axis", "legend", "label_column", "plot_report"]

# Colour roles. Series colours are assigned in this fixed order and never
# cycled; the first three stay distinguishable under protanopia and
# deuteranopia even when any two marks can sit side by side (scatter plots).
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}


def style_axes(ax, theme: dict, grid_axis: str | None = "y") -> None:
    """Quiet chrome: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(which="both", colors=theme["muted"], labelsize=9, length=3, width=0.8)
    ax.xaxis.label.set_color(theme["ink2"])
    ax.yaxis.label.set_color(theme["ink2"])
    ax.title.set_color(theme["ink"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def plain_log_axis(axis, span: tuple[float, float] | None = None) -> None:
    """Log axis with plain-number tick labels (0.5, 2, 100).

    Ticks sit at 1-2-5 steps, or at finer steps when ``span`` (the axis limits)
    covers less than a decade and 1-2-5 would leave a single label.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    narrow = span is not None and span[1] / span[0] < 10
    subs = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0) if narrow else (1.0, 2.0, 5.0)
    axis.set_major_locator(LogLocator(base=10, subs=subs, numticks=15))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def legend(ax, theme: dict, **kwargs):
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(theme["ink2"])
    return leg


def label_column(ax, xs, ys, names, theme: dict, x_text: float = 1.03, min_gap: float = 0.075) -> None:
    """Label points in a column right of the axes, joined by hairline leaders.

    Labels are spread vertically so that they never overlap, however close the
    points are -- nudging labels next to converging points detaches them from
    their marks, a leader line does not.
    """
    if len(names) == 0:
        return
    y0, y1 = ax.get_ylim()
    frac = (np.asarray(ys, dtype=float) - y0) / (y1 - y0)
    order = np.argsort(-frac, kind="stable")
    slots = np.empty(len(names))
    top = 1.0
    for rank, i in enumerate(order):
        top = min(frac[i], top - min_gap) if rank else min(frac[i], 1.0)
        slots[i] = top
    shift = max(0.0, 0.02 - slots.min())  # keep the column inside the axes height
    slots += shift
    for i, name in enumerate(names):
        ax.annotate(
            name,
            xy=(xs[i], ys[i]),
            xycoords="data",
            xytext=(x_text, slots[i]),
            textcoords="axes fraction",
            va="center",
            ha="left",
            fontsize=9,
            color=theme["ink"],
            arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.7},
            annotation_clip=False,
            zorder=6,
        )


def plot_report(
    a: Analysis,
    path: str | Path,
    named: dict[str, int] | None = None,
    title: str | None = None,
    theme: str = "light",
    dpi: int = 160,
) -> None:
    """Two-panel summary figure for one trait.

    Left: normalised information against node height (in signal half-lives)
    for every internal node, its monotone trend, and any named ancestors.
    Right: the profile log-likelihood of the overall rate.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    internal = a.internal
    half_life = a.best.half_life
    x = a.height[internal] / half_life
    y = a.info.normalised_information[internal]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.4),
        dpi=dpi,
        facecolor=t["surface"],
        layout="constrained",
        gridspec_kw={"wspace": 0.08},
    )

    # ---- panel 1: information against height
    style_axes(ax1, t)
    x_floor = min(0.03, float(x.max()) / 30)
    shown = x >= x_floor
    ax1.scatter(
        x[shown], y[shown], s=7, color=t["muted"], alpha=0.35, linewidths=0, label="internal nodes"
    )
    order = np.argsort(x, kind="stable")
    trend = isotonic_decreasing(y[order])
    keep = x[order] >= x_floor
    ax1.plot(
        x[order][keep], trend[keep], color=t["series"][0], linewidth=2, label="monotone trend"
    )
    ax1.set_xscale("log")
    plain_log_axis(ax1.xaxis)
    ax1.set_xlim(x_floor, float(x.max()) * 1.25)
    ax1.set_ylim(-0.03, 1.05)
    ax1.axhline(0.5, color=t["axis"], linewidth=0.8)
    ax1.annotate(
        "half of the prior uncertainty removed",
        xy=(0.01, 0.5),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8,
        color=t["muted"],
    )
    if named:
        nodes = np.array(list(named.values()))
        nx = a.height[nodes] / half_life
        ny = a.info.normalised_information[nodes]
        ax1.scatter(
            nx,
            ny,
            s=64,
            color=t["series"][1],
            edgecolors=t["surface"],
            linewidths=1.6,
            zorder=5,
            label="named ancestors",
        )
        label_column(ax1, nx, ny, list(named), t)
    ax1.set_xlabel("node height: mean distance to its tips, in signal half-lives")
    ax1.set_ylabel("normalised information")
    ax1.set_title("What the tips can say about each ancestor", fontsize=11, loc="left")
    legend(ax1, t, loc="lower left")

    # ---- panel 2: rate-scale profile
    style_axes(ax2, t)
    prof = a.profile
    rel = prof.loglik - prof.loglik.max()
    floor = -12.0
    ax2.plot(prof.scale, np.where(rel >= floor, rel, np.nan), color=t["series"][0], linewidth=2)
    ax2.axvspan(prof.ci_low, prof.ci_high, color=t["series"][0], alpha=0.10, linewidth=0)
    ax2.axhline(-1.92, color=t["axis"], linewidth=0.8)
    ax2.annotate(
        "95% interval",
        xy=(0.01, -1.92),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8,
        color=t["muted"],
    )
    if not prof.bounded_above:
        ax2.annotate(
            "no upper bound:\nsaturation cannot be excluded",
            xy=(0.98, 0.06),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            color=t["ink"],
        )
    ax2.set_xscale("log")
    plain_log_axis(ax2.xaxis, (prof.scale[0], prof.scale[-1]))
    ax2.set_xlim(prof.scale[0], prof.scale[-1])
    ax2.set_ylim(floor, 0.6)
    ax2.set_xlabel("multiplier applied to all fitted rates")
    ax2.set_ylabel("log-likelihood relative to the best")
    ax2.set_title("Is the overall rate pinned down?", fontsize=11, loc="left")

    if title:
        fig.suptitle(title, fontsize=12, color=t["ink"], x=0.01, ha="left")
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
