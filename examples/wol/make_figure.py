#!/usr/bin/env python
"""Summary figure for the gene-family sweep (reads results/gene_families.tsv).

    python examples/wol/make_figure.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import HERE, RESULTS

from mkhorizon.plot import THEMES, label_column, legend, plain_log_axis, style_axes

LABELLED = ["K02111", "K00399", "K02406", "K02588", "K17836", "K15342", "K07483", "K01153"]


def main(theme: str = "light") -> None:
    t = THEMES[theme]
    table = pd.read_csv(RESULTS / "gene_families.tsv", sep="\t")
    random = table[table.set == "random"]
    panel = table[table.ko.isin(LABELLED)].set_index("ko").loc[LABELLED]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.6), dpi=160, facecolor=t["surface"], layout="constrained",
        gridspec_kw={"wspace": 0.1},
    )  # fmt: skip

    # ---- left: saturation against information at the bacterial crown node
    style_axes(ax1, t)
    ax1.scatter(
        random.saturation_index, random.ni_crown_Bacteria, s=14, color=t["muted"], alpha=0.45,
        linewidths=0, label=f"{len(random)} random gene families",
    )  # fmt: skip
    ax1.scatter(
        panel.saturation_index, panel.ni_crown_Bacteria, s=64, color=t["series"][1],
        edgecolors=t["surface"], linewidths=1.6, zorder=5, label="hand-picked families",
    )  # fmt: skip
    ax1.set_xscale("log")
    plain_log_axis(ax1.xaxis)
    ax1.set_ylim(-0.03, 1.05)
    label_column(
        ax1, panel.saturation_index.values, panel.ni_crown_Bacteria.values,
        [s.split(" (")[0] for s in panel.label], t, min_gap=0.085,
    )  # fmt: skip
    ax1.set_xlabel("saturation index: root-to-tip distance in signal half-lives")
    ax1.set_ylabel("normalised information, crown Bacteria")
    ax1.set_title("Fast-turnover families leave nothing to reconstruct", fontsize=11, loc="left")
    legend(ax1, t, loc="lower left")

    # ---- right: does the answer survive a change of model class?
    style_axes(ax2, t)
    ax2.scatter(
        random.ni_crown_Bacteria, random.model_disagreement_crown_Bacteria, s=14,
        color=t["series"][0], alpha=0.55, linewidths=0,
    )  # fmt: skip
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("normalised information, crown Bacteria (under the best model)")
    ax2.set_ylabel("|P(present) under ARD - under ER|")
    ax2.set_title("High information does not mean model-robust", fontsize=11, loc="left")

    out = HERE / f"gene_families_{theme}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main("light")
    main("dark")
