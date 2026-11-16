#!/usr/bin/env python
"""Print the headline numbers quoted in the README from results/gene_families.tsv.

    python examples/wol/summarise.py > examples/wol/results/gene_families_summary.txt
"""

from __future__ import annotations

import pandas as pd
from common import RESULTS
from scipy.stats import spearmanr

NODES = {"root": "root", "crown Bacteria": "crown_Bacteria", "crown Archaea": "crown_Archaea"}


def main() -> None:
    table = pd.read_csv(RESULTS / "gene_families.tsv", sep="\t")
    random = table[table.set == "random"]
    n = len(random)
    print(f"random gene families: {n}")
    print(f"best model: {random.model.value_counts().to_dict()}")
    forbids = random.negligible_transitions.fillna("").ne("").sum()
    print(f"fitted model effectively forbids a transition: {forbids} ({forbids / n:.1%})")
    q = random.saturation_index.quantile([0.25, 0.5, 0.75])
    print(f"saturation index, quartiles: {q.iloc[0]:.2f} / {q.iloc[1]:.2f} / {q.iloc[2]:.2f}")
    print()
    print(f"{'node':16s} {'informative':>12s} {'weak':>8s} {'uninformative':>14s} "
          f"{'median NI':>10s} {'model-sensitive':>16s} {'informative AND model-robust':>30s}")  # fmt: skip
    for label, key in NODES.items():
        ni = random[f"ni_{key}"]
        gap = random[f"model_disagreement_{key}"]
        informative = ni >= 0.5
        print(
            f"{label:16s} {informative.mean():12.1%} {((ni >= 0.1) & ~informative).mean():8.1%} "
            f"{(ni < 0.1).mean():14.1%} {ni.median():10.2f} {(gap > 0.5).mean():16.1%} "
            f"{(informative & (gap <= 0.1)).mean():30.1%}"
        )
    print()
    rho = spearmanr(random.saturation_index, random.ni_crown_Bacteria).statistic
    print(f"Spearman correlation, saturation index vs NI at crown Bacteria: {rho:.2f}")
    confident = (random.post_present_crown_Bacteria >= 0.9) | (random.post_present_crown_Bacteria <= 0.1)
    flips = confident & (random.model_disagreement_crown_Bacteria > 0.5)
    print(
        f"crown Bacteria: posterior >= 0.9 for one state in {confident.mean():.1%} of families; "
        f"in {flips.sum() / confident.sum():.1%} of those the other model class moves it by > 0.5"
    )
    print()
    print("model-sensitive = |P(present) under ARD - under ER| > 0.5; model-robust = that gap <= 0.1")


if __name__ == "__main__":
    main()
