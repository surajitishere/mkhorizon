#!/usr/bin/env python
"""Part 1 of the worked example: five classic phenotypes on the Web of Life tree.

    python examples/wol/phenotypes.py

Phenotypes come from the species-level synthesis of Madin et al. (2020) and are
joined to the 10,575-genome reference phylogeny of Zhu et al. (2019) through
NCBI species taxonomy IDs. Each trait is reduced to two states; species with
other or missing values are left out.
"""

from __future__ import annotations

import json

import pandas as pd
from common import DATA, MIN_BRANCH, RESULTS, domain_nodes, load_metadata, load_tree

from mkhorizon.plot import plot_report
from mkhorizon.report import analyse, named_node_table, node_table, summary

#: trait name -> (column in the Madin table, {raw value: state})
TRAITS = {
    "gram_stain": ("gram_stain", {"negative": "negative", "positive": "positive"}),
    "sporulation": ("sporulation", {"no": "no", "yes": "yes"}),
    "motility": (
        "motility",
        {
            "no": "nonmotile",
            "yes": "motile",
            "flagella": "motile",
            "gliding": "motile",
            "axial filament": "motile",
        },
    ),
    "cell_shape": ("cell_shape", {"bacillus": "rod", "coccus": "coccus"}),
    "thermophily": (
        "range_tmp",
        {
            "mesophilic": "mesophile",
            "psychrophilic": "mesophile",
            "psychrotolerant": "mesophile",
            "facultative psychrophilic": "mesophile",
            "thermotolerant": "mesophile",
            "thermophilic": "thermophile",
            "extreme thermophilic": "thermophile",
        },
    ),
}


def main() -> None:
    out = RESULTS / "phenotypes"
    out.mkdir(parents=True, exist_ok=True)
    tree = load_tree()
    meta = load_metadata()
    madin = pd.read_csv(
        DATA / "condensed_species_NCBI.csv", dtype={"species_tax_id": str}, low_memory=False
    )
    joined = meta.merge(madin, left_on="species_taxid", right_on="species_tax_id", how="left")

    rows = []
    for trait, (column, coding) in TRAITS.items():
        states = joined[column].map(coding)
        tip_states = {g: s for g, s in zip(joined.genome, states, strict=True) if isinstance(s, str)}
        a = analyse(
            tree, tip_states, n_reps=400, n_perm=199, min_branch_length=MIN_BRANCH, seed=1
        )
        named = named_node_table(a, domain_nodes(meta, tip_states))
        folder = out / trait
        folder.mkdir(exist_ok=True)
        node_table(a).to_csv(folder / "nodes.tsv", sep="\t", index=False, float_format="%.6g")
        named.to_csv(folder / "named_nodes.tsv", sep="\t", index=False, float_format="%.6g")
        info = summary(a)
        (folder / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
        plot_report(
            a,
            folder / "report.png",
            named=dict(zip(named.name, named.node.astype(int), strict=True)),
            title=trait,
        )
        by_name = named.set_index("name")
        rows.append(
            {
                "trait": trait,
                "n_tips": info["n_tips"],
                "states": "/".join(f"{s}={c}" for s, c in info["state_counts"].items()),
                "model": info["model"],
                "negligible_transitions": ";".join(info["negligible_transitions"]),
                "half_life": info["signal_half_life"],
                "saturation_index": info["saturation_index"],
                "rate_interval_low": info["rate_scale_interval"]["low"],
                "rate_interval_high": info["rate_scale_interval"]["high"],
                "rate_bounded_above": info["rate_scale_interval"]["bounded_above"],
                "signal_p": info["signal_test"]["p_value"],
                "fraction_informative": info["information"]["fraction_informative"],
                **{
                    f"ni_{name.replace(' ', '_')}": by_name.loc[name, "normalised_information"]
                    for name in by_name.index
                },
                **{
                    f"map_{name.replace(' ', '_')}": (
                        f"{by_name.loc[name, 'map_state']} ({by_name.loc[name, 'map_posterior']:.2f})"
                    )
                    for name in by_name.index
                },
                **{
                    f"model_disagreement_{name.replace(' ', '_')}": by_name.loc[
                        name, "model_disagreement"
                    ]
                    for name in by_name.index
                },
            }
        )
        print(f"{trait}: done ({info['n_tips']} tips)")
    table = pd.DataFrame(rows)
    table.to_csv(out / "phenotypes.tsv", sep="\t", index=False, float_format="%.4g")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
