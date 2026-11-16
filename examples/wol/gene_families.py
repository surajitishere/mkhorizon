#!/usr/bin/env python
"""Part 2 of the worked example: gene-family presence/absence across 4,431 genomes.

    python examples/wol/gene_families.py [--n-random 400] [--jobs 2]

KEGG-orthologue (KO) copy numbers for IMG genomes are taken from the reference
tables shipped with PICRUSt2 v2.5.2 and reduced to presence/absence. Genomes are
matched to the Web of Life phylogeny through the IMG identifiers in its
metadata. The diagnostic is run on

* a hand-picked panel that spans vertically inherited to highly mobile genes, and
* a random sample of families in which both states occur at least 50 times,

and one row per family is written to ``results/gene_families.tsv``.
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from common import DATA, MIN_BRANCH, RESULTS, domain_nodes, load_metadata, load_tree

from mkhorizon.report import analyse, named_node_table, summary

PANEL = {
    "K02111": "atpA (F-type ATPase alpha)",
    "K00399": "mcrA (methyl-CoM reductase)",
    "K02406": "fliC (flagellin)",
    "K01601": "rbcL (RuBisCO large chain)",
    "K02588": "nifH (nitrogenase)",
    "K01428": "ureC (urease alpha)",
    "K17836": "penP (beta-lactamase A)",
    "K18220": "tetM (tetracycline resistance)",
    "K15342": "cas1 (CRISPR)",
    "K12056": "traG (conjugal transfer)",
    "K14059": "int (integrase)",
    "K07483": "transposase",
    "K01153": "hsdR (type I restriction)",
}
MIN_COUNT = 50

_STATE: dict = {}


def load_presence(meta: pd.DataFrame) -> pd.DataFrame:
    """Boolean genomes x KO table for Web of Life genomes with an IMG profile."""
    ko = pd.read_csv(DATA / "ko.txt.gz", sep="\t", index_col=0, dtype={"assembly": str})
    ko.index = ko.index.astype(str)
    known = set(ko.index)

    def first_match(cell) -> str | None:
        if not isinstance(cell, str):
            return None
        for part in cell.replace(";", ",").split(","):
            if part.strip() in known:
                return part.strip()
        return None

    matched = meta.assign(img=meta.img_id.map(first_match)).dropna(subset=["img"])
    presence = ko.loc[matched.img.values] > 0
    presence.index = matched.genome.values
    return presence


def _init(tree, presence, named) -> None:
    _STATE.update(tree=tree, presence=presence, named=named)


def _one(ko: str) -> dict:
    tree, presence, named = _STATE["tree"], _STATE["presence"], _STATE["named"]
    column = presence[ko]
    tip_states = {g: ("present" if v else "absent") for g, v in column.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = analyse(
            tree,
            tip_states,
            states=["absent", "present"],
            n_reps=200,
            n_perm=0,
            rate_sensitivity=False,
            min_branch_length=MIN_BRANCH,
            seed=1,
        )
        nodes = named_node_table(a, named).set_index("name")
    info = summary(a)
    row = {
        "ko": ko,
        "prevalence": float(column.mean()),
        "model": info["model"],
        "gain_rate": float(a.best.q[0, 1]),
        "loss_rate": float(a.best.q[1, 0]),
        "negligible_transitions": ";".join(info["negligible_transitions"]),
        "half_life": info["signal_half_life"],
        "saturation_index": info["saturation_index"],
        "loglik_gain_over_saturation": info["loglik_gain_over_saturation"],
        "rate_bounded_above": info["rate_scale_interval"]["bounded_above"],
        "fraction_informative": info["information"]["fraction_informative"],
        "fraction_uninformative": info["information"]["fraction_uninformative"],
        "horizon_status": info["horizon"]["status"],
        "horizon_half_lives": info["horizon"]["height_half_lives"],
    }
    for name in nodes.index:
        key = name.replace(" ", "_")
        row[f"ni_{key}"] = nodes.loc[name, "normalised_information"]
        row[f"accuracy_{key}"] = nodes.loc[name, "expected_accuracy"]
        row[f"baseline_{key}"] = nodes.loc[name, "baseline_accuracy"]
        row[f"post_present_{key}"] = nodes.loc[name, "post_present"]
        row[f"model_disagreement_{key}"] = nodes.loc[name, "model_disagreement"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-random", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    meta = load_metadata()
    presence = load_presence(meta)
    tree = load_tree().prune(presence.index)
    named = domain_nodes(meta, presence.index)
    counts = presence.sum()
    eligible = counts.index[(counts >= MIN_COUNT) & (len(presence) - counts >= MIN_COUNT)]
    rng = np.random.default_rng(args.seed)
    pool_ids = sorted(set(eligible) - set(PANEL))
    sample = list(rng.choice(pool_ids, size=min(args.n_random, len(pool_ids)), replace=False))
    todo = [k for k in PANEL if k in presence.columns] + sample
    print(f"{len(presence)} genomes, {len(eligible)} eligible families, running {len(todo)}")

    descriptions = pd.read_csv(
        DATA / "ko_info.tsv.gz", sep="\t", header=None, names=["ko", "description"], dtype=str
    ).set_index("ko")["description"]

    with Pool(args.jobs, initializer=_init, initargs=(tree, presence[todo], named)) as pool:
        rows = []
        for i, row in enumerate(pool.imap(_one, todo, chunksize=4), 1):
            rows.append(row)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    table = pd.DataFrame(rows)
    table.insert(1, "set", ["panel" if k in PANEL else "random" for k in table.ko])
    table.insert(2, "label", [PANEL.get(k, "") for k in table.ko])
    table.insert(3, "description", [descriptions.get(k, "") for k in table.ko])
    table.to_csv(RESULTS / "gene_families.tsv", sep="\t", index=False, float_format="%.5g")
    print(f"wrote {RESULTS / 'gene_families.tsv'}")


if __name__ == "__main__":
    main()
