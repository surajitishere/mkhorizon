"""Shared loading code for the Web of Life worked example."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mkhorizon import Tree, read_newick

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

#: Smallest branch length used. A handful of genomes in the reference tree are
#: effectively identical; without a floor, two such genomes that differ in a
#: trait would have likelihood zero under any Mk model.
MIN_BRANCH = 1e-6


def load_tree() -> Tree:
    return read_newick(DATA / "astral.cons.nid.nwk")


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(DATA / "metadata.tsv.xz", sep="\t", dtype=str)
    return meta.rename(columns={"#genome": "genome"})


def domain_nodes(meta: pd.DataFrame, genomes) -> dict[str, list[str]]:
    """Named ancestors: the crown node of each domain and the root between them."""
    have = meta[meta.genome.isin(set(genomes))]
    return {
        "root": have.genome.tolist(),
        "crown Bacteria": have.loc[have.superkingdom == "Bacteria", "genome"].tolist(),
        "crown Archaea": have.loc[have.superkingdom == "Archaea", "genome"].tolist(),
    }
