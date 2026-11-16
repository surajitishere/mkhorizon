#!/usr/bin/env python
"""Generate the toy data set used in the README quick start and in the tests.

    python examples/toy/make_toy.py

A 120-tip pure-birth (Yule) tree of height 1 carries two simulated binary traits:

* ``slow`` -- signal half-life 1.0, so the root is one half-life from the tips;
* ``fast`` -- signal half-life 0.03, about 33 half-lives from root to tip.

A simulated trait is redrawn until at least 15% of the tips carry the rarer
state, so both example traits are worth reconstructing.

Ten per cent of the values are blanked out to mimic missing data. Everything is
seeded, so the committed files can be regenerated exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mkhorizon import Engine, build_q, parse_newick
from mkhorizon.simulate import simulate_states

HERE = Path(__file__).parent
N_TIPS = 120
SEED = 20261110


def yule_newick(n: int, rng: np.random.Generator) -> str:
    """Pure-birth (Yule) tree with ``n`` tips, rescaled to unit height."""
    # Each lineage is [newick so far, time it was born]; splits happen at rate 1.
    lineages = [["", 0.0], ["", 0.0]]
    children: dict[int, list] = {}
    nodes = [{"parent": None, "born": 0.0}, {"parent": None, "born": 0.0}]
    active = [0, 1]
    now = 0.0
    while len(active) < n:
        now += rng.exponential(1.0 / len(active))
        parent = active.pop(int(rng.integers(len(active))))
        nodes[parent]["died"] = now
        for _ in range(2):
            nodes.append({"parent": parent, "born": now})
            children.setdefault(parent, []).append(len(nodes) - 1)
            active.append(len(nodes) - 1)
    now += rng.exponential(1.0 / n)  # let the last split grow terminal branches
    for i in active:
        nodes[i]["died"] = now
    del lineages

    names = iter(rng.permutation([f"sp{i + 1:03d}" for i in range(n)]))

    def render(i: int) -> str:
        length = (nodes[i]["died"] - nodes[i]["born"]) / now
        if i not in children:
            return f"{next(names)}:{length:.8f}"
        return "(" + ",".join(render(c) for c in children[i]) + f"):{length:.8f}"

    return "(" + render(0) + "," + render(1) + ");"


def main() -> None:
    rng = np.random.default_rng(SEED)
    newick = yule_newick(N_TIPS, rng)
    (HERE / "tree.nwk").write_text(newick + "\n")
    tree = parse_newick(newick)
    engine = Engine(tree)
    columns = {}
    for name, half_life in (("slow", 1.0), ("fast", 0.03)):
        rate = np.log(2) / half_life / 2  # two states, equal rates: gap = 2 * rate
        q = build_q(2, "ER", [rate])
        while True:
            states = simulate_states(engine, q, np.array([0.5, 0.5]), 1, rng)[0]
            share = states[tree.tips].mean()
            if 0.15 <= share <= 0.85:
                break
        values = np.where(states[tree.tips] == 1, "present", "absent").astype(object)
        values[rng.random(N_TIPS) < 0.10] = ""
        columns[name] = values
    lines = ["taxon\tslow\tfast"]
    for i, tip in enumerate(tree.tip_names):
        lines.append(f"{tip}\t{columns['slow'][i]}\t{columns['fast'][i]}")
    (HERE / "traits.tsv").write_text("\n".join(lines) + "\n")

    # Three named ancestors: the root and its two daughters.
    left, right = tree.children[0]
    def tips_of(node: int) -> str:
        return ",".join(tree.names[t] for t in tree.descendants_tips(node))

    nodes = [
        "# name<TAB>comma-separated tips; the ancestor is their MRCA",
        f"root\t{tips_of(0)}",
        f"clade A\t{tips_of(left)}",
        f"clade B\t{tips_of(right)}",
    ]
    (HERE / "nodes.tsv").write_text("\n".join(nodes) + "\n")
    print(f"wrote tree.nwk, traits.tsv and nodes.tsv to {HERE}")


if __name__ == "__main__":
    main()
