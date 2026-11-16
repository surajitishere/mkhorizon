"""Shared helpers for the test-suite (no fixtures with hidden state)."""

from __future__ import annotations

import itertools

import numpy as np

from mkhorizon.model import transition_probabilities
from mkhorizon.tree import Tree, parse_newick


def random_newick(n_tips: int, rng: np.random.Generator, polytomies: bool = False, scale: float = 0.3) -> str:
    """A random rooted tree built by repeatedly joining random lineages."""
    nodes = [f"t{i}" for i in range(n_tips)]
    while len(nodes) > 1:
        m = 2
        if polytomies and len(nodes) >= 3:
            m = int(rng.integers(2, min(4, len(nodes)) + 1))
        picked = sorted(rng.choice(len(nodes), size=m, replace=False), reverse=True)
        parts = [nodes.pop(i) for i in picked]
        nodes.append("(" + ",".join(f"{p}:{rng.exponential(scale):.6f}" for p in parts) + ")")
    return nodes[0] + ";"


def random_tree(n_tips: int, rng: np.random.Generator, **kwargs) -> Tree:
    return parse_newick(random_newick(n_tips, rng, **kwargs))


def one_hot(states: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((states.size, k))
    out[np.arange(states.size), states] = 1.0
    return out


def brute_force(tree: Tree, q: np.ndarray, tip_like: np.ndarray, prior: np.ndarray):
    """Log-likelihood and node marginals by summing over every internal assignment."""
    n, k = tree.n_nodes, q.shape[0]
    p = transition_probabilities(q, tree.length)
    internal = np.flatnonzero(~tree.is_tip)
    tip_row = {int(t): tip_like[i] for i, t in enumerate(tree.tips)}
    total = 0.0
    marginal = np.zeros((n, k))
    for assignment in itertools.product(range(k), repeat=internal.size):
        state = dict(zip(internal.tolist(), assignment, strict=True))
        prob = prior[state[0]]
        for v in range(1, n):
            par = state[int(tree.parent[v])]
            prob *= p[v][par, state[v]] if v in state else (p[v][par] * tip_row[v]).sum()
        total += prob
        for v, s in state.items():
            marginal[v, s] += prob
    return np.log(total), marginal / total


def pairwise_tip_distances(tree: Tree) -> dict[tuple[str, str], float]:
    depth = tree.depths()
    out = {}
    tips = tree.tips
    for i, a in enumerate(tips):
        for b in tips[i + 1 :]:
            m = tree.mrca([a, b])
            key = tuple(sorted((tree.names[a], tree.names[b])))
            out[key] = depth[a] + depth[b] - 2 * depth[m]
    return out
