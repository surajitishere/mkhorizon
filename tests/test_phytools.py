"""Cross-validation against phytools (R).

``tests/data/phytools_reference.json`` was produced by
``validation/phytools_reference.R``; see that script for what it contains.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mkhorizon import Engine, fit_mk, parse_newick, tip_likelihoods

REFERENCE = json.loads((Path(__file__).parent / "data" / "phytools_reference.json").read_text())
CASES = [
    (case, fit)
    for case in REFERENCE["cases"]
    for fit in case["fits"]
]
IDS = [f"{case['name']}-{fit['model']}-{fit['root_prior']}" for case, fit in CASES]


def _setup(case):
    tree = parse_newick(case["newick"]).binarize()
    engine = Engine(tree)
    tip_like = tip_likelihoods(tree, case["tip_states"], case["states"])
    by_clade = {
        "|".join(sorted(tree.names[t] for t in tree.descendants_tips(int(v)))): int(v)
        for v in np.flatnonzero(~tree.is_tip)
    }
    return tree, engine, tip_like, by_clade


@pytest.mark.parametrize(("case", "fit"), CASES, ids=IDS)
def test_likelihood_at_the_phytools_rate_matrix(case, fit):
    _, engine, tip_like, _ = _setup(case)
    loglik = engine.loglik(np.array(fit["q"]), tip_like, fit["root_prior"])[0]
    assert loglik == pytest.approx(fit["loglik"], abs=1e-7)


@pytest.mark.parametrize(("case", "fit"), CASES, ids=IDS)
def test_marginal_ancestral_states_match_ancr(case, fit):
    tree, engine, tip_like, by_clade = _setup(case)
    _, post = engine.reconstruct(np.array(fit["q"]), tip_like, fit["root_prior"])
    assert len(fit["posterior"]) == tree.n_tips - 1  # every internal node is compared
    for clade, expected in fit["posterior"].items():
        assert post[0][by_clade[clade]] == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(("case", "fit"), CASES, ids=IDS)
def test_optimiser_reaches_the_phytools_optimum(case, fit):
    _, engine, tip_like, _ = _setup(case)
    mine = fit_mk(engine, tip_like, case["states"], fit["model"], fit["root_prior"])
    assert mine.loglik >= fit["loglik"] - 1e-3
