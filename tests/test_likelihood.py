import numpy as np
import pytest

from helpers import brute_force, one_hot, random_tree
from mkhorizon.likelihood import Engine, entropy_bits, tip_likelihoods
from mkhorizon.model import build_q, stationary_distribution
from mkhorizon.tree import parse_newick


@pytest.mark.parametrize("trial", range(24))
def test_likelihood_and_marginals_match_exhaustive_enumeration(trial):
    rng = np.random.default_rng(100 + trial)
    k = 2 + trial % 2
    n_tips = int(rng.integers(3, 8))
    tree = random_tree(n_tips, rng, polytomies=trial % 3 == 0)
    q = build_q(k, "ARD", rng.exponential(1.0, k * (k - 1)))
    prior = rng.dirichlet(np.ones(k))
    tip_like = one_hot(rng.integers(0, k, n_tips), k)
    if trial % 4 == 0:
        tip_like[int(rng.integers(0, n_tips))] = 1.0  # one tip with missing data
    expected_ll, expected_marg = brute_force(tree, q, tip_like, prior)

    binary = tree.binarize()
    loglik, post = Engine(binary).reconstruct(q, tip_like, prior)
    real = np.flatnonzero(~binary.virtual)
    internal = np.flatnonzero(~tree.is_tip)
    assert np.isclose(loglik[0], expected_ll, atol=1e-10)
    assert np.allclose(post[0][real][internal], expected_marg[internal], atol=1e-10)


def test_two_tip_likelihood_by_hand():
    tree = parse_newick("(A:0.3,B:0.5);")
    q = build_q(2, "ER", [1.0])
    stay = lambda t: 0.5 + 0.5 * np.exp(-2 * t)  # noqa: E731
    change = lambda t: 0.5 - 0.5 * np.exp(-2 * t)  # noqa: E731
    like = one_hot(np.array([0, 1]), 2)
    expected = np.log(0.5 * stay(0.3) * change(0.5) + 0.5 * change(0.3) * stay(0.5))
    assert np.isclose(Engine(tree).loglik(q, like, "equal")[0], expected)


def test_root_position_is_irrelevant_for_a_reversible_model_at_stationarity():
    # The same unrooted tree, rooted on the internal edge and on the branch to A.
    mid = parse_newick("((A:0.1,B:0.2):0.05,(C:0.3,D:0.4):0.15);")
    on_a = parse_newick("(A:0.04,(B:0.2,(C:0.3,D:0.4):0.2):0.06);")
    q = build_q(3, "SYM", [0.4, 1.1, 0.7])
    states = {"A": "x", "B": "y", "C": "y", "D": "z"}
    values = []
    for tree in (mid, on_a):
        like = tip_likelihoods(tree, states, ["x", "y", "z"])
        values.append(Engine(tree).loglik(q, like, "stationary")[0])
    assert np.isclose(values[0], values[1], atol=1e-12)


def test_missing_tip_is_equivalent_to_pruning_it():
    rng = np.random.default_rng(7)
    tree = random_tree(12, rng)
    q = build_q(2, "ARD", [0.6, 1.4])
    states = {name: "ab"[int(rng.integers(0, 2))] for name in tree.tip_names}
    states["t3"] = None
    pruned = tree.prune([t for t in tree.tip_names if t != "t3"])
    # Pruning can move the root; a stationary prior makes the two comparable.
    with_missing = Engine(tree).loglik(q, tip_likelihoods(tree, states, ["a", "b"]), "stationary")
    without = Engine(pruned).loglik(q, tip_likelihoods(pruned, states, ["a", "b"]), "stationary")
    assert np.isclose(with_missing[0], without[0], atol=1e-10)


def test_batched_evaluation_equals_one_at_a_time():
    rng = np.random.default_rng(8)
    tree = random_tree(25, rng)
    engine = Engine(tree)
    q = build_q(3, "ARD", rng.exponential(1.0, 6))
    batch = np.stack([one_hot(rng.integers(0, 3, 25), 3) for _ in range(5)])
    loglik, post = engine.reconstruct(q, batch, "fitzjohn")
    for b in range(5):
        one_ll, one_post = engine.reconstruct(q, batch[b], "fitzjohn")
        assert np.isclose(one_ll[0], loglik[b]) and np.allclose(one_post[0], post[b])


def test_posteriors_are_distributions_and_observed_tips_are_certain():
    rng = np.random.default_rng(9)
    tree = random_tree(30, rng)
    engine = Engine(tree)
    states = rng.integers(0, 2, 30)
    _, post = engine.reconstruct(build_q(2, "ER", [0.8]), one_hot(states, 2), "equal")
    assert np.allclose(post.sum(axis=2), 1.0)
    assert np.array_equal(post[0][tree.tips].argmax(axis=1), states)
    assert np.allclose(post[0][tree.tips].max(axis=1), 1.0)


def test_zero_length_branches_and_impossible_data():
    tree = parse_newick("((A:0,B:0):1,C:1);")
    engine = Engine(tree)
    q = build_q(2, "ER", [0.5])
    same = engine.loglik(q, one_hot(np.array([0, 0, 1]), 2), "equal")[0]
    assert np.isfinite(same)
    # A and B sit at zero distance from their ancestor, so they cannot differ
    assert engine.loglik(q, one_hot(np.array([0, 1, 1]), 2), "equal")[0] == -np.inf


def test_prior_marginals_follow_the_chain():
    rng = np.random.default_rng(10)
    tree = random_tree(15, rng)
    engine = Engine(tree)
    q = build_q(2, "ARD", [0.3, 1.5])
    pi = stationary_distribution(q)
    assert np.allclose(engine.prior_marginals(q, pi), np.tile(pi, (tree.n_nodes, 1)))
    start = np.array([1.0, 0.0])
    marg = engine.prior_marginals(q, start)
    depth = tree.depths()
    expected0 = pi[0] + pi[1] * np.exp(-(0.3 + 1.5) * depth)
    assert np.allclose(marg[:, 0], expected0)


def test_parsimony_score():
    tree = parse_newick("(((A:1,B:1):1,C:1):1,(D:1,E:1):1);")
    engine = Engine(tree)
    assert engine.parsimony_changes(one_hot(np.array([0, 0, 0, 0, 0]), 2)) == 0
    assert engine.parsimony_changes(one_hot(np.array([0, 0, 0, 1, 1]), 2)) == 1
    assert engine.parsimony_changes(one_hot(np.array([0, 1, 0, 1, 0]), 2)) == 2


def test_engine_requires_a_binary_tree_and_known_states():
    with pytest.raises(ValueError, match="binary"):
        Engine(parse_newick("(A:1,B:1,C:1);"))
    tree = parse_newick("(A:1,B:1);")
    with pytest.raises(KeyError):
        tip_likelihoods(tree, {"A": "x", "B": "oops"}, ["x", "y"])
    with pytest.raises(ValueError):
        Engine(tree).loglik(build_q(2, "ER", [1.0]), np.ones((2, 2)), "nonsense")


def test_entropy_bits():
    assert np.isclose(entropy_bits(np.array([0.5, 0.5])), 1.0)
    assert entropy_bits(np.array([1.0, 0.0])) == 0.0
    assert np.isclose(entropy_bits(np.full(4, 0.25)), 2.0)
