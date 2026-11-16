import numpy as np

from helpers import random_tree
from mkhorizon.likelihood import Engine
from mkhorizon.model import build_q
from mkhorizon.simulate import simulate_states, states_to_tip_likelihoods


def test_simulated_frequencies_match_the_analytic_marginals():
    rng = np.random.default_rng(11)
    tree = random_tree(30, rng)
    engine = Engine(tree)
    q = build_q(3, "ARD", [0.5, 0.2, 1.0, 0.3, 0.7, 0.4])
    prior = np.array([0.6, 0.3, 0.1])
    n_reps = 40_000
    states = simulate_states(engine, q, prior, n_reps, rng)
    observed = np.stack([(states == s).mean(axis=0) for s in range(3)], axis=1)
    expected = engine.prior_marginals(q, prior)
    # binomial standard error is at most 0.0025 here; allow five of them
    assert np.abs(observed - expected).max() < 0.0125


def test_parent_child_transitions_follow_the_branch_matrix():
    rng = np.random.default_rng(12)
    tree = random_tree(6, rng)
    engine = Engine(tree)
    q = build_q(2, "ARD", [0.9, 0.4])
    states = simulate_states(engine, q, np.array([0.5, 0.5]), 60_000, rng)
    p = engine.transition(q)
    for child in range(1, tree.n_nodes):
        from_zero = states[:, tree.parent[child]] == 0
        observed = states[from_zero, child].mean()
        assert abs(observed - p[child][0, 1]) < 0.012


def test_simulation_is_reproducible_and_seed_dependent():
    tree = random_tree(20, np.random.default_rng(13))
    engine = Engine(tree)
    q = build_q(2, "ER", [1.0])
    prior = np.array([0.5, 0.5])
    a = simulate_states(engine, q, prior, 8, np.random.default_rng(1))
    b = simulate_states(engine, q, prior, 8, np.random.default_rng(1))
    c = simulate_states(engine, q, prior, 8, np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_tip_likelihoods_from_states_and_missing_mask():
    states = np.array([[0, 1, 2, 1, 0], [2, 2, 0, 1, 1]])
    tips = np.array([1, 3, 4])
    like = states_to_tip_likelihoods(states, tips, 3)
    assert like.shape == (2, 3, 3)
    assert np.array_equal(like.argmax(axis=2), states[:, tips])
    assert np.allclose(like.sum(axis=2), 1.0)
    hidden = states_to_tip_likelihoods(states, tips, 3, observed=np.array([True, False, True]))
    assert np.all(hidden[:, 1] == 1.0) and np.array_equal(hidden[:, 0], like[:, 0])
