import numpy as np
import pytest
from scipy.linalg import expm

from mkhorizon.model import (
    MkModel,
    build_q,
    n_rate_params,
    signal_half_life,
    stationary_distribution,
    transition_probabilities,
)


def test_rate_matrix_layouts():
    assert np.allclose(build_q(2, "ER", [0.5]), [[-0.5, 0.5], [0.5, -0.5]])
    q = build_q(2, "ARD", [0.2, 0.7])  # q01, q10
    assert np.allclose(q, [[-0.2, 0.2], [0.7, -0.7]])
    q = build_q(3, "SYM", [1.0, 2.0, 3.0])  # q01, q02, q12
    assert np.allclose(q, q.T) and q[0, 1] == 1.0 and q[0, 2] == 2.0 and q[1, 2] == 3.0
    q = build_q(3, "ARD", [1, 2, 3, 4, 5, 6])
    assert np.allclose(q[~np.eye(3, dtype=bool)], [1, 2, 3, 4, 5, 6])
    for model in ("ER", "SYM", "ARD"):
        q = build_q(4, model, np.ones(n_rate_params(4, model)))
        assert np.allclose(q.sum(axis=1), 0.0)


def test_rate_matrix_validation():
    with pytest.raises(ValueError):
        build_q(2, "ARD", [1.0])
    with pytest.raises(ValueError):
        build_q(2, "ER", [-1.0])
    with pytest.raises(ValueError):
        build_q(2, "XYZ", [1.0])


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("model", ["ER", "SYM", "ARD"])
def test_transition_probabilities_match_matrix_exponential(k, model):
    rng = np.random.default_rng(k * 10 + len(model))
    q = build_q(k, model, rng.exponential(1.0, n_rate_params(k, model)))
    t = np.array([0.0, 1e-6, 0.3, 2.0, 50.0])
    p = transition_probabilities(q, t)
    assert np.allclose(p, np.stack([expm(q * x) for x in t]), atol=1e-10)
    assert np.allclose(p.sum(axis=2), 1.0)
    assert np.allclose(p[0], np.eye(k))


def test_two_state_closed_form():
    a, b = 0.3, 1.2
    q = build_q(2, "ARD", [a, b])
    t = np.array([0.1, 1.0, 4.0])
    p = transition_probabilities(q, t)
    pi1 = a / (a + b)
    assert np.allclose(p[:, 0, 1], pi1 * (1 - np.exp(-(a + b) * t)))
    assert np.allclose(stationary_distribution(q), [b / (a + b), pi1])


def test_stationary_distribution_is_a_left_null_vector():
    rng = np.random.default_rng(5)
    q = build_q(4, "ARD", rng.exponential(1.0, 12))
    pi = stationary_distribution(q)
    assert np.allclose(pi @ q, 0.0, atol=1e-12) and np.isclose(pi.sum(), 1.0)
    assert np.allclose(transition_probabilities(q, [500.0])[0], np.tile(pi, (4, 1)), atol=1e-8)


def test_half_life_halves_the_memory_of_the_starting_state():
    a, b = 0.4, 0.9
    q = build_q(2, "ARD", [a, b])
    h = signal_half_life(q)
    assert np.isclose(h, np.log(2) / (a + b))
    p0 = transition_probabilities(q, [0.0, h, 2 * h])
    gap = p0[:, 0, 0] - p0[:, 1, 0]  # dependence on where the lineage started
    assert np.allclose(gap, [1.0, 0.5, 0.25])
    assert signal_half_life(np.zeros((2, 2))) == float("inf")


def test_model_container_validates_and_scales():
    q = build_q(2, "ARD", [0.2, 0.7])
    m = MkModel(["a", "b"], q, np.array([0.5, 0.5]), "ARD")
    assert np.isclose(m.scaled(3.0).half_life, m.half_life / 3.0)
    with pytest.raises(ValueError):
        MkModel(["a", "b"], q, np.array([0.9, 0.9]))
    with pytest.raises(ValueError):
        MkModel(["a", "b"], np.ones((2, 2)), np.array([0.5, 0.5]))
