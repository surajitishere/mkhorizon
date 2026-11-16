import numpy as np
import pytest
from scipy.stats import binom

from helpers import random_tree
from mkhorizon.diagnose import (
    expected_information,
    horizon,
    isotonic_decreasing,
    scale_profile,
    signal_test,
    verdict,
)
from mkhorizon.fit import fit_mk, saturated_loglik
from mkhorizon.likelihood import Engine, entropy_bits
from mkhorizon.model import MkModel, build_q
from mkhorizon.simulate import simulate_states, states_to_tip_likelihoods
from mkhorizon.tree import parse_newick


def _model(rate: float) -> MkModel:
    return MkModel(["a", "b"], build_q(2, "ER", [rate]), np.array([0.5, 0.5]), "ER")


def _exact_star_information(n: int, t: float, rate: float) -> float:
    """I(root; tips) for n tips hanging off the root at distance t, by enumeration.

    The tips are independent given the root, so the number of tips in state b
    is sufficient and binomial.
    """
    stay = 0.5 + 0.5 * np.exp(-2 * rate * t)
    m = np.arange(n + 1)
    given_a = binom.pmf(m, n, 1 - stay)
    given_b = binom.pmf(m, n, stay)
    marginal = 0.5 * (given_a + given_b)
    post_a = 0.5 * given_a / marginal
    cond_entropy = (marginal * entropy_bits(np.stack([post_a, 1 - post_a], axis=1))).sum()
    return 1.0 - cond_entropy


@pytest.mark.parametrize(("n", "t", "rate"), [(6, 0.5, 1.0), (12, 1.0, 0.8), (30, 1.5, 1.0)])
def test_information_matches_the_exact_value_on_a_star_tree(n, t, rate):
    newick = "(" + ",".join(f"t{i}:{t}" for i in range(n)) + ");"
    engine = Engine(parse_newick(newick).binarize())
    info = expected_information(engine, _model(rate), n_reps=6000, seed=1, chunk=1000)
    exact = _exact_star_information(n, t, rate)
    assert abs(info.information[0] - exact) < 5 * info.conditional_entropy_se[0] + 1e-3
    assert info.prior_entropy[0] == pytest.approx(1.0)


def test_expected_accuracy_is_calibrated():
    rng = np.random.default_rng(21)
    engine = Engine(random_tree(40, rng))
    info = expected_information(engine, _model(1.2), n_reps=4000, seed=2, chunk=1000)
    internal = ~engine.tree.is_tip
    gap = np.abs(info.accuracy - info.accuracy_empirical)[internal]
    assert gap.max() < 0.035  # ~4.4 binomial standard errors at 4000 replicates
    assert np.all(info.accuracy[internal] >= info.baseline_accuracy[internal] - 1e-9)


def test_information_limits_slow_and_fast():
    engine = Engine(random_tree(40, np.random.default_rng(22)))
    internal = ~engine.tree.is_tip
    slow = expected_information(engine, _model(1e-4), n_reps=200, seed=3)
    fast = expected_information(engine, _model(1e4), n_reps=200, seed=3)
    assert slow.normalised_information[internal].min() > 0.98
    assert fast.normalised_information[internal].max() < 0.02
    assert np.allclose(fast.accuracy[internal], 0.5, atol=0.02)
    # tips are observed, so the data resolve them completely
    tips = engine.tree.tips
    assert np.allclose(slow.information[tips], slow.prior_entropy[tips])


def test_hiding_tips_loses_information():
    rng = np.random.default_rng(23)
    engine = Engine(random_tree(60, rng))
    observed = rng.random(60) < 0.3
    full = expected_information(engine, _model(0.8), n_reps=1500, seed=4, chunk=500)
    sparse = expected_information(engine, _model(0.8), observed, n_reps=1500, seed=4, chunk=500)
    assert sparse.information[0] < full.information[0]
    assert sparse.information.sum() < full.information.sum()
    hidden_tips = engine.tree.tips[~observed]
    assert np.all(sparse.normalised_information[hidden_tips] < 1.0)


def test_stationary_prior_gives_constant_prior_entropy():
    engine = Engine(random_tree(20, np.random.default_rng(24)))
    q = build_q(2, "ARD", [0.2, 0.6])
    model = MkModel(["a", "b"], q, np.array([0.75, 0.25]), "ARD")  # the stationary distribution
    info = expected_information(engine, model, n_reps=50, seed=5)
    assert np.allclose(info.prior, [0.75, 0.25])
    assert np.allclose(info.prior_entropy, entropy_bits(np.array([0.75, 0.25])))
    assert np.allclose(info.baseline_accuracy, 0.75)


def test_expected_information_needs_two_replicates():
    engine = Engine(parse_newick("(A:1,B:1);"))
    with pytest.raises(ValueError):
        expected_information(engine, _model(1.0), n_reps=1)


# ------------------------------------------------------------------ isotonic
def test_isotonic_decreasing_properties():
    rng = np.random.default_rng(25)
    y = rng.normal(size=200).cumsum() * -0.1 + rng.normal(size=200)
    fit = isotonic_decreasing(y)
    assert fit.shape == y.shape
    assert np.all(np.diff(fit) <= 1e-12)
    assert np.isclose(fit.mean(), y.mean())
    already = np.array([5.0, 4.0, 4.0, 1.0])
    assert np.array_equal(isotonic_decreasing(already), already)
    assert np.allclose(isotonic_decreasing(np.array([1.0, 3.0])), [2.0, 2.0])


def test_isotonic_fit_is_the_least_squares_solution_on_a_small_case():
    # brute force over a grid is overkill; compare with the known PAVA answer
    y = np.array([4.0, 5.0, 1.0, 2.0, 0.0])
    assert np.allclose(isotonic_decreasing(y), [4.5, 4.5, 1.5, 1.5, 0.0])


def test_horizon_statuses():
    height = np.linspace(0.1, 10, 100)
    crossing = horizon(height, np.where(height < 4, 0.9, 0.1))
    assert crossing.status == "crossed" and 3.9 < crossing.height < 4.1
    assert horizon(height, np.full(100, 0.9)).status == "always_above"
    assert horizon(height, np.full(100, 0.9)).height is None
    assert horizon(height, np.full(100, 0.05)).status == "always_below"


def test_verdict_labels():
    assert [verdict(x) for x in (0.9, 0.5, 0.3, 0.1, 0.01)] == [
        "informative", "informative", "weak", "weak", "uninformative",
    ]  # fmt: skip


# ------------------------------------------------------------ rate and signal
def _simulated_data(n_tips: int, rate: float, seed: int, scale: float = 0.3):
    rng = np.random.default_rng(seed)
    engine = Engine(random_tree(n_tips, rng, scale=scale))
    model = _model(rate)
    while True:
        states = simulate_states(engine, model.q, model.root_prior, 1, rng)
        tip_like = states_to_tip_likelihoods(states, engine.tips, 2)[0]
        if 0.2 < tip_like[:, 0].mean() < 0.8:
            return engine, tip_like


def test_scale_profile_brackets_an_identifiable_rate():
    engine, tip_like = _simulated_data(200, 0.5, seed=31)
    fit = fit_mk(engine, tip_like, ["a", "b"], "ER")
    profile = scale_profile(engine, fit.as_model(), tip_like)
    assert profile.bounded_below and profile.bounded_above
    assert profile.ci_low < 1.0 < profile.ci_high
    assert profile.loglik.max() == pytest.approx(fit.loglik, abs=1e-4)
    # the interval ends sit 1.92 log-likelihood units below the maximum
    for end in (profile.ci_low, profile.ci_high):
        value = engine.loglik(fit.q * end, tip_like, fit.root_prior)[0]
        assert value == pytest.approx(fit.loglik - 1.92, abs=1e-3)


def test_scale_profile_has_no_upper_bound_for_shuffled_tips():
    rng = np.random.default_rng(32)
    engine = Engine(random_tree(80, rng, scale=1.0))
    tip_like = np.zeros((80, 2))
    tip_like[np.arange(80), rng.integers(0, 2, 80)] = 1.0
    fit = fit_mk(engine, tip_like, ["a", "b"], "ER")
    profile = scale_profile(engine, fit.as_model(), tip_like)
    assert not profile.bounded_above


def test_fast_rates_converge_on_the_saturated_likelihood():
    engine, tip_like = _simulated_data(60, 0.5, seed=33)
    counts = tip_like.sum(axis=0)
    freq = counts / counts.sum()
    q = build_q(2, "ARD", [freq[1], freq[0]]) * 1e6  # stationary distribution = freq
    assert engine.loglik(q, tip_like, "equal")[0] == pytest.approx(
        saturated_loglik(counts, "ARD"), abs=1e-6
    )
    er = build_q(2, "ER", [1e6])
    assert engine.loglik(er, tip_like, "equal")[0] == pytest.approx(
        saturated_loglik(counts, "ER"), abs=1e-6
    )


def test_signal_test_detects_structure_and_ignores_noise():
    engine, tip_like = _simulated_data(120, 0.4, seed=34)
    structured = signal_test(engine, tip_like, n_perm=99, seed=1)
    assert structured.p_value == pytest.approx(0.01)
    assert structured.statistic > structured.null.max()

    rng = np.random.default_rng(35)
    shuffled = tip_like[rng.permutation(120)]
    noise = signal_test(engine, shuffled, n_perm=99, seed=1)
    assert noise.p_value > 0.05
    assert noise.statistic >= -1e-9  # the tree can never do worse than independence


def test_signal_test_requires_variation():
    engine = Engine(random_tree(10, np.random.default_rng(36)))
    constant = np.zeros((10, 2))
    constant[:, 0] = 1.0
    with pytest.raises(ValueError):
        signal_test(engine, constant, n_perm=9)
