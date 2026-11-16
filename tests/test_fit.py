import numpy as np
import pytest

from helpers import random_tree
from mkhorizon.fit import fit_best, fit_mk, saturated_loglik, transition_labels
from mkhorizon.likelihood import Engine
from mkhorizon.model import build_q
from mkhorizon.simulate import simulate_states, states_to_tip_likelihoods


def _simulate(engine, q, prior, seed):
    rng = np.random.default_rng(seed)
    states = simulate_states(engine, q, np.asarray(prior, float), 1, rng)
    return states_to_tip_likelihoods(states, engine.tips, q.shape[0])[0]


def test_recovers_simulated_rates_on_a_large_tree():
    engine = Engine(random_tree(1500, np.random.default_rng(41)))
    truth = np.array([0.4, 1.2])
    tip_like = _simulate(engine, build_q(2, "ARD", truth), [0.5, 0.5], seed=42)
    fit = fit_mk(engine, tip_like, ["a", "b"], "ARD")
    assert np.all(np.abs(np.log(fit.rates / truth)) < np.log(1.5))
    assert fit.n_params == 2 and fit.aic == pytest.approx(4 - 2 * fit.loglik)
    assert not fit.at_upper_bound and fit.negligible_transitions == []
    assert fit.loglik > saturated_loglik(fit.counts, "ARD") + 50


def test_nested_models_are_ordered_by_likelihood():
    engine = Engine(random_tree(150, np.random.default_rng(43)))
    q = build_q(3, "ARD", [0.9, 0.1, 0.2, 0.6, 0.8, 0.1])
    tip_like = _simulate(engine, q, [1 / 3] * 3, seed=44)
    best, fits = fit_best(engine, tip_like, ["x", "y", "z"])
    by_model = {f.model: f for f in fits}
    assert set(by_model) == {"ER", "SYM", "ARD"}
    assert by_model["ER"].loglik <= by_model["SYM"].loglik + 1e-4
    assert by_model["SYM"].loglik <= by_model["ARD"].loglik + 1e-4
    assert best.aic == min(f.aic for f in fits)


def test_irreversible_process_is_flagged():
    engine = Engine(random_tree(300, np.random.default_rng(45)))
    # the trait starts present and can only be lost
    q = build_q(2, "ARD", [0.0, 0.3])
    tip_like = _simulate(engine, q, [0.0, 1.0], seed=47)
    assert 0.1 < tip_like[:, 1].mean() < 0.9
    fit = fit_mk(engine, tip_like, ["absent", "present"], "ARD")
    assert fit.negligible_transitions == ["absent->present"]


def test_saturated_data_run_into_the_upper_bound():
    rng = np.random.default_rng(47)
    engine = Engine(random_tree(100, rng, scale=2.0))
    tip_like = np.zeros((100, 2))
    tip_like[np.arange(100), rng.integers(0, 2, 100)] = 1.0
    fit = fit_mk(engine, tip_like, ["a", "b"], "ER")
    assert fit.loglik == pytest.approx(saturated_loglik(fit.counts, "ER"), abs=0.5)
    assert fit.half_life < np.median(engine.tree.length[1:])


def test_root_prior_options_and_serialisation():
    engine = Engine(random_tree(80, np.random.default_rng(48)))
    tip_like = _simulate(engine, build_q(2, "ARD", [0.3, 0.9]), [0.5, 0.5], seed=49)
    fits = {p: fit_mk(engine, tip_like, ["a", "b"], "ARD", p) for p in ("equal", "stationary", "fitzjohn")}
    assert np.allclose(fits["equal"].root_prior, 0.5)
    q = fits["stationary"].q
    assert np.allclose(fits["stationary"].root_prior @ q, 0.0, atol=1e-10)
    assert np.isclose(fits["fitzjohn"].root_prior.sum(), 1.0)
    assert fits["fitzjohn"].loglik >= fits["equal"].loglik - 1e-6
    as_dict = fits["equal"].to_dict()
    assert isinstance(as_dict["q"], list) and as_dict["model"] == "ARD"
    with pytest.raises(ValueError):
        fit_mk(engine, tip_like, ["a", "b"], "ARD", "nope")


def test_transition_labels():
    assert transition_labels(["a", "b"], "ER") == ["all transitions"]
    assert transition_labels(["a", "b"], "ARD") == ["a->b", "b->a"]
    assert transition_labels(["a", "b", "c"], "SYM") == ["a<->b", "a<->c", "b<->c"]
