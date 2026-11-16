"""Maximum-likelihood fitting of Mk models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from .likelihood import ROOT_PRIORS, Engine
from .model import MkModel, build_q, n_rate_params, signal_half_life, stationary_distribution

__all__ = ["FitResult", "fit_mk", "fit_best", "saturated_loglik", "transition_labels"]


def transition_labels(states: list[str], model: str) -> list[str]:
    """Human-readable label for each free rate, in the order used by ``build_q``."""
    k = len(states)
    if model == "ER":
        return ["all transitions"]
    if model == "SYM":
        return [f"{states[i]}<->{states[j]}" for i in range(k) for j in range(i + 1, k)]
    return [f"{states[i]}->{states[j]}" for i in range(k) for j in range(k) if i != j]


@dataclass
class FitResult:
    """Outcome of one Mk fit."""

    model: str
    states: list[str]
    rates: np.ndarray
    q: np.ndarray
    loglik: float
    n_params: int
    aic: float
    root_prior_kind: str
    root_prior: np.ndarray
    counts: np.ndarray
    at_upper_bound: bool
    half_life: float
    negligible_transitions: list[str]

    def as_model(self) -> MkModel:
        """The fitted model, with a concrete root-prior vector."""
        return MkModel(list(self.states), self.q.copy(), self.root_prior.copy(), self.model)

    def to_dict(self) -> dict:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
        return out


def saturated_loglik(counts, model: str) -> float:
    """Log-likelihood in the limit of infinitely fast evolution.

    As rates grow without bound every tip becomes an independent draw from the
    stationary distribution, whatever the tree. Under ARD that distribution is
    free, so the limit is the multinomial likelihood at the observed
    frequencies; under ER and SYM it is uniform. An Mk fit that barely beats
    this number has found no phylogenetic structure in the trait.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if model == "ARD":
        nz = counts[counts > 0]
        return float((nz * np.log(nz / total)).sum())
    return float(total * np.log(1.0 / counts.size))


def _rate_bounds(engine: Engine) -> tuple[float, float]:
    """Loose bounds that still bracket every identifiable rate.

    Lower: about 1e-4 expected changes over the whole tree. Upper: a signal
    half-life a thousand times shorter than the median branch, beyond which the
    likelihood is flat (the saturation plateau).
    """
    lengths = engine.tree.length[1:]
    total = float(lengths.sum())
    positive = lengths[lengths > 0]
    if positive.size == 0 or total <= 0:
        raise ValueError("tree has no positive branch lengths")
    return 1e-4 / total, 1e3 / float(np.median(positive))


def fit_mk(
    engine: Engine,
    tip_like: np.ndarray,
    states: list[str],
    model: str = "ARD",
    root_prior: str = "equal",
    n_starts: int = 4,
    seed: int = 0,
) -> FitResult:
    """Fit one Mk model by maximum likelihood.

    Rates are optimised on the log scale with L-BFGS-B from several starting
    points (a parsimony-based guess plus perturbations of it).
    """
    if root_prior not in ROOT_PRIORS:
        raise ValueError(f"root_prior must be one of {ROOT_PRIORS}")
    tip_like = np.asarray(tip_like, dtype=float)
    k = len(states)
    if tip_like.shape != (engine.tips.size, k):
        raise ValueError("tip_like must have shape (n_tips, k)")
    observed = ~np.all(tip_like == 1.0, axis=1)
    counts = tip_like[observed].sum(axis=0)
    n_par = n_rate_params(k, model)
    lo, hi = _rate_bounds(engine)

    def negloglik(theta: np.ndarray) -> float:
        q = build_q(k, model, np.exp(theta))
        value = float(engine.loglik(q, tip_like, root_prior)[0])
        return -value if np.isfinite(value) else 1e300

    total = float(engine.tree.length.sum())
    changes = max(engine.parsimony_changes(tip_like), 1)
    guess = np.clip(changes / total / max(k - 1, 1), lo * 10, hi / 10)
    rng = np.random.default_rng(seed)
    starts = [np.full(n_par, np.log(guess))]
    for factor in (10.0, 0.1):
        if len(starts) < n_starts:
            starts.append(np.full(n_par, np.log(np.clip(guess * factor, lo * 10, hi / 10))))
    while len(starts) < n_starts:
        starts.append(np.log(guess) + rng.normal(0.0, 1.5, n_par))
    bounds = [(np.log(lo), np.log(hi))] * n_par

    best = None
    for start in starts:
        start = np.clip(start, np.log(lo), np.log(hi))
        res = minimize(negloglik, start, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    if n_par <= 3:  # cheap derivative-free polish; guards against a flat gradient
        res = minimize(
            negloglik,
            best.x,
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 400 * n_par},
        )
        if res.fun < best.fun:
            best = res

    rates = np.exp(best.x)
    q = build_q(k, model, rates)
    loglik = -float(best.fun)
    if root_prior == "equal":
        prior = np.full(k, 1.0 / k)
    elif root_prior == "stationary":
        prior = stationary_distribution(q)
    else:
        prior = engine._down(q, tip_like, "fitzjohn").root_prior[0].copy()
    return FitResult(
        model=model,
        states=list(states),
        rates=rates,
        q=q,
        loglik=loglik,
        n_params=n_par,
        aic=2 * n_par - 2 * loglik,
        root_prior_kind=root_prior,
        root_prior=prior,
        counts=counts,
        at_upper_bound=bool(np.any(best.x > np.log(hi) - 1e-3)),
        half_life=signal_half_life(q),
        # A transition expected fewer than 0.1 times over the whole tree is one
        # the fitted model effectively forbids.
        negligible_transitions=[
            label
            for label, rate in zip(transition_labels(list(states), model), rates, strict=True)
            if rate * total < 0.1
        ],
    )


def fit_best(
    engine: Engine,
    tip_like: np.ndarray,
    states: list[str],
    models: tuple[str, ...] | None = None,
    root_prior: str = "equal",
    n_starts: int = 4,
    seed: int = 0,
) -> tuple[FitResult, list[FitResult]]:
    """Fit several models and return the one with the lowest AIC, plus all fits."""
    if models is None:
        models = ("ER", "ARD") if len(states) == 2 else ("ER", "SYM", "ARD")
    fits = [fit_mk(engine, tip_like, states, m, root_prior, n_starts, seed) for m in models]
    best = min(fits, key=lambda f: f.aic)
    return best, fits
