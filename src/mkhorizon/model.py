"""The Mk model: rate matrices, transition probabilities, stationary behaviour.

Mk (Lewis 2001; Pagel 1994) is the continuous-time Markov chain behind most
discrete ancestral-state reconstructions -- ``ape::ace``, ``phytools::fitMk`` and
``make.simmap``, ``castor::asr_mk_model``. Three parameterisations are offered:

``ER``   one rate shared by every transition
``SYM``  one rate per unordered pair of states
``ARD``  one rate per ordered pair of states
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

__all__ = [
    "MODELS",
    "MkModel",
    "n_rate_params",
    "build_q",
    "stationary_distribution",
    "transition_probabilities",
    "signal_half_life",
]

MODELS = ("ER", "SYM", "ARD")


def n_rate_params(k: int, model: str) -> int:
    """Number of free rates for ``k`` states under ``model``."""
    if model == "ER":
        return 1
    if model == "SYM":
        return k * (k - 1) // 2
    if model == "ARD":
        return k * (k - 1)
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


def build_q(k: int, model: str, rates) -> np.ndarray:
    """Assemble a ``k x k`` rate matrix from a flat vector of rates.

    Ordering of ``rates``: ER takes one value; SYM fills the upper triangle
    row by row; ARD fills the off-diagonal row by row (``q01, q02, ..., q10,
    q12, ...``). ``Q[i, j]`` is the rate of change *from* state i *to* state j.
    """
    rates = np.atleast_1d(np.asarray(rates, dtype=float))
    if rates.size != n_rate_params(k, model):
        raise ValueError(f"{model} with k={k} needs {n_rate_params(k, model)} rates")
    if np.any(rates < 0) or not np.all(np.isfinite(rates)):
        raise ValueError("rates must be finite and non-negative")
    q = np.zeros((k, k))
    if model == "ER":
        q[:] = rates[0]
    elif model == "SYM":
        iu = np.triu_indices(k, 1)
        q[iu] = rates
        q = q + q.T
    else:
        mask = ~np.eye(k, dtype=bool)
        q[mask] = rates
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


def stationary_distribution(q: np.ndarray) -> np.ndarray:
    """Stationary distribution ``pi`` with ``pi @ Q = 0``.

    Solved as a least-squares system with the normalisation constraint
    appended, which is stable for the small, irreducible matrices used here.
    """
    k = q.shape[0]
    a = np.vstack([q.T, np.ones((1, k))])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def transition_probabilities(q: np.ndarray, t) -> np.ndarray:
    """``P(t) = expm(Q t)`` for an array of branch lengths, shape ``(len(t), k, k)``.

    Uses one eigendecomposition of Q for all branches. ARD matrices with three
    or more states can have complex eigenvalues, so the arithmetic is done in
    complex numbers and the (exactly real) result is taken at the end. If Q is
    numerically defective the function falls back to ``scipy.linalg.expm``.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    k = q.shape[0]
    if np.any(t < 0):
        raise ValueError("branch lengths must be non-negative")
    if not np.any(q):
        return np.broadcast_to(np.eye(k), (t.size, k, k)).copy()
    w, v = np.linalg.eig(q)
    if np.linalg.cond(v) < 1e10:
        vinv = np.linalg.inv(v)
        p = np.einsum("ij,tj,jk->tik", v, np.exp(np.multiply.outer(t, w)), vinv).real
    else:  # pragma: no cover - defective matrices are a measure-zero corner
        uniq, inverse = np.unique(t, return_inverse=True)
        p = np.stack([expm(q * x) for x in uniq])[inverse]
    p = np.clip(p, 0.0, None)
    p /= p.sum(axis=2, keepdims=True)
    p[t == 0.0] = np.eye(k)  # exactly: no time, no change
    return p


def signal_half_life(q: np.ndarray) -> float:
    """Branch length over which the memory of a starting state halves.

    Along a single lineage the dependence of the present state on the past one
    decays as ``exp(-rho * t)`` where ``rho`` is the spectral gap of Q (the
    smallest non-zero ``|Re(eigenvalue)|``). The half-life is ``ln 2 / rho``.
    For two states ``rho = q01 + q10``.
    """
    w = np.linalg.eigvals(q)
    mags = np.sort(np.abs(w.real))
    scale = max(mags[-1], 1e-300)
    nonzero = mags[mags > 1e-9 * scale]
    if nonzero.size == 0:
        return float("inf")
    return float(np.log(2.0) / nonzero[0])


@dataclass
class MkModel:
    """A fully specified Mk model: states, rate matrix and root prior."""

    states: list[str]
    q: np.ndarray
    root_prior: np.ndarray
    model: str = "custom"

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)
        self.root_prior = np.asarray(self.root_prior, dtype=float)
        k = len(self.states)
        if self.q.shape != (k, k) or self.root_prior.shape != (k,):
            raise ValueError("Q and root prior must match the number of states")
        if not np.allclose(self.q.sum(axis=1), 0.0, atol=1e-9 * (1 + np.abs(self.q).max())):
            raise ValueError("rows of Q must sum to zero")
        if not np.isclose(self.root_prior.sum(), 1.0) or np.any(self.root_prior < 0):
            raise ValueError("root prior must be a probability vector")

    @property
    def k(self) -> int:
        return len(self.states)

    @property
    def stationary(self) -> np.ndarray:
        return stationary_distribution(self.q)

    @property
    def half_life(self) -> float:
        return signal_half_life(self.q)

    def scaled(self, factor: float) -> MkModel:
        """The same model with every rate multiplied by ``factor``."""
        return MkModel(list(self.states), self.q * factor, self.root_prior.copy(), self.model)

    def transition_probabilities(self, t) -> np.ndarray:
        return transition_probabilities(self.q, t)
