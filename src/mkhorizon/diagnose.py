"""How much can the tips tell you about an ancestor?

Everything here answers one question about a fitted Mk model: *if this model
were exactly true, how well could the state of each internal node be recovered
from the tips?* The answer depends on the tree, the rates and which tips carry
data -- not on the observed trait values -- and is obtained by simulating the
model on the tree and reconstructing every simulated history.

Quantities (all per node ``v``; entropies in bits):

``prior``            distribution of X_v before seeing data
``H0 = H(X_v)``      entropy of that prior
``H(X_v | D)``       expected posterior entropy, averaged over simulated data
``I = H0 - H(X_v|D)`` mutual information between X_v and the tip data
``NI = I / H0``      fraction of the prior uncertainty the data can remove
``accuracy``         probability that the maximum-posterior state is correct
``baseline``         accuracy of guessing the prior's most probable state
``skill``            (accuracy - baseline) / (1 - baseline)

Because marginal posteriors are exact for a given data set, the average of
their entropies over simulated data sets is an unbiased Monte-Carlo estimate of
the conditional entropy, and its standard error is reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .likelihood import Engine, entropy_bits
from .model import MkModel
from .simulate import simulate_states, states_to_tip_likelihoods

__all__ = [
    "NodeInformation",
    "expected_information",
    "ScaleProfile",
    "scale_profile",
    "SignalTest",
    "signal_test",
    "Horizon",
    "horizon",
    "isotonic_decreasing",
    "verdict",
]


@dataclass
class NodeInformation:
    """Per-node information diagnostics (arrays indexed by node)."""

    prior: np.ndarray  # (n, k)
    prior_entropy: np.ndarray  # (n,)
    conditional_entropy: np.ndarray  # (n,)
    conditional_entropy_se: np.ndarray  # (n,)
    information: np.ndarray  # (n,) bits
    normalised_information: np.ndarray  # (n,) in [0, 1]
    accuracy: np.ndarray  # (n,) expected max posterior
    accuracy_empirical: np.ndarray  # (n,) fraction of replicates reconstructed correctly
    baseline_accuracy: np.ndarray  # (n,)
    skill: np.ndarray  # (n,)
    n_reps: int


def expected_information(
    engine: Engine,
    model: MkModel,
    observed_tips: np.ndarray | None = None,
    n_reps: int = 200,
    seed: int = 0,
    chunk: int = 50,
) -> NodeInformation:
    """Monte-Carlo estimate of the information the tips carry about every node.

    Parameters
    ----------
    engine : Engine
        Likelihood engine for the (binary) tree.
    model : MkModel
        Rates and root prior to simulate from and reconstruct under.
    observed_tips : bool array over tips, optional
        Which tips have data in the real data set. Simulated values at the
        other tips are hidden, so sparse sampling is accounted for.
    n_reps : int
        Number of simulated histories.
    """
    if n_reps < 2:
        raise ValueError("n_reps must be at least 2")
    rng = np.random.default_rng(seed)
    q, prior_root = model.q, model.root_prior
    k = model.k
    n = engine.tree.n_nodes
    p = engine.transition(q)
    prior = engine.prior_marginals(q, prior_root, p)
    h0 = entropy_bits(prior)

    sum_h = np.zeros(n)
    sum_h2 = np.zeros(n)
    sum_max = np.zeros(n)
    sum_hit = np.zeros(n)
    done = 0
    while done < n_reps:
        b = min(chunk, n_reps - done)
        states = simulate_states(engine, q, prior_root, b, rng, p)
        tip_like = states_to_tip_likelihoods(states, engine.tips, k, observed_tips)
        _, post = engine.reconstruct(q, tip_like, prior_root, p)
        h = entropy_bits(post)
        sum_h += h.sum(axis=0)
        sum_h2 += (h**2).sum(axis=0)
        sum_max += post.max(axis=2).sum(axis=0)
        sum_hit += (post.argmax(axis=2) == states).sum(axis=0)
        done += b

    cond = sum_h / n_reps
    var = np.clip(sum_h2 / n_reps - cond**2, 0.0, None) * n_reps / (n_reps - 1)
    info = h0 - cond
    with np.errstate(divide="ignore", invalid="ignore"):
        ni = np.where(h0 > 1e-12, info / h0, 0.0)
    baseline = prior.max(axis=1)
    acc = sum_max / n_reps
    with np.errstate(divide="ignore", invalid="ignore"):
        skill = np.where(baseline < 1 - 1e-12, (acc - baseline) / (1 - baseline), 0.0)
    return NodeInformation(
        prior=prior,
        prior_entropy=h0,
        conditional_entropy=cond,
        conditional_entropy_se=np.sqrt(var / n_reps),
        information=info,
        normalised_information=np.clip(ni, 0.0, 1.0),
        accuracy=acc,
        accuracy_empirical=sum_hit / n_reps,
        baseline_accuracy=baseline,
        skill=skill,
        n_reps=n_reps,
    )


# --------------------------------------------------------------------------- #
# Rate-scale profile and saturation
# --------------------------------------------------------------------------- #
@dataclass
class ScaleProfile:
    """Profile log-likelihood along a multiplier applied to all fitted rates."""

    scale: np.ndarray
    loglik: np.ndarray
    ci_low: float
    ci_high: float
    bounded_above: bool
    bounded_below: bool


def scale_profile(
    engine: Engine,
    model: MkModel,
    tip_like: np.ndarray,
    log10_range: float = 3.0,
    drop: float = 1.92,
    n_coarse: int = 25,
    n_fine: int = 61,
    window: float = 12.0,
) -> ScaleProfile:
    """Log-likelihood of the data as all rates are multiplied by a common factor.

    The factors within ``drop`` log-likelihood units of the best one form an
    approximate 95% interval for the overall rate (conditional on the fitted
    ratios between rates). If the interval runs into the upper search limit
    (``10 ** log10_range``) the data cannot exclude arbitrarily fast evolution:
    the trait may be saturated, and deep reconstructions rest on the prior.

    A coarse grid locates the region within ``window`` log-likelihood units of
    the optimum; a fine grid resolves it; the interval ends are then found by
    root finding.
    """
    from scipy.optimize import brentq

    def loglik(scale: float) -> float:
        return float(engine.loglik(model.q * scale, tip_like, model.root_prior)[0])

    if n_coarse % 2 == 0:
        n_coarse += 1  # keep scale = 1 on the grid
    coarse = np.logspace(-log10_range, log10_range, n_coarse)
    ll_coarse = np.array([loglik(s) for s in coarse])
    keep = np.flatnonzero(ll_coarse >= ll_coarse.max() - window)
    lo_i = max(int(keep[0]) - 1, 0)
    hi_i = min(int(keep[-1]) + 1, n_coarse - 1)
    scale = np.unique(
        np.append(np.logspace(np.log10(coarse[lo_i]), np.log10(coarse[hi_i]), n_fine), 1.0)
    )
    ll = np.array([loglik(s) for s in scale])
    target = ll.max() - drop
    inside = np.flatnonzero(ll >= target)
    first, last = int(inside[0]), int(inside[-1])

    def crossing(a: float, b: float) -> float:
        root = brentq(lambda u: loglik(10.0**u) - target, np.log10(a), np.log10(b), xtol=1e-6)
        return float(10.0**root)

    bounded_below = first > 0
    bounded_above = last < scale.size - 1
    ci_low = crossing(scale[first - 1], scale[first]) if bounded_below else float(scale[0])
    ci_high = crossing(scale[last], scale[last + 1]) if bounded_above else float(scale[-1])
    return ScaleProfile(
        scale=scale,
        loglik=ll,
        ci_low=ci_low,
        ci_high=ci_high,
        bounded_above=bounded_above,
        bounded_below=bounded_below,
    )


@dataclass
class SignalTest:
    """Permutation test for any phylogenetic structure in the trait."""

    statistic: float
    null: np.ndarray
    p_value: float
    n_perm: int


def signal_test(
    engine: Engine,
    tip_like: np.ndarray,
    n_perm: int = 199,
    seed: int = 0,
    n_points: int = 25,
    chunk: int = 50,
) -> SignalTest:
    """Is the trait distinguishable from states dealt to tips at random?

    The statistic is the best log-likelihood, over a grid of rate scales, of a
    model whose stationary frequencies equal the observed state frequencies
    (``q_ij`` proportional to ``pi_j``), minus the log-likelihood of independent
    draws from those frequencies. It is zero when the tree explains nothing.
    Its null distribution comes from permuting states among the tips that have
    data; the rate direction depends only on the state frequencies, which a
    permutation leaves untouched, so the test is exact up to Monte-Carlo error.
    """
    tip_like = np.asarray(tip_like, dtype=float)
    k = tip_like.shape[1]
    observed = np.flatnonzero(~np.all(tip_like == 1.0, axis=1))
    counts = tip_like[observed].sum(axis=0)
    freq = counts / counts.sum()
    if np.count_nonzero(freq) < 2:
        raise ValueError("signal_test needs at least two states among the tips")
    q = np.tile(freq, (k, 1))
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    q /= -(freq * np.diag(q)).sum()  # one expected change per unit branch length
    nz = counts[counts > 0]
    iid = float((nz * np.log(nz / counts.sum())).sum())

    lengths = engine.tree.length[1:]
    positive = lengths[lengths > 0]
    total = float(lengths.sum())
    scales = np.logspace(np.log10(0.01 / total), np.log10(100.0 / np.median(positive)), n_points)
    transitions = [engine.transition(q * s) for s in scales]

    def best_loglik(batch: np.ndarray) -> np.ndarray:
        out = np.full(batch.shape[0], -np.inf)
        for s, p in zip(scales, transitions, strict=True):
            out = np.maximum(out, engine.loglik(q * s, batch, freq, p))
        return out

    stat = float(best_loglik(tip_like[None])[0] - iid)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        batch = np.broadcast_to(tip_like, (b,) + tip_like.shape).copy()
        for row in range(b):
            batch[row, observed] = tip_like[rng.permutation(observed)]
        null[done : done + b] = best_loglik(batch) - iid
        done += b
    p_value = (1 + int((null >= stat - 1e-9).sum())) / (1 + n_perm)
    return SignalTest(statistic=stat, null=null, p_value=float(p_value), n_perm=n_perm)


# --------------------------------------------------------------------------- #
# Horizon
# --------------------------------------------------------------------------- #
def isotonic_decreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Least-squares non-increasing fit to ``y`` (pool-adjacent-violators)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    values: list[float] = []
    weights: list[float] = []
    sizes: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        values.append(float(yi))
        weights.append(float(wi))
        sizes.append(1)
        while len(values) > 1 and values[-2] < values[-1]:
            tot = weights[-2] + weights[-1]
            merged = (values[-2] * weights[-2] + values[-1] * weights[-1]) / tot
            size = sizes[-2] + sizes[-1]
            del values[-1], weights[-1], sizes[-1]
            values[-1], weights[-1], sizes[-1] = merged, tot, size
    return np.repeat(values, sizes)


@dataclass
class Horizon:
    """Where, going back from the tips, the tips stop being informative."""

    threshold: float
    height: float | None  # None when the trend never crosses the threshold
    status: str  # "crossed", "always_above" or "always_below"
    curve_height: np.ndarray
    curve_information: np.ndarray


def horizon(height: np.ndarray, normalised_information: np.ndarray, threshold: float = 0.5) -> Horizon:
    """Summarise the decay of information with node height in one number.

    A monotone (non-increasing) trend of normalised information against node
    height is fitted by isotonic regression. The horizon is the height at which
    that trend first drops below ``threshold``. Individual nodes scatter around
    the trend -- a shallow node above a long branch can be worse off than a
    deep node with many descendants -- so the per-node table, not this number,
    is what to consult for a specific ancestor.
    """
    height = np.asarray(height, dtype=float)
    ni = np.asarray(normalised_information, dtype=float)
    order = np.argsort(height, kind="stable")
    x, y = height[order], ni[order]
    fit = isotonic_decreasing(y)
    if fit[0] < threshold:
        return Horizon(threshold, 0.0, "always_below", x, fit)
    below = np.flatnonzero(fit < threshold)
    if below.size == 0:
        return Horizon(threshold, None, "always_above", x, fit)
    j = int(below[0])
    return Horizon(threshold, float(0.5 * (x[j - 1] + x[j])), "crossed", x, fit)


def verdict(normalised_information: float) -> str:
    """Plain-language label for one node's normalised information."""
    if normalised_information >= 0.5:
        return "informative"
    if normalised_information >= 0.1:
        return "weak"
    return "uninformative"
