"""End-to-end analysis: fit, reconstruct, diagnose, tabulate."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .diagnose import (
    Horizon,
    NodeInformation,
    ScaleProfile,
    SignalTest,
    expected_information,
    horizon,
    scale_profile,
    signal_test,
    verdict,
)
from .fit import FitResult, fit_best, fit_mk, saturated_loglik
from .likelihood import Engine, entropy_bits, tip_likelihoods
from .model import MkModel
from .tree import Tree

__all__ = ["Analysis", "analyse", "node_table", "named_node_table", "summary"]


@dataclass
class Analysis:
    """Everything produced by :func:`analyse`."""

    tree: Tree
    engine: Engine
    states: list[str]
    tip_like: np.ndarray
    fits: list[FitResult]
    best: FitResult
    model: MkModel
    loglik: float
    posterior: np.ndarray  # (n, k) marginal posteriors for the observed data
    posterior_by_model: dict[str, np.ndarray]  # the same, under every fitted model
    info: NodeInformation
    info_slow: NodeInformation | None
    info_fast: NodeInformation | None
    profile: ScaleProfile
    signal: SignalTest | None
    horizon: Horizon
    height: np.ndarray  # mean distance to descendant tips
    depth: np.ndarray  # distance from the root
    seed: int

    @property
    def internal(self) -> np.ndarray:
        """Indices of real (non-virtual) internal nodes."""
        return np.flatnonzero(~self.tree.is_tip & ~self.tree.virtual)

    @property
    def saturation_index(self) -> float:
        """Mean root-to-tip distance measured in signal half-lives."""
        return float(self.depth[self.tree.tips].mean() / self.best.half_life)


def analyse(
    tree: Tree,
    tip_states: dict[str, str | None],
    states: list[str] | None = None,
    model: str = "best",
    root_prior: str = "equal",
    n_reps: int = 200,
    n_perm: int = 199,
    rate_sensitivity: bool = True,
    min_branch_length: float | None = None,
    seed: int = 1,
    chunk: int = 50,
) -> Analysis:
    """Run the full diagnostic for one discrete trait.

    Parameters
    ----------
    tree : Tree
        Rooted tree with branch lengths. Tips without a state are pruned
        (equivalent to treating them as missing, and faster).
    tip_states : dict
        Tip name -> state label. ``None`` or absent means missing.
    states : list of str, optional
        State labels, in the order to report them. Defaults to the sorted
        labels found in the data.
    model : {"best", "ER", "SYM", "ARD"}
        ``"best"`` fits every applicable model and keeps the lowest AIC.
    root_prior : {"equal", "stationary", "fitzjohn"}
    n_reps : int
        Simulated histories used for the information estimates.
    n_perm : int
        Permutations for the phylogenetic-signal test (0 skips it).
    rate_sensitivity : bool
        Also evaluate the information at both ends of the profile-likelihood
        interval of the overall rate.
    min_branch_length : float, optional
        Floor applied to every branch length. Needed when tips separated by
        zero-length branches differ in state, which no Mk model can produce.
    """
    known = {t: s for t, s in tip_states.items() if s is not None}
    present = [t for t in tree.tip_names if t in known]
    if len(present) < 4:
        raise ValueError("fewer than four tips have a state; nothing to analyse")
    missing_from_tree = len(known) - len(present)
    if missing_from_tree:
        warnings.warn(
            f"{missing_from_tree} taxa with a state are not tips of the tree", stacklevel=2
        )
    sub = tree.prune(present)
    if min_branch_length is not None:
        sub.length[1:] = np.maximum(sub.length[1:], min_branch_length)
    sub = sub.binarize()
    if states is None:
        states = sorted({known[t] for t in present})
    if len(states) < 2:
        raise ValueError("the trait has a single state among the tips")
    engine = Engine(sub)
    tip_like = tip_likelihoods(sub, known, states)

    if model == "best":
        best, fits = fit_best(engine, tip_like, states, root_prior=root_prior, seed=seed)
    else:
        best = fit_mk(engine, tip_like, states, model, root_prior, seed=seed)
        fits = [best]
    if not np.isfinite(best.loglik) or best.loglik < -1e200:
        raise ValueError(
            "the data have zero likelihood under every Mk model: tips joined by "
            "zero-length branches differ in state. Set a minimum branch length "
            "(--min-branch-length) or collapse identical tips."
        )
    mk = best.as_model()
    loglik, post = engine.reconstruct(mk.q, tip_like, mk.root_prior)
    by_model = {
        f.model: engine.reconstruct(f.q, tip_like, f.root_prior)[1][0] for f in fits if f is not best
    }
    by_model[best.model] = post[0]

    info = expected_information(engine, mk, n_reps=n_reps, seed=seed, chunk=chunk)
    profile = scale_profile(engine, mk, tip_like)
    info_slow = info_fast = None
    if rate_sensitivity:
        info_slow = expected_information(
            engine, mk.scaled(profile.ci_low), n_reps=n_reps, seed=seed + 1, chunk=chunk
        )
        info_fast = expected_information(
            engine, mk.scaled(profile.ci_high), n_reps=n_reps, seed=seed + 2, chunk=chunk
        )
    signal = signal_test(engine, tip_like, n_perm=n_perm, seed=seed) if n_perm > 0 else None

    height = sub.mean_tip_distance()
    internal = np.flatnonzero(~sub.is_tip & ~sub.virtual)
    hz = horizon(height[internal], info.normalised_information[internal])
    return Analysis(
        tree=sub,
        engine=engine,
        states=list(states),
        tip_like=tip_like,
        fits=fits,
        best=best,
        model=mk,
        loglik=float(loglik[0]),
        posterior=post[0],
        posterior_by_model=by_model,
        info=info,
        info_slow=info_slow,
        info_fast=info_fast,
        profile=profile,
        signal=signal,
        horizon=hz,
        height=height,
        depth=sub.depths(),
        seed=seed,
    )


def node_table(a: Analysis, nodes: np.ndarray | None = None) -> pd.DataFrame:
    """One row per internal node: reconstruction plus information diagnostics."""
    tree = a.tree
    nodes = a.internal if nodes is None else np.asarray(nodes)
    counts = tree.tip_counts()
    half_life = a.best.half_life
    first_tip, last_tip = [], []
    for v in nodes:
        tips = tree.descendants_tips(int(v))
        first_tip.append(tree.names[tips[0]])
        last_tip.append(tree.names[tips[-1]])
    post = a.posterior[nodes]
    data: dict[str, object] = {
        "node": nodes,
        "label": [tree.names[v] for v in nodes],
        "n_tips": counts[nodes],
        "tip_a": first_tip,
        "tip_b": last_tip,
        "height": a.height[nodes],
        "height_half_lives": a.height[nodes] / half_life,
        "depth": a.depth[nodes],
        "map_state": [a.states[j] for j in post.argmax(axis=1)],
        "map_posterior": post.max(axis=1),
    }
    for j, s in enumerate(a.states):
        data[f"post_{s}"] = post[:, j]
    for j, s in enumerate(a.states):
        data[f"prior_{s}"] = a.info.prior[nodes, j]
    ni = a.info.normalised_information[nodes]
    data.update(
        {
            "prior_entropy_bits": a.info.prior_entropy[nodes],
            "posterior_entropy_bits": entropy_bits(post),
            "information_bits": a.info.information[nodes],
            "information_se": a.info.conditional_entropy_se[nodes],
            "normalised_information": ni,
            "expected_accuracy": a.info.accuracy[nodes],
            "baseline_accuracy": a.info.baseline_accuracy[nodes],
            "skill": a.info.skill[nodes],
            "verdict": [verdict(x) for x in ni],
        }
    )
    others = {m: p for m, p in a.posterior_by_model.items() if m != a.best.model}
    if others:
        # How far the answer moves when only the model class changes.
        map_index = post.argmax(axis=1)
        rows = np.arange(nodes.size)
        disagreement = np.zeros(nodes.size)
        for m, p in others.items():
            data[f"map_posterior_{m}"] = p[nodes][rows, map_index]
            disagreement = np.maximum(disagreement, np.abs(p[nodes] - post).max(axis=1))
        data["model_disagreement"] = disagreement
    if a.info_slow is not None and a.info_fast is not None:
        slow = a.info_slow.normalised_information[nodes]
        fast = a.info_fast.normalised_information[nodes]
        data["ni_slow_rate"] = slow
        data["ni_fast_rate"] = fast
        data["verdict_robust"] = [
            verdict(x) == verdict(y) == verdict(z) for x, y, z in zip(ni, slow, fast, strict=True)
        ]
    return pd.DataFrame(data)


def named_node_table(a: Analysis, named: dict[str, list[str]]) -> pd.DataFrame:
    """Rows of :func:`node_table` for ancestors defined as MRCAs of tip sets.

    ``named`` maps a display name to tip names; the ancestor is their MRCA in
    the analysed tree. Tips that were pruned (no trait data) are ignored, and a
    definition left with fewer than two tips is skipped with a warning.
    """
    index = a.tree.tip_index()
    names, nodes, used = [], [], []
    for name, tips in named.items():
        have = [index[t] for t in tips if t in index]
        if len(have) < 2:
            warnings.warn(f"named node {name!r}: fewer than two of its tips have data", stacklevel=2)
            continue
        names.append(name)
        nodes.append(a.tree.mrca(have))
        used.append(len(have))
    table = node_table(a, np.array(nodes, dtype=np.int64))
    table.insert(0, "name", names)
    table.insert(1, "n_defining_tips", used)
    return table


def summary(a: Analysis) -> dict:
    """A JSON-serialisable summary of the analysis."""
    internal = a.internal
    ni = a.info.normalised_information[internal]
    best = a.best
    sat = saturated_loglik(best.counts, best.model)
    out: dict[str, object] = {
        "n_tips": int(a.tree.n_tips),
        "states": a.states,
        "state_counts": {s: int(c) for s, c in zip(a.states, best.counts, strict=True)},
        "model": best.model,
        "root_prior": best.root_prior_kind,
        "root_prior_vector": best.root_prior.tolist(),
        "rate_matrix": best.q.tolist(),
        "loglik": best.loglik,
        "fits": [
            {
                "model": f.model,
                "loglik": f.loglik,
                "n_params": f.n_params,
                "aic": f.aic,
                "rates": f.rates.tolist(),
                "at_upper_bound": f.at_upper_bound,
                "negligible_transitions": f.negligible_transitions,
            }
            for f in a.fits
        ],
        "negligible_transitions": best.negligible_transitions,
        "signal_half_life": best.half_life,
        "mean_root_to_tip": float(a.depth[a.tree.tips].mean()),
        "saturation_index": a.saturation_index,
        "saturated_loglik": sat,
        "loglik_gain_over_saturation": best.loglik - sat,
        "rate_scale_interval": {
            "low": a.profile.ci_low,
            "high": a.profile.ci_high,
            "bounded_below": a.profile.bounded_below,
            "bounded_above": a.profile.bounded_above,
        },
        "information": {
            "n_reps": a.info.n_reps,
            "root_normalised_information": float(a.info.normalised_information[0]),
            "root_information_bits": float(a.info.information[0]),
            "root_expected_accuracy": float(a.info.accuracy[0]),
            "root_baseline_accuracy": float(a.info.baseline_accuracy[0]),
            "median_normalised_information": float(np.median(ni)),
            "fraction_informative": float((ni >= 0.5).mean()),
            "fraction_weak": float(((ni >= 0.1) & (ni < 0.5)).mean()),
            "fraction_uninformative": float((ni < 0.1).mean()),
        },
        "horizon": {
            "threshold": a.horizon.threshold,
            "status": a.horizon.status,
            "height": a.horizon.height,
            "height_half_lives": (
                None if a.horizon.height is None else a.horizon.height / best.half_life
            ),
        },
        "seed": a.seed,
    }
    if a.signal is not None:
        out["signal_test"] = {
            "statistic": a.signal.statistic,
            "p_value": a.signal.p_value,
            "n_perm": a.signal.n_perm,
            "null_max": float(a.signal.null.max()),
        }
    return out
