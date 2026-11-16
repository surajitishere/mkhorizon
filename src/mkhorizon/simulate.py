"""Forward simulation of discrete traits on a tree, batched over replicates."""

from __future__ import annotations

import numpy as np

from .likelihood import Engine

__all__ = ["simulate_states", "states_to_tip_likelihoods"]


def simulate_states(
    engine: Engine,
    q: np.ndarray,
    root_prior: np.ndarray,
    n_reps: int,
    rng: np.random.Generator,
    p: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate the state of **every** node for ``n_reps`` independent histories.

    The root state is drawn from ``root_prior``; each child state is then drawn
    from the row of its branch's transition matrix selected by the parent's
    state. Returns an integer array of shape ``(n_reps, n_nodes)`` so that the
    true ancestral states are available for scoring reconstructions.
    """
    if p is None:
        p = engine.transition(q)
    root_prior = np.asarray(root_prior, dtype=float)
    k = root_prior.size
    n = engine.tree.n_nodes
    states = np.empty((n_reps, n), dtype=np.int16)
    states[:, 0] = rng.choice(k, size=n_reps, p=root_prior)
    for g in reversed(engine.groups):
        parent_state = states[:, g]  # (B, |g|)
        for child in (engine.c1[g], engine.c2[g]):
            # rows[b, j, :] = P_child[j][parent_state[b, j], :]
            rows = p[child][np.arange(child.size)[None, :], parent_state]
            cdf = np.cumsum(rows[..., :-1], axis=-1)
            u = rng.random(parent_state.shape)
            states[:, child] = (u[..., None] >= cdf).sum(axis=-1)
    return states


def states_to_tip_likelihoods(
    states: np.ndarray, tips: np.ndarray, k: int, observed: np.ndarray | None = None
) -> np.ndarray:
    """One-hot tip likelihoods ``(B, n_tips, k)`` from simulated node states.

    ``observed`` is a boolean mask over tips; unobserved tips become rows of
    ones, mirroring missing data in the real data set.
    """
    tip_states = states[:, tips]
    out = np.zeros(tip_states.shape + (k,))
    np.put_along_axis(out, tip_states[..., None].astype(np.int64), 1.0, axis=-1)
    if observed is not None:
        out[:, ~np.asarray(observed, dtype=bool)] = 1.0
    return out
