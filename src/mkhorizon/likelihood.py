"""Vectorised likelihood and marginal ancestral-state passes for Mk models.

Two classic sweeps over the tree:

* **down** (tips -> root): Felsenstein's pruning algorithm (Felsenstein 1981)
  gives, for every node, the likelihood of the data below it conditional on
  the node's state, and hence the likelihood of the whole data set.
* **up** (root -> tips): the complementary "outside" pass gives, for every
  node, the probability of everything *not* below it jointly with the node's
  state. The normalised product of the two is the marginal posterior of the
  node's state (Yang, Kumar & Nei 1995; Koshi & Goldstein 1996).

Both sweeps are batched over an arbitrary number of data sets on the same tree
and vectorised across all nodes of equal height, so a few hundred simulated
replicates on a 10,000-tip tree take seconds rather than hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import stationary_distribution, transition_probabilities
from .tree import Tree

__all__ = ["Engine", "ROOT_PRIORS", "tip_likelihoods", "entropy_bits"]

ROOT_PRIORS = ("equal", "stationary", "fitzjohn")


def entropy_bits(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in bits along ``axis`` (0 log 0 := 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=axis)


def tip_likelihoods(tree: Tree, tip_states: dict[str, str | None], states: list[str]) -> np.ndarray:
    """One-hot tip likelihood matrix ``(n_tips, k)`` in ``tree.tips`` order.

    Tips absent from ``tip_states`` or mapped to ``None`` are treated as
    missing data (a row of ones), which is the standard likelihood treatment
    and equivalent to pruning them.
    """
    index = {s: j for j, s in enumerate(states)}
    out = np.ones((tree.n_tips, len(states)))
    for row, name in enumerate(tree.tip_names):
        value = tip_states.get(name)
        if value is None:
            continue
        if value not in index:
            raise KeyError(f"tip {name!r} has state {value!r}, not one of {states}")
        out[row] = 0.0
        out[row, index[value]] = 1.0
    return out


@dataclass
class _Pass:
    loglik: np.ndarray  # (B,)
    partial: np.ndarray  # (B, n, k) scaled conditional likelihoods
    message: np.ndarray  # (B, n, k) child -> parent messages
    root_prior: np.ndarray  # (B, k)


class Engine:
    """Likelihood engine bound to one binary tree.

    Parameters
    ----------
    tree : Tree
        Must be strictly binary; call ``tree.binarize()`` first if needed.
    """

    def __init__(self, tree: Tree) -> None:
        if not tree.is_binary():
            raise ValueError("Engine needs a binary tree; call tree.binarize() first")
        self.tree = tree
        n = tree.n_nodes
        self.tips = tree.tips
        self.internal = np.flatnonzero(~tree.is_tip)
        self.c1 = np.full(n, -1, dtype=np.int64)
        self.c2 = np.full(n, -1, dtype=np.int64)
        for i in self.internal:
            self.c1[i], self.c2[i] = tree.children[i]
        heights = tree.edge_heights()
        order = self.internal[np.argsort(heights[self.internal], kind="stable")]
        cuts = np.flatnonzero(np.diff(heights[order])) + 1
        #: internal nodes grouped by height, lowest first; children of a group
        #: always live in earlier groups (or are tips).
        self.groups: list[np.ndarray] = np.split(order, cuts)

    # ---------------------------------------------------------------- helpers
    def transition(self, q: np.ndarray) -> np.ndarray:
        """Per-node transition matrices ``(n, k, k)`` for the branch above each node."""
        return transition_probabilities(q, self.tree.length)

    def _as_batch(self, tip_like: np.ndarray) -> np.ndarray:
        tip_like = np.asarray(tip_like, dtype=float)
        if tip_like.ndim == 2:
            tip_like = tip_like[None]
        if tip_like.ndim != 3 or tip_like.shape[1] != self.tips.size:
            raise ValueError("tip likelihoods must have shape (n_tips, k) or (B, n_tips, k)")
        return tip_like

    def _resolve_prior(self, root_prior, q: np.ndarray, root_partial: np.ndarray) -> np.ndarray:
        """Turn a root-prior specification into a ``(B, k)`` array."""
        batch, k = root_partial.shape
        if isinstance(root_prior, str):
            if root_prior == "equal":
                pi = np.full(k, 1.0 / k)
            elif root_prior == "stationary":
                pi = stationary_distribution(q)
            elif root_prior == "fitzjohn":
                # FitzJohn, Maddison & Otto (2009): weight each root state by
                # its own conditional likelihood.
                return root_partial / root_partial.sum(axis=1, keepdims=True)
            else:
                raise ValueError(f"root_prior must be one of {ROOT_PRIORS} or an array")
        else:
            pi = np.asarray(root_prior, dtype=float)
            if pi.shape != (k,):
                raise ValueError("root prior array must have one entry per state")
        return np.broadcast_to(pi, (batch, k))

    # ------------------------------------------------------------- down pass
    def _down(self, q: np.ndarray, tip_like: np.ndarray, root_prior, p=None) -> _Pass:
        tip_like = self._as_batch(tip_like)
        batch, _, k = tip_like.shape
        n = self.tree.n_nodes
        if p is None:
            p = self.transition(q)
        partial = np.empty((batch, n, k))
        partial[:, self.tips] = tip_like
        message = np.empty((batch, n, k))
        logscale = np.zeros(batch)
        for g in self.groups:
            a, b = self.c1[g], self.c2[g]
            m1 = np.einsum("nij,bnj->bni", p[a], partial[:, a])
            m2 = np.einsum("nij,bnj->bni", p[b], partial[:, b])
            message[:, a] = m1
            message[:, b] = m2
            prod = m1 * m2
            s = prod.sum(axis=2)
            ok = s > 0
            with np.errstate(divide="ignore"):
                logscale += np.where(ok, np.log(np.where(ok, s, 1.0)), -np.inf).sum(axis=1)
            partial[:, g] = np.where(ok[..., None], prod / np.where(ok, s, 1.0)[..., None], 1.0 / k)
        prior = self._resolve_prior(root_prior, q, partial[:, 0])
        with np.errstate(divide="ignore"):
            loglik = logscale + np.log((prior * partial[:, 0]).sum(axis=1))
        return _Pass(loglik, partial, message, prior)

    def loglik(self, q: np.ndarray, tip_like: np.ndarray, root_prior="equal", p=None) -> np.ndarray:
        """Log-likelihood of each data set in the batch, shape ``(B,)``."""
        return self._down(q, tip_like, root_prior, p).loglik

    # --------------------------------------------------------------- up pass
    def reconstruct(self, q: np.ndarray, tip_like: np.ndarray, root_prior="equal", p=None):
        """Log-likelihoods and marginal posteriors for every node.

        Returns
        -------
        loglik : (B,) array
        posterior : (B, n_nodes, k) array
            Marginal posterior of each node's state. Rows for observed tips are
            their observed state; rows for missing tips are predictions.
        """
        if p is None:
            p = self.transition(q)
        down = self._down(q, tip_like, root_prior, p)
        partial, message = down.partial, down.message
        outside = np.empty_like(partial)
        outside[:, 0] = down.root_prior
        for g in reversed(self.groups):
            a, b = self.c1[g], self.c2[g]
            base = outside[:, g]
            u1 = np.einsum("bni,nij->bnj", base * message[:, b], p[a])
            u2 = np.einsum("bni,nij->bnj", base * message[:, a], p[b])
            outside[:, a] = _normalise(u1)
            outside[:, b] = _normalise(u2)
        posterior = _normalise(outside * partial)
        return down.loglik, posterior

    # ------------------------------------------------------- prior marginals
    def prior_marginals(self, q: np.ndarray, root_prior: np.ndarray, p=None) -> np.ndarray:
        """Distribution of each node's state before seeing any data, ``(n, k)``.

        The root prior is pushed down the tree through the transition
        matrices. With a stationary root prior every row equals it.
        """
        if p is None:
            p = self.transition(q)
        root_prior = np.asarray(root_prior, dtype=float)
        out = np.empty((self.tree.n_nodes, root_prior.size))
        out[0] = root_prior
        for g in reversed(self.groups):
            for child in (self.c1[g], self.c2[g]):
                out[child] = np.einsum("ni,nij->nj", out[g], p[child])
        return out

    # ---------------------------------------------------------------- Fitch
    def parsimony_changes(self, tip_like: np.ndarray) -> int:
        """Fitch parsimony score (minimum number of state changes).

        Used only to pick sensible starting values for the rate optimiser.
        """
        tip_like = np.asarray(tip_like)
        k = tip_like.shape[-1]
        weights = 1 << np.arange(k)
        sets = np.zeros(self.tree.n_nodes, dtype=np.int64)
        sets[self.tips] = (tip_like > 0).astype(np.int64) @ weights
        changes = 0
        for g in self.groups:
            a, b = sets[self.c1[g]], sets[self.c2[g]]
            inter = a & b
            miss = inter == 0
            changes += int(miss.sum())
            sets[g] = np.where(miss, a | b, inter)
        return changes


def _normalise(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=-1, keepdims=True)
    k = x.shape[-1]
    return np.where(s > 0, x / np.where(s > 0, s, 1.0), 1.0 / k)
