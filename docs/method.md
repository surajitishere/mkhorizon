# Method notes

## Model

A discrete trait with states `1..k` evolves along the branches of a rooted
tree as a continuous-time Markov chain with rate matrix `Q` (`Q[i, j]` is the
rate from `i` to `j`; rows sum to zero). Over a branch of length `t` the
transition probabilities are `P(t) = exp(Qt)`.

| Model | Free rates | Constraint |
|---|---|---|
| ER | 1 | all rates equal |
| SYM | k(k−1)/2 | `Q[i, j] = Q[j, i]` |
| ARD | k(k−1) | none |

Root prior: `equal` (uniform), `stationary` (the stationary distribution of
`Q`), or `fitzjohn` (each root state weighted by its conditional likelihood;
FitzJohn et al. 2009). The prior that was used to fit the model is the one
used to simulate from it.

## Likelihood and marginal posteriors

*Down pass* (Felsenstein 1981). For node `v` with children `a`, `b`,

    L_v(i) = [Σ_j P_a(i, j) L_a(j)] · [Σ_j P_b(i, j) L_b(j)]

with `L` a one-hot vector at observed tips and all ones at tips with missing
data. Partials are rescaled at every node and the log scale factors summed.

*Up pass.* Let `U_v(i)` be the probability of everything outside the subtree
of `v`, jointly with `X_v = i`. `U_root = π`, and for a child `a` of `v` with
sibling `b`,

    U_a(j) = Σ_i U_v(i) · m_b(i) · P_a(i, j),   m_b(i) = Σ_j P_b(i, j) L_b(j)

The marginal posterior is `P(X_v = i | D) ∝ U_v(i) L_v(i)`.

Both passes process all nodes of equal height (longest path to a tip) in one
vectorised step and carry a leading batch dimension, so `B` data sets cost one
sweep.

## Information

For node `v`:

    H(X_v)        entropy of the prior marginal (π pushed down the tree)
    H(X_v | D)    = E_D[ H(X_v | D = d) ], estimated by averaging the entropy
                    of the exact posterior over simulated data sets d
    I(X_v; D)     = H(X_v) − H(X_v | D)
    NI            = I / H(X_v)

The estimator is unbiased because each simulated `d` is an exact draw from the
model's distribution over tip data and the posterior given `d` is exact.
Standard errors are the standard deviation of the per-replicate entropies
divided by `√R`.

Expected accuracy is `E_D[max_i P(X_v = i | D)]`, the probability that the
maximum-posterior state equals the true state. The tests confirm it equals the
fraction of simulated ancestors recovered. The baseline is `max_i P(X_v = i)`.

Tips without data in the real data set are pruned before fitting, so sparse
sampling is reflected in every number.

## Signal half-life and saturation index

The eigenvalues of `Q` are `0 > −ρ ≥ ...`. Along a single lineage the
dependence of the present state on the starting state decays as `exp(−ρt)`;
the half-life is `ln 2 / ρ`. For two states `ρ = q01 + q10`.

    saturation index = (mean root-to-tip distance) / half-life

It is a single-lineage quantity. A node with many descendants can remain
reconstructable several half-lives back because independent lineages each
carry a little information; the per-node numbers account for that, the index
does not.

## Rate interval

All rates are multiplied by a common factor `s` and the log-likelihood profiled
over `s ∈ [10⁻³, 10³]` with the ratios between rates held at their fitted
values. Factors within 1.92 log-likelihood units of the best form an
approximate 95% interval, whose ends are located by root finding. If the upper
end is the search limit the interval is reported as unbounded. The node
information is recomputed at both ends.

As rates grow the likelihood converges on that of independent draws from the
stationary distribution (`saturated_loglik`). A fit that does not beat that
limit has found no tree structure.

## Signal test

Rates `q_ij ∝ f_j`, with `f` the observed state frequencies, give a model whose
stationary distribution is `f`. The statistic is the best log-likelihood of
that model over a grid of overall rates minus the log-likelihood of independent
draws from `f`. Its null distribution comes from permuting states among the
tips with data. Because `f` is invariant under permutation, the same rate
direction applies to every permuted data set and the test is exact up to
Monte-Carlo error.

## Effectively forbidden transitions

A transition whose fitted rate times the total tree length is below 0.1 is
expected to happen less than once in ten trees of this size. When ARD does
this, deep states are fixed by the model: if gains are forbidden, every clade
that has the trait forces its ancestors to have had it.

## Model disagreement

For every fitted model class the marginal posteriors are computed and the
largest absolute difference from the best model's posterior is reported per
node. A small value means the reconstruction does not depend on the choice
between ER, SYM and ARD. It says nothing about models outside that family.
