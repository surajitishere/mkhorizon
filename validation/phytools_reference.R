#!/usr/bin/env Rscript
# Reference values for mkhorizon's likelihood, generated with phytools.
#
#   Rscript validation/phytools_reference.R tests/data/phytools_reference.json
#
# For a set of random trees and simulated traits this script records, for
# several models and root priors:
#   * the rate matrix phytools::fitMk converged to and its log-likelihood,
#   * the marginal ancestral states from phytools::ancr at every internal node.
# tests/test_phytools.py then (a) evaluates mkhorizon's likelihood at the same
# rate matrix, (b) compares marginal posteriors node by node, and (c) checks
# that mkhorizon's own optimiser reaches a likelihood at least as high.
#
# Nodes are identified by their sorted descendant tip labels, so nothing
# depends on how either package numbers nodes.
#
# Versions used for the committed file are written into the JSON.

suppressPackageStartupMessages({
  library(ape)
  library(phytools)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[1] else "phytools_reference.json"
set.seed(20261110)

clade_key <- function(tree, node) {
  tips <- extract.clade(tree, node)$tip.label
  paste(sort(tips), collapse = "|")
}

one_case <- function(name, n_tips, states, q_true, scale = 1) {
  tree <- rtree(n_tips)
  tree$edge.length <- tree$edge.length * scale
  k <- length(states)
  dimnames(q_true) <- list(states, states)
  x <- sim.Mk(tree, q_true)
  x <- setNames(as.character(x), names(x))
  # keep going until every state is present, so all models are estimable
  while (length(unique(x)) < k) {
    x <- sim.Mk(tree, q_true)
    x <- setNames(as.character(x), names(x))
  }
  x <- factor(x, levels = states)
  models <- if (k == 2) c("ER", "ARD") else c("ER", "SYM", "ARD")
  priors <- c("equal", "fitzjohn")
  fits <- list()
  for (m in models) {
    for (p in priors) {
      fit <- fitMk(tree, x, model = m, pi = p)
      q <- as.Qmatrix(fit)
      q <- unclass(q)[states, states]
      anc <- ancr(fit, tips = FALSE)
      ace <- anc$ace[, states, drop = FALSE]
      nodes <- as.integer(rownames(ace))
      keys <- vapply(nodes, function(nd) clade_key(tree, nd), character(1))
      fits[[length(fits) + 1]] <- list(
        model = m,
        root_prior = p,
        loglik = as.numeric(logLik(fit)),
        q = q,
        root_pi = as.numeric(fit$pi[states]),
        posterior = setNames(lapply(seq_along(nodes), function(i) as.numeric(ace[i, ])), keys)
      )
    }
  }
  list(
    name = name,
    newick = write.tree(tree),
    states = states,
    tip_states = as.list(setNames(as.character(x), names(x))),
    fits = fits
  )
}

q2 <- matrix(c(-0.4, 0.4, 1.1, -1.1), 2, 2, byrow = TRUE)
q2_fast <- q2 * 6
q3 <- matrix(c(-0.9, 0.6, 0.3,
               0.2, -0.5, 0.3,
               0.8, 0.1, -0.9), 3, 3, byrow = TRUE)

cases <- list(
  one_case("binary_60", 60, c("a", "b"), q2),
  one_case("binary_150_fast", 150, c("a", "b"), q2_fast),
  one_case("three_state_80", 80, c("x", "y", "z"), q3),
  one_case("binary_25_short", 25, c("a", "b"), q2, scale = 0.3)
)

out <- list(
  generated_with = list(
    R = R.version.string,
    ape = as.character(packageVersion("ape")),
    phytools = as.character(packageVersion("phytools"))
  ),
  cases = cases
)
write_json(out, out_path, digits = NA, auto_unbox = TRUE, pretty = FALSE)
cat("wrote", out_path, "\n")
