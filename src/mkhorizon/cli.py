"""Command-line interface: ``mkhorizon run`` and ``mkhorizon fit``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .io import read_named_nodes, read_trait
from .likelihood import ROOT_PRIORS
from .report import analyse, named_node_table, node_table, summary
from .tree import read_newick


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tree", required=True, help="rooted Newick tree with branch lengths")
    p.add_argument("--traits", required=True, help="trait table (TSV or CSV with a header row)")
    p.add_argument("--trait", required=True, help="column holding the discrete trait")
    p.add_argument("--id-col", help="column holding tip names (default: first column)")
    p.add_argument("--states", help="comma-separated state order (default: sorted labels)")
    p.add_argument(
        "--model",
        default="best",
        choices=["best", "ER", "SYM", "ARD"],
        help="Mk parameterisation; 'best' picks the lowest AIC (default)",
    )
    p.add_argument("--root-prior", default="equal", choices=ROOT_PRIORS)
    p.add_argument(
        "--min-branch-length",
        type=float,
        help="floor for branch lengths (use when identical tips differ in state)",
    )
    p.add_argument("--seed", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mkhorizon",
        description="How far back can a discrete trait be reconstructed on this tree?",
    )
    parser.add_argument("--version", action="version", version=f"mkhorizon {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="fit, reconstruct and compute per-node information diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _common(run)
    run.add_argument("--nodes", help="named ancestors: lines of 'name<TAB>tip,tip,...'")
    run.add_argument("--reps", type=int, default=200, help="simulated histories")
    run.add_argument("--perm", type=int, default=199, help="permutations for the signal test (0 = skip)")
    run.add_argument(
        "--no-rate-sensitivity",
        action="store_true",
        help="skip re-evaluating the information at the ends of the rate interval",
    )
    run.add_argument("--no-plot", action="store_true", help="do not write report.png")
    run.add_argument("--out", required=True, help="output directory")

    fit = sub.add_parser(
        "fit",
        help="fit Mk models only and print the comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _common(fit)
    return parser


def _load(args):
    tree = read_newick(args.tree)
    trait = read_trait(args.traits, args.trait, args.id_col)
    states = [s.strip() for s in args.states.split(",")] if args.states else None
    return tree, trait, states


def _print_fits(a, stream) -> None:
    print(f"{'model':6s} {'lnL':>12s} {'AIC':>12s}  rates", file=stream)
    for f in a.fits:
        flag = "  (at upper bound)" if f.at_upper_bound else ""
        if f.negligible_transitions:
            flag += "  (effectively forbids: " + ", ".join(f.negligible_transitions) + ")"
        mark = "*" if f is a.best else " "
        rates = " ".join(f"{r:.4g}" for r in f.rates)
        print(f"{mark}{f.model:5s} {f.loglik:12.3f} {f.aic:12.3f}  {rates}{flag}", file=stream)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, KeyError, FileNotFoundError) as err:
        message = err.args[0] if err.args else str(err)
        print(f"mkhorizon: error: {message}", file=sys.stderr)
        return 2


def _run(args) -> int:
    tree, trait, states = _load(args)

    if args.command == "fit":
        a = analyse(
            tree, trait, states, args.model, args.root_prior,
            n_reps=2, n_perm=0, rate_sensitivity=False,
            min_branch_length=args.min_branch_length, seed=args.seed,
        )  # fmt: skip
        _print_fits(a, sys.stdout)
        return 0

    a = analyse(
        tree,
        trait,
        states,
        model=args.model,
        root_prior=args.root_prior,
        n_reps=args.reps,
        n_perm=args.perm,
        rate_sensitivity=not args.no_rate_sensitivity,
        min_branch_length=args.min_branch_length,
        seed=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    node_table(a).to_csv(out / "nodes.tsv", sep="\t", index=False, float_format="%.6g")
    info = summary(a)
    named_nodes = None
    if args.nodes:
        named = named_node_table(a, read_named_nodes(args.nodes))
        named.to_csv(out / "named_nodes.tsv", sep="\t", index=False, float_format="%.6g")
        named_nodes = dict(zip(named["name"], named["node"].astype(int), strict=True))
        info["named_nodes"] = json.loads(named.to_json(orient="records"))
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    if not args.no_plot:
        from .plot import plot_report

        plot_report(a, out / "report.png", named=named_nodes, title=f"{args.trait}")

    s = info
    print(f"trait {args.trait!r}: {s['n_tips']} tips, states {s['state_counts']}")
    _print_fits(a, sys.stdout)
    print(
        f"signal half-life {s['signal_half_life']:.4g}; mean root-to-tip distance "
        f"{s['mean_root_to_tip']:.4g} = {s['saturation_index']:.2f} half-lives"
    )
    iv = s["rate_scale_interval"]
    upper = f"{iv['high']:.3g}" if iv["bounded_above"] else "unbounded"
    print(f"overall-rate interval: x{iv['low']:.3g} to x{upper}")
    if "signal_test" in s:
        print(
            f"phylogenetic signal: statistic {s['signal_test']['statistic']:.1f}, "
            f"p = {s['signal_test']['p_value']:.3g}"
        )
    inf = s["information"]
    print(
        f"internal nodes: {inf['fraction_informative']:.1%} informative, "
        f"{inf['fraction_weak']:.1%} weak, {inf['fraction_uninformative']:.1%} uninformative"
    )
    print(
        f"root: normalised information {inf['root_normalised_information']:.3f}, expected accuracy "
        f"{inf['root_expected_accuracy']:.3f} (baseline {inf['root_baseline_accuracy']:.3f})"
    )
    if s["negligible_transitions"]:
        print(
            "WARNING: the fitted model effectively forbids "
            + ", ".join(s["negligible_transitions"])
            + ". Deep states are then dictated by the model class, not by the tips; "
            "compare the per-model columns in nodes.tsv before trusting them."
        )
    if not iv["bounded_above"]:
        print(
            "WARNING: the overall rate has no upper bound; the trait may be saturated "
            "and deep reconstructions rest on the root prior."
        )
    print(f"wrote {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
