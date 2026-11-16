import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mkhorizon import read_newick
from mkhorizon.cli import main
from mkhorizon.io import read_named_nodes, read_trait
from mkhorizon.report import analyse, named_node_table, node_table, summary

TOY = Path(__file__).parents[1] / "examples" / "toy"


@pytest.fixture(scope="module")
def toy():
    tree = read_newick(TOY / "tree.nwk")
    named = read_named_nodes(TOY / "nodes.tsv")
    return tree, named


@pytest.fixture(scope="module")
def slow(toy):
    tree, _ = toy
    return analyse(tree, read_trait(TOY / "traits.tsv", "slow"), n_reps=300, n_perm=39, seed=1)


@pytest.fixture(scope="module")
def fast(toy):
    tree, _ = toy
    return analyse(tree, read_trait(TOY / "traits.tsv", "fast"), n_reps=300, n_perm=39, seed=1)


def test_missing_values_are_pruned(toy, slow):
    tree, _ = toy
    trait = read_trait(TOY / "traits.tsv", "slow")
    n_known = sum(v is not None for v in trait.values())
    assert n_known < tree.n_tips
    assert slow.tree.n_tips == n_known
    assert slow.states == ["absent", "present"]


def test_slow_trait_is_informative_at_depth_and_fast_trait_is_not(slow, fast):
    assert slow.info.normalised_information[0] > 0.4
    assert fast.info.normalised_information[0] < 0.02
    # with no information the root posterior is just the root prior
    assert fast.posterior[0] == pytest.approx([0.5, 0.5], abs=0.02)
    assert fast.saturation_index > 10 * slow.saturation_index
    assert summary(fast)["information"]["fraction_uninformative"] > 0.5
    assert summary(slow)["information"]["fraction_informative"] > 0.9


def test_node_table_contents(slow):
    table = node_table(slow)
    assert len(table) == slow.tree.n_tips - 1
    for column in (
        "tip_a", "tip_b", "height_half_lives", "map_state", "post_absent", "post_present",
        "normalised_information", "expected_accuracy", "baseline_accuracy", "verdict",
        "model_disagreement", "ni_slow_rate", "ni_fast_rate", "verdict_robust",
    ):  # fmt: skip
        assert column in table.columns
    assert np.allclose(table.post_absent + table.post_present, 1.0)
    assert table.normalised_information.between(0, 1).all()
    assert set(table.verdict) <= {"informative", "weak", "uninformative"}
    # tip_a and tip_b identify each node: their MRCA is the node itself
    index = slow.tree.tip_index()
    for row in table.itertuples():
        assert slow.tree.mrca([index[row.tip_a], index[row.tip_b]]) == row.node


def test_named_nodes(toy, slow):
    _, named = toy
    table = named_node_table(slow, named)
    assert list(table.name) == ["root", "clade A", "clade B"]
    assert table.loc[0, "node"] == 0
    assert table.n_tips.iloc[0] == table.n_tips.iloc[1] + table.n_tips.iloc[2]
    with pytest.warns(UserWarning, match="fewer than two"):
        assert named_node_table(slow, {"ghost": ["nobody", "sp001"]}).empty


def test_summary_is_json_serialisable(slow):
    text = json.dumps(summary(slow))
    back = json.loads(text)
    assert back["n_tips"] == slow.tree.n_tips
    assert back["signal_test"]["p_value"] <= 0.05
    assert back["rate_scale_interval"]["low"] < 1 < back["rate_scale_interval"]["high"]


def test_analyse_input_checks(toy):
    tree, _ = toy
    with pytest.raises(ValueError, match="single state"):
        analyse(tree, dict.fromkeys(tree.tip_names, "x"), n_reps=2, n_perm=0)
    with pytest.raises(ValueError, match="fewer than four"):
        analyse(tree, {tree.tip_names[0]: "x"}, n_reps=2, n_perm=0)
    states = dict.fromkeys(tree.tip_names[:30], "x") | dict.fromkeys(tree.tip_names[30:60], "y")
    states["not_in_tree"] = "x"
    with pytest.warns(UserWarning, match="not tips of the tree"):
        analyse(tree, states, n_reps=2, n_perm=0, rate_sensitivity=False)


def test_conflicting_states_at_zero_distance_need_a_branch_floor():
    from mkhorizon import parse_newick

    tree = parse_newick("(((A:0,B:0):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);")
    states = dict(zip("ABCDEFGH", "xyxxyyxy", strict=True))  # A and B differ
    with pytest.raises(ValueError, match="zero-length"):
        analyse(tree, states, n_reps=2, n_perm=0, rate_sensitivity=False)
    a = analyse(tree, states, n_reps=2, n_perm=0, rate_sensitivity=False, min_branch_length=1e-3)
    assert np.isfinite(a.loglik)


def test_trait_table_reader(tmp_path):
    path = tmp_path / "traits.csv"
    path.write_text("id,colour,size\nA,red,big\nB,,small\nC,NA,big\n")
    assert read_trait(path, "colour") == {"A": "red", "B": None, "C": None}
    assert read_trait(path, "size", id_col="id")["B"] == "small"
    with pytest.raises(KeyError):
        read_trait(path, "weight")
    path.write_text("id\tcolour\nA\tred\nA\tblue\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_trait(path, "colour")


def test_named_nodes_reader(tmp_path):
    path = tmp_path / "nodes.tsv"
    path.write_text("# comment\nancestor\tA, B,C\n\n")
    assert read_named_nodes(path) == {"ancestor": ["A", "B", "C"]}
    path.write_text("bad line without a tab\n")
    with pytest.raises(ValueError):
        read_named_nodes(path)
    path.write_text("lonely\tA\n")
    with pytest.raises(ValueError):
        read_named_nodes(path)


def test_cli_run_writes_every_output(tmp_path, capsys):
    out = tmp_path / "out"
    code = main([
        "run", "--tree", str(TOY / "tree.nwk"), "--traits", str(TOY / "traits.tsv"),
        "--trait", "fast", "--nodes", str(TOY / "nodes.tsv"),
        "--reps", "60", "--perm", "19", "--out", str(out),
    ])  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert "half-lives" in printed and "uninformative" in printed
    nodes = pd.read_csv(out / "nodes.tsv", sep="\t")
    named = pd.read_csv(out / "named_nodes.tsv", sep="\t")
    info = json.loads((out / "summary.json").read_text())
    assert len(nodes) == info["n_tips"] - 1
    assert list(named.name) == ["root", "clade A", "clade B"]
    assert [n["name"] for n in info["named_nodes"]] == ["root", "clade A", "clade B"]
    assert (out / "report.png").stat().st_size > 10_000


def test_cli_fit_and_errors(tmp_path, capsys):
    args = ["--tree", str(TOY / "tree.nwk"), "--traits", str(TOY / "traits.tsv")]
    assert main(["fit", *args, "--trait", "slow"]) == 0
    printed = capsys.readouterr().out
    assert "ER" in printed and "ARD" in printed
    assert main(["fit", *args, "--trait", "missing_column"]) == 2
    assert "error" in capsys.readouterr().err
    assert main(["fit", "--tree", str(tmp_path / "nope.nwk"), "--traits", "x", "--trait", "y"]) == 2
