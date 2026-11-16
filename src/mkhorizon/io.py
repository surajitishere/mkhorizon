"""Reading trait tables and named-ancestor definitions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["read_table", "read_trait", "read_named_nodes"]

_MISSING = {"", "NA", "NaN", "nan", "None", "?", "-"}


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a delimited table as strings; the delimiter is sniffed (tab or comma)."""
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)


def read_trait(path: str | Path, trait: str, id_col: str | None = None) -> dict[str, str | None]:
    """Tip name -> state for one column of a trait table.

    ``id_col`` defaults to the first column. Empty cells and the usual
    missing-value markers become ``None``.
    """
    table = read_table(path)
    id_col = id_col or table.columns[0]
    for col in (id_col, trait):
        if col not in table.columns:
            raise KeyError(f"column {col!r} not found; available: {list(table.columns)}")
    if table[id_col].duplicated().any():
        dup = table.loc[table[id_col].duplicated(), id_col].iloc[0]
        raise ValueError(f"duplicate identifier in {id_col!r}: {dup!r}")
    return {
        name: (None if value.strip() in _MISSING else value.strip())
        for name, value in zip(table[id_col], table[trait], strict=True)
    }


def read_named_nodes(path: str | Path) -> dict[str, list[str]]:
    """Named ancestors, one per line: ``name<TAB>tip1,tip2[,tip3...]``.

    The ancestor is the most recent common ancestor of the listed tips. Lines
    starting with ``#`` are ignored.
    """
    out: dict[str, list[str]] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"expected 'name<TAB>tip,tip,...', got: {raw!r}")
        tips = [t.strip() for t in parts[1].split(",") if t.strip()]
        if len(tips) < 2:
            raise ValueError(f"named node {parts[0]!r} needs at least two tips")
        out[parts[0].strip()] = tips
    return out
