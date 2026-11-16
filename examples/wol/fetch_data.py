#!/usr/bin/env python
"""Download the public inputs of the worked example (about 40 MB) into ./data.

    python examples/wol/fetch_data.py

Every file is fetched from a pinned commit or tag and checked against a SHA-256
digest, so the example is reproducible byte for byte. Nothing is redistributed
with this repository; see README.md here for sources and licences.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"

WOL = "https://raw.githubusercontent.com/biocore/wol/ff92d2251c0ea72299a2ead1aecfd748a9acd698/data"
MADIN = (
    "https://raw.githubusercontent.com/bacteria-archaea-traits/bacteria-archaea-traits/"
    "396dcd4d98ca20b613ba280f5514611651b6fb91/output"
)
PICRUST = "https://raw.githubusercontent.com/picrust/picrust2/v2.5.2/picrust2/default_files"

FILES = {
    "astral.cons.nid.nwk": (
        f"{WOL}/trees/astral/branch_length/cons/astral.cons.nid.nwk",
        "3d600bc365c26ae1492f72d384f21c0c22afeb7a2e5c8b9eb5611d6efe9cdf8d",
    ),
    "metadata.tsv.xz": (
        f"{WOL}/genomes/metadata.tsv.xz",
        "b811ff3c69b6a6f9a94d55cc9a177bced4dddd8b1132603c47669bf132c84e9b",
    ),
    "condensed_species_NCBI.csv": (
        f"{MADIN}/condensed_species_NCBI.csv",
        "b9ab9c72ddcec447ef78a8c25282f79afba5b3183c6bce7c220b5c671bd3db2f",
    ),
    "ko.txt.gz": (
        f"{PICRUST}/prokaryotic/ko.txt.gz",
        "bb92a34e31e16cb1a27ad542c77eadc6b8b34a82fb8c8e367f258c92d19a69e0",
    ),
    "ko_info.tsv.gz": (
        f"{PICRUST}/description_mapfiles/ko_info.tsv.gz",
        "0de41d05252cc7c8b2cea31f416c7493fc4f24a5b38ac391ee9b0099415fa8ad",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    DATA.mkdir(exist_ok=True)
    for name, (url, expected) in FILES.items():
        target = DATA / name
        if not (target.exists() and sha256(target) == expected):
            print(f"fetching {name}")
            urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https URLs
        actual = sha256(target)
        if actual != expected:
            print(f"checksum mismatch for {name}: {actual}", file=sys.stderr)
            return 1
        print(f"ok  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
