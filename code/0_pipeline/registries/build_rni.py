"""Flatten the public registry (RNI) across the 2017-2025 annual publications.

Input: the annual parquet extracts, each a snapshot at 31 December, restricted
to the aggregate of installations below 36 kW. These are open data from the
national energy data platform and are not redistributed here; place them in a
``raw/`` folder beside this script, or set ``RNI_RAW_DIR``.

Output: ``paths.RNI``, one row per IRIS or municipality aggregate per year, with
the installation count, the capacity and the year.

    python build_rni.py
"""

import sys as _sys
from pathlib import Path as _Path


def _repo_root(start=_Path(__file__).resolve().parent):
    for parent in [start, *start.parents]:
        if (parent / "paths.py").exists() and (parent / "code").is_dir():
            return parent
    raise FileNotFoundError("this script must live inside the repository")


_sys.path.insert(0, str(_repo_root()))
import paths

import os

import pandas as pd

RAW_DIR = os.environ.get("RNI_RAW_DIR", "raw")

# The registry publishes several aggregation levels in the same file. This is
# the one that matches the paper's perimeter.
SMALL_ROOFTOP_LABEL = "Agrégation des installations de moins de 36KW"

COLS = ["nominstallation", "codeiris", "codeinseecommune", "commune",
        "codedepartement", "departement", "coderegion", "region",
        "nbinstallations", "puismaxrac"]

rni = {}
for file in os.listdir(RAW_DIR):
    year = int(file.split("-")[-1].split(".")[0][-2:])
    tmp = pd.read_parquet(os.path.join(RAW_DIR, file))
    tmp = tmp[tmp["nominstallation"] == SMALL_ROOFTOP_LABEL]
    tmp = tmp[COLS].rename(columns={"puismaxrac": "p_inst"})
    rni[2000 + year] = tmp

rni = dict(sorted(rni.items()))

rni_flatten = pd.concat(
    [df.assign(year=year) for year, df in rni.items()],
    ignore_index=True,
)
rni_flatten.to_csv(paths.RNI, index=False)
print(f"{len(rni_flatten):,} rows across {len(rni)} vintages -> {paths.RNI}")
