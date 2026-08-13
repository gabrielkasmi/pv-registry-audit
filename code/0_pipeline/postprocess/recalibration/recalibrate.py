"""Correct installed capacity and installation counts, unit by unit.

Each reporting unit's raw totals are multiplied by the detector's correction
factor P/R, propagated through a Monte-Carlo bootstrap (B=10,000 by default).
The sampling itself lives in ``bootstrap.py``, isolated so it can be reused.

Inputs, all defaulting to the locations declared in ``paths.py``:

``--detections``    the filtered detection file; columns used are ``array_id``
                    (counted), ``dpt`` and ``kWp`` (summed)
``--calibration``   ``table.csv``, the output of ``score_and_aggregation``, with
                    the per-unit precision and recall counts
``--departements``  departmental geometries

Output, one GeoJSON row per unit carrying its geometry and:

- ``n_inst_raw``, ``p_inst_raw`` -- raw count and capacity, before correction
- ``n_inst_mean``, ``p_inst_mean`` -- posterior mean after correction
- ``{n_inst,p_inst}_ci{95,99}_{low,high}`` -- credible intervals

A unit that appears in the detections but has no complete calibration keeps its
raw columns and gets NaN in the corrected ones. No correction is invented for a
unit that has no validation sample; see ``recalibrate()``.

    python recalibrate.py
    python recalibrate.py --n-boot 20000 --seed 123
    python recalibrate.py --detections <some other export>.geojson
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

import argparse
from pathlib import Path

import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from bootstrap import ALPHA_PRIOR, BETA_PRIOR, DEFAULT_N_BOOT, bootstrap_recalibration

# Reporting units: Paris and its inner ring are merged. See units.py at the root.
def _find_root(start, marker='paths.py'):
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(marker)
sys.path.insert(0, str(_find_root(Path(__file__).resolve().parent)))
import units

DETECTIONS_FILE = Path(paths.DETECTIONS)   # already filtered to <=36 kWp
CALIBRATION_FILE = Path(paths.EVAL_TABLE)
DEPARTEMENTS_FILE = Path(paths.DEPARTEMENTS_DETAILED)
OUTPUT_FILE = Path(paths.RECALIBRATION)
SEED = 42

RAW_COLS = ["n_inst_raw", "p_inst_raw"]
REDRESSED_SUFFIXES = ["mean", "ci95_low", "ci95_high", "ci99_low", "ci99_high"]


def aggregate_raw(detections_path):
    """Aggregate the detections by reporting unit, before any correction.

    Returns ``n_inst_raw``, the number of detected arrays, and ``p_inst_raw``,
    their total estimated capacity in kWp.
    """
    detections = gpd.read_file(detections_path, columns=["array_id", "dpt", "kWp"], ignore_geometry=True)
    detections["dpt"] = units.to_unit(detections["dpt"])   # 75, 92, 93, 94 -> 75PC
    agg = (
        detections.groupby("dpt")
        .agg(n_inst_raw=("array_id", "count"), p_inst_raw=("kWp", "sum"))
        .reset_index()
    )
    return agg


def _nan_row(dept, n_inst_raw, p_inst_raw):
    row = {"dpt": dept, "n_inst_raw": n_inst_raw, "p_inst_raw": p_inst_raw}
    for name in ["n_inst", "p_inst"]:
        for suffix in REDRESSED_SUFFIXES:
            row[f"{name}_{suffix}"] = np.nan
    return row


def recalibrate(agg, calibration, n_boot=DEFAULT_N_BOOT, seed=SEED):
    """Correct the raw totals of every unit in ``agg``.

    One correction factor per unit, applied to both quantities so they stay
    mutually consistent, with a deterministic per-unit draw. Returns one row per
    unit.
    """
    # The Beta parameters are read straight from table.csv, from the alpha_* and
    # beta_* columns that build_scores_table writes. The prior chosen upstream,
    # Jeffreys or empirical Bayes, therefore propagates to the bootstrap without
    # being re-derived here, which is what keeps the two in step. Older tables
    # without those columns fall back to reconstructing Jeffreys from the counts.
    beta_cols = ["alpha_precision", "beta_precision", "alpha_recall", "beta_recall"]
    has_beta_cols = all(col in calibration.columns for col in beta_cols)
    calib_cols = beta_cols if has_beta_cols else ["tp_precision", "fp_precision", "tp_recall", "fn_recall"]
    rows = []
    n_missing_calib = 0

    for _, row in agg.iterrows():
        dept, n_inst_raw, p_inst_raw = row["dpt"], row["n_inst_raw"], row["p_inst_raw"]
        calib = calibration.loc[calibration["dpt"] == dept]

        if calib.empty or calib[calib_cols].isna().any(axis=None):
            n_missing_calib += 1
            rows.append(_nan_row(dept, n_inst_raw, p_inst_raw))
            continue

        c = calib.iloc[0]
        if has_beta_cols:
            alpha_p, beta_p = c["alpha_precision"], c["beta_precision"]
            alpha_r, beta_r = c["alpha_recall"], c["beta_recall"]
        else:
            alpha_p, beta_p = ALPHA_PRIOR + c["tp_precision"], BETA_PRIOR + c["fp_precision"]
            alpha_r, beta_r = ALPHA_PRIOR + c["tp_recall"], BETA_PRIOR + c["fn_recall"]

        result = bootstrap_recalibration(
            {"n_inst": n_inst_raw, "p_inst": p_inst_raw},
            alpha_p, beta_p, alpha_r, beta_r,
            n_boot=n_boot, seed=seed, key=dept,
        )

        out_row = {"dpt": dept, "n_inst_raw": n_inst_raw, "p_inst_raw": p_inst_raw}
        for name in ["n_inst", "p_inst"]:
            for suffix in REDRESSED_SUFFIXES:
                out_row[f"{name}_{suffix}"] = result[name][suffix]
        rows.append(out_row)

    if n_missing_calib:
        print(f"warning: {n_missing_calib} unit(s) without a complete calibration; "
              f"corrected values left at NaN, raw values kept")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Correct installed capacity and installation counts per reporting "
                     "unit, by Monte-Carlo bootstrap of the precision/recall factor."
    )
    parser.add_argument("--detections", type=Path, default=DETECTIONS_FILE)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_FILE)
    parser.add_argument("--departements", type=Path, default=DEPARTEMENTS_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                         help=f"bootstrap draws per unit (default: {DEFAULT_N_BOOT})")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"reading detections: {args.detections}")
    agg = aggregate_raw(args.detections)
    print(f"{len(agg)} units, {int(agg['n_inst_raw'].sum())} arrays, "
          f"{agg['p_inst_raw'].sum():,.1f} kWp raw, before correction")

    print(f"reading calibration: {args.calibration}")
    calibration = pd.read_csv(args.calibration, dtype={"dpt": str})

    print(f"bootstrap (B={args.n_boot}, seed={args.seed})")
    result = recalibrate(agg, calibration, n_boot=args.n_boot, seed=args.seed)

    print(f"reading geometries: {args.departements}")
    departements = gpd.read_file(args.departements)[["code", "nom", "geometry"]]
    departements = units.dissolve_units(departements)   # one polygon for 75PC
    out_gdf = departements.merge(result, left_on="code", right_on="dpt", how="left")
    out_gdf = out_gdf.drop(columns=["dpt"]).rename(columns={"code": "dpt"})
    out_gdf = gpd.GeoDataFrame(out_gdf, geometry="geometry", crs=departements.crs)

    n_no_detections = out_gdf["n_inst_raw"].isna().sum()
    if n_no_detections:
        print(f"note: {n_no_detections} unit(s) in the geometry file carry no detection "
              f"at all; their rows are NaN in the output")

    # Write through a temporary file: a direct overwrite is refused on some filesystems.
    import tempfile, shutil, os
    _tmp = os.path.join(tempfile.mkdtemp(), args.output.name)
    out_gdf.to_file(_tmp, driver="GeoJSON")
    shutil.copy(_tmp, args.output)
    print(f"wrote {args.output} ({len(out_gdf)} units)")

    n_up = (result["p_inst_mean"] > result["p_inst_raw"]).sum()
    n_down = (result["p_inst_mean"] < result["p_inst_raw"]).sum()
    print(f"corrected upward: {n_up} unit(s); downward: {n_down} unit(s)")


if __name__ == "__main__":
    main()
