
import sys as _sys
from pathlib import Path as _Path


def _repo_root(start=_Path(__file__).resolve().parent):
    for parent in [start, *start.parents]:
        if (parent / "paths.py").exists() and (parent / "code").is_dir():
            return parent
    raise FileNotFoundError("this script must live inside the repository")


_sys.path.insert(0, str(_repo_root()))
import paths

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from shapely.geometry import Point
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────
# 0. Config
# ────────────────────────────────────────────────────────────────────────
ASSEMBLAGE_DIR = Path(__file__).parent / "assemblage"
OUTPUT_NAME = paths.IMAGERY_DATES

# The mosaicking graphs hold polygons with very large vertex counts: loading a
# whole department at once can take several GB of RAM. The file is read in
# batches to keep the memory footprint bounded.
GRAPHE_CHUNK_SIZE = 300

# Folder names under ASSEMBLAGE_DIR follow several conventions depending on
# where the download came from: the product prefix varies in order, and the
# GPKG/SHP format token is sometimes absent. Only the stable suffix is matched.
# D{dept}_{year}-01-01
ASSEMBLAGE_SUFFIX_RE = re.compile(r"(D[0-9A-Za-z]{3})_(\d{4})-01-01$")


def ign_dept_code(dept: str) -> str:
    """'2' -> 'D002', '35' -> 'D035', '2A' -> 'D02A', '971' -> 'D971'."""
    return f"D{dept.zfill(3)}"


def vector_file_in(folder: Path) -> Path | None:
    """Find a .gpkg or .shp in the folder, recursively.

    The archive sometimes unpacks with one extra level of nesting.
    """
    vector_files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".gpkg", ".shp"))
    return vector_files[0] if vector_files else None


def scan_assemblage(dept_universe: list[str]) -> dict[str, tuple[int, Path]]:
    """Scan ASSEMBLAGE_DIR and return ``{dpt: (year, folder)}`` for every graph found.

    ``dpt`` uses the same codes as the detections. Where several folders exist
    for one department, the most recent is kept.
    """
    ign_to_dept = {ign_dept_code(d): d for d in dept_universe}

    found: dict[str, tuple[int, Path]] = {}
    for folder in sorted(ASSEMBLAGE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        m = ASSEMBLAGE_SUFFIX_RE.search(folder.name)
        if not m:
            continue
        ign_code, year = m.group(1).upper(), int(m.group(2))
        dept = ign_to_dept.get(ign_code)
        if dept is None:
            continue  # folder present, but this department is out of scope
        if dept not in found or year > found[dept][0]:
            found[dept] = (year, folder)
    return found


def barycenter(points):
    return Point(np.mean([p.x for p in points]), np.mean([p.y for p in points]))


def dates_from_graphe(detections_dept: gpd.GeoDataFrame, vector_path: Path) -> pd.DataFrame:
    """Intersect one department's installations with the mosaicking graph.

    Returns one date per municipality, the most recent among its installations.

    Read in batches of GRAPHE_CHUNK_SIZE features. A full ``gpd.read_file``, or
    even one filtered by bounding box, ends up loading nearly every tile, since
    the installations cover the whole department, and can consume several GB of
    RAM because of the polygons' vertex counts.
    """
    detections_l93 = detections_dept.to_crs(epsg=2154)[["array_id", "insee", "geometry"]]

    info = pyogrio.read_info(str(vector_path))
    total_features = info["features"]
    crs = info["crs"]

    matches = []
    for start in range(0, total_features, GRAPHE_CHUNK_SIZE):
        chunk = pyogrio.read_dataframe(str(vector_path), skip_features=start, max_features=GRAPHE_CHUNK_SIZE)
        chunk = gpd.GeoDataFrame(chunk, geometry="geometry", crs=crs)
        chunk["DATE"] = pd.to_datetime(chunk["DATE"])

        joined = gpd.sjoin(
            detections_l93,
            chunk[["DATE", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        if len(joined):
            matches.append(joined[["array_id", "insee", "DATE"]])

    if not matches:
        return pd.DataFrame(columns=["insee", "imagedate"])

    # An installation straddling several tiles takes the most recent date.
    per_installation = pd.concat(matches, ignore_index=True).groupby(["array_id", "insee"], as_index=False)["DATE"].max()

    # Per municipality: the most recent date among its installations.
    per_commune = (
        per_installation.groupby("insee", as_index=False)["DATE"]
        .max()
        .rename(columns={"DATE": "imagedate"})
    )
    return per_commune


def run(
    src_path: Path,
    dest_dir: Path,
    dpt_list: list[str] | None = None,
    to_exclude: list[str] | None = None,
    update: list[str] | None = None,
):
    # ── 1. Chargement ──────────────────────────────────────────────
    print(f"reading detections from {src_path}")
    detections = gpd.read_file(src_path)
    print(f"{len(detections)} installations, {detections['insee'].nunique()} municipalities")

    if dpt_list:
        detections = detections[detections["dpt"].isin(dpt_list)].reset_index(drop=True)
    if to_exclude:
        detections = detections[~detections["dpt"].isin(to_exclude)].reset_index(drop=True)

    universe = sorted(detections["dpt"].unique())
    print(f"{len(universe)} departments in scope for this run")

    # 2. Existing output file: this script accumulates incrementally.
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / OUTPUT_NAME

    existing = gpd.read_file(output_path) if output_path.exists() else None
    depts_done = sorted(existing["dpt"].unique()) if existing is not None else []

    # 2b. Forced update: drop the requested departments from "already done" and
    # from the existing output, so they go through normal processing, including
    # the usual year check, and replace their old rows.
    #
    # Keep a copy of what was removed. A department requested with --update can
    # end up in error or pending, if its graph is missing or its year does not
    # match, in which case it never reaches depts_todo and its rows would be
    # lost outright. Step 3b restores them.
    existing_removed = None
    if update:
        not_done = [d for d in update if d not in depts_done]
        if not_done:
            print(f"--update: {', '.join(not_done)} not yet in {OUTPUT_NAME}, processed normally")
        depts_done = [d for d in depts_done if d not in update]
        if existing is not None:
            existing_removed = existing[existing["dpt"].isin(update)].reset_index(drop=True)
            existing = existing[~existing["dpt"].isin(update)].reset_index(drop=True)
            if existing.empty:
                existing = None

    # 3. Discover the graphs available locally and sort the departments.
    assemblage = scan_assemblage(universe)

    depts_todo, depts_error, depts_remaining = [], [], []
    error_details = []

    for dept in universe:
        if dept in depts_done:
            continue
        if dept not in assemblage:
            depts_remaining.append(dept)
            continue

        year_found, _ = assemblage[dept]
        year_expected = int(detections.loc[detections["dpt"] == dept, "year"].max())
        if year_found != year_expected:
            depts_error.append(dept)
            error_details.append((dept, year_expected, year_found))
            continue

        depts_todo.append(dept)

    print("─" * 50)
    print(f"already done   : {len(depts_done)}")
    print(f"to process     : {len(depts_todo)} -> {', '.join(depts_todo) if depts_todo else '-'}")
    print(f"in error       : {len(depts_error)} -> {', '.join(depts_error) if depts_error else '-'}")
    print(f"pending        : {len(depts_remaining)} -> {', '.join(depts_remaining) if depts_remaining else '-'}")
    if error_details:
        print("errors in detail (department: detection year vs graph year):")
        for dept, y_exp, y_found in error_details:
            print(f"  {dept}: {y_exp} (detections) vs {y_found} (assemblage)")
    print("─" * 50)

    # 3b. Restore the --update departments that were not actually reprocessed.
    # Their graph was missing or its year did not match, so they went to error
    # or pending rather than depts_todo. Without this, the removal in step 2b
    # would drop them from the output for good, with nothing replacing them.
    if existing_removed is not None and len(existing_removed):
        lost = [d for d in existing_removed["dpt"].unique() if d not in depts_todo]
        if lost:
            print(f"--update: {', '.join(lost)} not reprocessed this run (see errors and pending "
                  f"above); their previous rows are restored as they were, nothing is lost")
            restored = existing_removed[existing_removed["dpt"].isin(lost)]
            existing = (
                gpd.GeoDataFrame(pd.concat([existing, restored], ignore_index=True), geometry="geometry", crs="EPSG:4326")
                if existing is not None else
                gpd.GeoDataFrame(restored, geometry="geometry", crs="EPSG:4326")
            )
            depts_done = sorted(set(depts_done) | set(lost))

    if not depts_todo:
        print("nothing new to process")
        return existing

    # 4. Process this run's departments.
    detections_todo = detections[detections["dpt"].isin(depts_todo)]

    gdf_l93 = detections_todo.to_crs(epsg=2154)
    gdf_l93["centroid"] = gdf_l93.geometry.centroid
    barycenters_l93 = gdf_l93.groupby("insee")["centroid"].apply(barycenter)
    barycenters_l93 = gpd.GeoSeries(barycenters_l93, crs=2154)
    barycenters = barycenters_l93.to_crs(4326)

    all_dates = []
    for dept in tqdm(list(depts_todo), desc="mosaicking graph", unit="dept"):
        detections_dept = detections_todo[detections_todo["dpt"] == dept]
        _, folder = assemblage[dept]
        vector_path = vector_file_in(folder)
        if vector_path is None:
            # Should not happen, the folder matched the pattern. Defensive.
            depts_error.append(dept)
            depts_todo.remove(dept)  # iterating over a copy, so mutation is safe
            continue
        per_commune = dates_from_graphe(detections_dept, vector_path)
        all_dates.append(per_commune)

    # detections_todo was built before the loop. Drop the departments that
    # failed, so they are not counted as processed.
    detections_todo = detections_todo[detections_todo["dpt"].isin(depts_todo)]

    imagedates = (
        pd.concat(all_dates, ignore_index=True)
        if all_dates
        else pd.DataFrame(columns=["insee", "imagedate"])
    )

    # 5. Aggregate by municipality, for this run's departments.
    commune_stats = (
        detections_todo.groupby("insee")
        .agg(
            nom=("nom", "first"),
            dpt=("dpt", "first"),
            dpt_source=("dpt_source", "first"),
            n_installations=("array_id", "count"),
            kWp_total=("kWp", "sum"),
            year_median=("year", "median"),
        )
        .reset_index()
    )
    commune_stats = commune_stats.merge(
        barycenters.rename("geometry").reset_index(), on="insee"
    ).merge(imagedates, on="insee", how="left")

    new_gdf = gpd.GeoDataFrame(commune_stats, geometry="geometry", crs="EPSG:4326")

    # 6. Write: append to the existing file. The --update departments were
    # already removed in step 2b, so there is no duplication.
    if existing is not None:
        combined = gpd.GeoDataFrame(
            pd.concat([existing, new_gdf], ignore_index=True), geometry="geometry", crs="EPSG:4326"
        )
    else:
        combined = new_gdf

    combined.to_file(output_path, driver="GeoJSON")

    print("─" * 50)
    print("summary")
    print(f"  departments processed        : {len(depts_todo)}")
    if update:
        updated_now = [d for d in depts_todo if d in update]
        print(f"  ... of which updated         : {len(updated_now)} -> {', '.join(updated_now) if updated_now else '-'}")
    print(f"  municipalities added/updated : {len(new_gdf)}")
    print(f"  Dates manquantes (ce run)    : {new_gdf['imagedate'].isna().sum()}")
    print(f"  total in file                : {len(combined)} municipalities, {len(depts_done) + len(depts_todo)} departments")
    print(f"  pending                      : {len(depts_remaining)}")
    print(f"  in error                     : {len(depts_error)}")
    print(f"  wrote                        : {output_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Recover the orthoimagery acquisition date per municipality from the "
                     "mosaicking graph, and aggregate the PV installations. Processes the "
                     "departments available locally and not yet in the output file."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(paths.DETECTIONS),
        help="path to the source detections GeoJSON",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("."),
        help="destination folder for the output",
    )
    parser.add_argument(
        "--dpt",
        type=str,
        nargs="+",
        default=None,
        help="restrict the scope to these departments (--dpt 54 08 51 or --dpt 54,08,51). "
             "Default: every department present in the source file.",
    )
    parser.add_argument(
        "--to_exclude",
        type=str,
        nargs="+",
        default=None,
        help="exclude these departments from the scope",
    )
    parser.add_argument(
        "--update",
        type=str,
        nargs="+",
        default=None,
        help="force reprocessing of these departments even if already in "
             f"{OUTPUT_NAME}. Their previous rows are replaced from the current "
             "graphs. Other departments keep the usual incremental behaviour.",
    )
    args = parser.parse_args()

    def parse_dept_arg(raw):
        if not raw:
            return None
        return [d.strip() for part in raw for d in part.split(",") if d.strip()]

    dpt_list = parse_dept_arg(args.dpt)
    to_exclude = parse_dept_arg(args.to_exclude)
    update = parse_dept_arg(args.update)

    run(args.src, args.dest, dpt_list, to_exclude, update)


if __name__ == "__main__":
    main()
