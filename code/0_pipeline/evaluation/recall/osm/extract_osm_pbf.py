"""
Extract PV installations from a local OpenStreetMap dump.

**Needs the France .pbf, which is not redistributed.** Download it from
Geofabrik, or set PV_AUDIT_OSM_PBF. The output ships, so nothing downstream
depends on running this.

Writes one Point per OSM object, the centroid for ways, carrying the estimated
footprint area, the object version and the timestamp of its last edit.

This replaces an earlier Overpass-based route. The dump is local and frozen, so
the extraction is reproducible: the same .pbf gives exactly the same output,
which an API query cannot promise.

The extraction is deliberately **unfiltered**. recall.ipynb applies the
thresholds (footprint above 36 kWp, same-building clustering, the temporal
filter, deduplication) so that those thresholds stay parameters of the analysis
rather than being baked into the extraction.

Tag filter:

    power=generator + generator:source=solar
    + (generator:method=photovoltaic, or the tag absent)

Objects carrying an explicit ``generator:method`` other than photovoltaic, solar
thermal for instance, are excluded. ``location`` (roof, ground, and so on) is
kept as a property but not filtered on: many rooftop installations lack the tag,
and large ground-mounted plants are removed by the footprint filter anyway.

Two passes over the .pbf, with no location index, which keeps memory low:

1. objects matching the tag filter, both nodes (with direct coordinates) and
   ways (with node references). Relations are ignored, being rare for PV below
   36 kWp, and counted in the log.
2. coordinates of the nodes referenced by the matched ways, filtered by id on
   the C++ side.

Area comes from the way's polygon projected to Lambert-93, by the shoelace
formula. Nodes have no area, which usually means a small residential system.

On the metadata: ``version == 1`` implies the timestamp is the creation date,
the object never having been edited. Otherwise the timestamp is only an upper
bound on creation, and what to do about that (keep, exclude, mark ambiguous) is
decided in recall.ipynb rather than here.

Usage :
    python extract_osm_pbf.py                              # ../../../src/dump-osm/*.pbf -> osm_pv_france.geojson
    python extract_osm_pbf.py --pbf /chemin/france.osm.pbf --output osm_pv_france.geojson
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
import glob
import json
import sys
from pathlib import Path

import numpy as np
import osmium
from pyproj import Transformer

DEFAULT_PBF_DIR = Path("../../../../src/dump-osm")   # 0_data/src/dump-osm depuis recall/osm/
OUTPUT_FILE = Path(paths.OSM_PV)

TR_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


def match_tags(tags):
    """power=generator, generator:source=solar, photovoltaic method or none."""
    if tags.get("power") != "generator":
        return False
    if tags.get("generator:source") != "solar":
        return False
    method = tags.get("generator:method")
    return method is None or method == "photovoltaic"


def polygon_area_m2(lonlats):
    """Area in m2 of a lon/lat ring, via Lambert-93 and the shoelace formula."""
    if len(lonlats) < 3:
        return None
    xs, ys = TR_L93.transform([p[0] for p in lonlats], [p[1] for p in lonlats])
    xs, ys = np.asarray(xs), np.asarray(ys)
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2)


def pass1(pbf_path):
    """Objects matching the tag filter. Returns (nodes, ways, n_relations_matched)."""
    nodes, ways, n_rel = [], [], 0
    kept_keys = ("location", "generator:method", "generator:output:electricity", "building",
                 "source", "source:date")
    fp = osmium.FileProcessor(str(pbf_path)) \
        .with_filter(osmium.filter.KeyFilter("generator:source"))
    for obj in fp:
        tags = dict((t.k, t.v) for t in obj.tags)
        if not match_tags(tags):
            continue
        meta = {
            "version": obj.version if obj.version else None,
            "last_edit": obj.timestamp.isoformat() if obj.timestamp else None,
        }
        extra = {k: tags[k] for k in kept_keys if k in tags}
        if obj.is_node():
            nodes.append({"osm_id": f"node/{obj.id}", "lon": obj.lon, "lat": obj.lat,
                          **meta, **extra})
        elif obj.is_way():
            ways.append({"osm_id": f"way/{obj.id}", "refs": [n.ref for n in obj.nodes],
                         **meta, **extra})
        else:
            n_rel += 1
    return nodes, ways, n_rel


def pass2(pbf_path, ways):
    """Coordinates of the nodes referenced by matched ways, filtered by id."""
    needed = sorted({r for w in ways for r in w["refs"]})
    coords = {}
    fp = osmium.FileProcessor(str(pbf_path), osmium.osm.NODE) \
        .with_filter(osmium.filter.IdFilter(needed))
    for obj in fp:
        coords[obj.id] = (obj.lon, obj.lat)
    return coords


def main():
    parser = argparse.ArgumentParser(description="Extract PV installations from an OSM .pbf dump")
    parser.add_argument("--pbf", type=Path, default=None,
                        help=f"path to the .pbf (default: the single *.pbf in {DEFAULT_PBF_DIR}/)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    pbf = args.pbf
    if pbf is None:
        candidates = sorted(glob.glob(str(DEFAULT_PBF_DIR / "*.pbf")))
        if len(candidates) != 1:
            sys.exit(f"{len(candidates)} .pbf files in {DEFAULT_PBF_DIR}/; pass --pbf")
        pbf = Path(candidates[0])
    print(f"Dump : {pbf} ({pbf.stat().st_size / 1e9:.1f} Go)", flush=True)

    print("pass 1 of 2: PV objects matching the tag filter", flush=True)
    nodes, ways, n_rel = pass1(pbf)
    print(f"  {len(nodes)} nodes, {len(ways)} ways matches"
          + (f" ({n_rel} relations ignorees)" if n_rel else ""), flush=True)

    print("pass 2 of 2: coordinates of the ways' nodes", flush=True)
    coords = pass2(pbf, ways)
    print(f"  {len(coords)} nodes resolus", flush=True)

    features = []
    n_incomplete = 0
    for n in nodes:
        props = {k: v for k, v in n.items() if k not in ("lon", "lat")}
        props.update({"osm_type": "node", "area_m2": None})
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Point", "coordinates": [round(n["lon"], 7), round(n["lat"], 7)]}})
    for w in ways:
        pts = [coords[r] for r in w["refs"] if r in coords]
        if len(pts) < 3:
            n_incomplete += 1
            continue
        lon = float(np.mean([p[0] for p in pts]))
        lat = float(np.mean([p[1] for p in pts]))
        props = {k: v for k, v in w.items() if k != "refs"}
        props.update({"osm_type": "way", "area_m2": round(polygon_area_m2(pts), 1)})
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]}})
    if n_incomplete:
        print(f"  {n_incomplete} incomplete ways skipped (nodes missing from the dump)", flush=True)

    out = {"type": "FeatureCollection", "features": features}
    with open(args.output, "w") as f:
        json.dump(out, f)
    n_meta = sum(1 for ft in features if ft["properties"].get("last_edit"))
    print(f"wrote {args.output}: {len(features)} points, {n_meta} with version and timestamp", flush=True)


if __name__ == "__main__":
    main()
