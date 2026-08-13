"""Companion to extract_osm_pbf.py: the `source` and `source:date` tags of each PV object.

**Needs the France OpenStreetMap dump, which is not redistributed.** Download it
from Geofabrik, or set OSM_PBF_DIR. The output ships, so nothing downstream
depends on running this.

One pass over the dump; the geometries in osm_pv_france.geojson stay valid. The
output is merged by recall.ipynb to requalify ground-truth points marked
`posterior`: an object whose source tag is an orthoimage dated at or before the
imagery the model saw was in fact visible to the model, and counts as a genuine
miss rather than a system built after the fact.
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

from pathlib import Path
import csv

import osmium

DEFAULT_PBF_DIR = paths.OSM_PBF.parent
OUTPUT = Path(paths.OSM_SOURCES)


def match_tags(tags):
    if tags.get("power") != "generator":
        return False
    if tags.get("generator:source") != "solar":
        return False
    method = tags.get("generator:method")
    return method is None or method == "photovoltaic"


def main():
    pbfs = sorted(DEFAULT_PBF_DIR.glob("*.osm.pbf"))
    assert pbfs, f"no .osm.pbf found in {DEFAULT_PBF_DIR}; see the module docstring"
    pbf = pbfs[-1]
    print(f"dump: {pbf}")
    rows = []
    fp = osmium.FileProcessor(str(pbf)).with_filter(
        osmium.filter.KeyFilter("generator:source"))
    for obj in fp:
        tags = dict((t.k, t.v) for t in obj.tags)
        if not match_tags(tags):
            continue
        src = tags.get("source")
        srcd = tags.get("source:date")
        if src is None and srcd is None:
            continue
        otype = "node" if obj.is_node() else ("way" if obj.is_way() else "relation")
        rows.append({"osm_type": otype, "osm_id": obj.id,
                     "source": src or "", "source_date": srcd or ""})
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["osm_type", "osm_id", "source", "source_date"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUTPUT}: {len(rows)} objects carrying a source tag")


if __name__ == "__main__":
    main()
