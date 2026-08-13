"""Perimeter alignment: keep only detections at or below KWP_MAX.

This is the first postprocess step, before dating, and its output is the root of
everything downstream: dating, precision and recall evaluation, correction, and
the registry comparisons.

Why it exists. The paper's perimeter is rooftop PV at or below 36 kVA, the
segment the two reference registries cover. The detection pipeline produces a
small fraction of arrays whose *estimated* power, surface times a coefficient,
exceeds 36 kWp: large commercial roofs, merged adjacent systems, fragments of
ground-mounted plants. Keeping them would inflate every comparison against a
<=36 kVA registry by around 7% of raw capacity.

The same filter is applied symmetrically to the recall ground truth, on the
estimated size of the OSM footprint. Detections, ground truth and references
therefore share one perimeter.

A caveat, symmetric and not fixable without exhaustive ground truth: a partial
detection of a large plant can fall under the threshold and be kept, while an
over-segmented roof can exceed it and be dropped. The filter acts on estimated
power. It is a best effort, and the paper says so.

    python filter_detections.py
    python filter_detections.py --input <raw detections>.geojson --kwp-max 36
"""

import argparse
import json
import os
from pathlib import Path

# The unfiltered detection product, published separately as OpenPVMapper v3 and
# not part of this release: the filtered file is what the analysis consumes, and
# it ships. Pass --input to point at the unfiltered product if you have it.
DEFAULT_INPUT = Path(os.environ.get("DPVM_RAW_DETECTIONS", "latest_dpvm.geojson"))
KWP_MAX = 36.0


def main():
    parser = argparse.ArgumentParser(
        description="Filter detections to kWp <= KWP_MAX (perimeter alignment)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--kwp-max", type=float, default=KWP_MAX)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <input stem>_filtered.geojson in this folder")
    args = parser.parse_args()

    output = args.output or Path(f"{args.input.stem}_filtered.geojson")

    print(f"reading {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    n_before = len(data["features"])
    kwp_before = sum(ft["properties"].get("kWp") or 0 for ft in data["features"])
    data["features"] = [ft for ft in data["features"]
                        if (ft["properties"].get("kWp") or 0) <= args.kwp_max]
    n_after = len(data["features"])
    kwp_after = sum(ft["properties"].get("kWp") or 0 for ft in data["features"])

    with open(output, "w") as f:
        json.dump(data, f)

    print(f"{n_before} -> {n_after} arrays "
          f"({n_before - n_after} dropped above {args.kwp_max:.0f} kWp, "
          f"{(kwp_before - kwp_after) / 1000:.0f} MWp, "
          f"{(kwp_before - kwp_after) / kwp_before:.1%} of raw capacity)")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
