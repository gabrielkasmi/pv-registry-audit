"""
Draw the two blind re-labelling samples of the annotation-quality protocol.

``to_relabel_precision.geojson`` — 500 detections drawn from the precision
sample, for Cohen's kappa on the true/false positive labels, that is, the
reproducibility of the judgement itself. Departments under audit are
oversampled: Paris and the weakly instrumented over-reported units get 40 points
each, and the remainder is drawn uniformly across the country.

``to_relabel_recall_fn.geojson`` — 250 unmatched ground-truth points, those the
detector missed, drawn from the recall sample. This is a validity re-review
rather than an agreement test: was the installation genuinely visible on the
imagery vintage the model saw? Half the sample is drawn from the eight
over-reported departments,
                                moitie en tirage uniforme.

**Blind by construction.** The exported files carry no original label at all,
only a sample id, the department and the stratum. That is what makes the
agreement measurable: a reviewer who can see the first label is not producing an
independent judgement. The key that maps sample ids back to original labels is
written separately and is not to be opened during annotation.

Geometry is a 1 by 1 m square centred on the point, which is what the annotation
tool expects. Returned annotations land in the label folders, on the same
conventions as the original campaign. The analysis lives in kappa.ipynb.

The draw is deterministic: re-running reproduces exactly the same samples.
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

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

SEED = 42
HALF_M = 0.5   # half-side of the 1 by 1 m square

PRECISION_FILE = Path(paths.PRECISION_POINTS)
RECALL_FILE = Path(paths.RECALL_POINTS)

PRECISION_N_TOTAL = 500
PRECISION_OVERSAMPLE = {'75': 40, '89': 40, '54': 40, '27': 40, '50': 40, '37': 40}
RECALL_N_TOTAL = 250
RECALL_ABOVE_DEPTS = ['89', '27', '54', '50', '37', '31', '81', '34']
RECALL_N_PER_ABOVE = 15

TO_L93 = Transformer.from_crs('EPSG:4326', 'EPSG:2154', always_xy=True)
TO_WGS = Transformer.from_crs('EPSG:2154', 'EPSG:4326', always_xy=True)


def square_feature(lon, lat, props):
    x, y = TO_L93.transform(lon, lat)
    ring = [TO_WGS.transform(x + dx, y + dy)
            for dx, dy in [(-HALF_M, -HALF_M), (HALF_M, -HALF_M), (HALF_M, HALF_M),
                           (-HALF_M, HALF_M), (-HALF_M, -HALF_M)]]
    return {'type': 'Feature', 'properties': props,
            'geometry': {'type': 'Polygon', 'coordinates': [[list(c) for c in ring]]}}


def sample_stratified(df, oversample, n_total, rng):
    parts = []
    for dpt, n in oversample.items():
        pool = df[df['dpt'] == dpt]
        parts.append(pool.sample(min(n, len(pool)), random_state=rng))
    over = pd.concat(parts) if parts else df.iloc[0:0]
    rest_pool = df.drop(over.index)
    n_rest = n_total - len(over)
    rest = rest_pool.sample(n_rest, random_state=rng)
    over = over.assign(stratum='oversampled')
    rest = rest.assign(stratum='uniform')
    out = pd.concat([over, rest]).sample(frac=1, random_state=rng)   # melange l'ordre
    return out.reset_index(drop=True)


def main():
    rng = np.random.RandomState(SEED)

    # Precision: Cohen's kappa. TP and FP pooled, original label not exported.
    pp = pd.read_csv(PRECISION_FILE, dtype={'dpt': str})
    samp = sample_stratified(pp, PRECISION_OVERSAMPLE, PRECISION_N_TOTAL, rng)
    feats = [square_feature(r['lon'], r['lat'],
                             {'sample_id': f'prec_{i:04d}', 'dpt': r['dpt'],
                              'stratum': r['stratum']})
             for i, r in samp.iterrows()]
    json.dump({'type': 'FeatureCollection', 'features': feats},
              open('to_relabel_precision.geojson', 'w'))
    # The key. Never to be opened while the annotation is under way.
    samp.assign(sample_id=[f'prec_{i:04d}' for i in range(len(samp))]) \
        .to_csv(paths.KEY_PRECISION, index=False)
    print(f"to_relabel_precision.geojson : {len(samp)} carres "
          f"({(samp['stratum'] == 'oversampled').sum()} oversampled)")

    # Recall: validity re-review of the false negatives.
    rp = pd.read_csv(RECALL_FILE, dtype={'dpt': str})
    fn = rp[rp['pred'] == 0].reset_index(drop=True)
    over_spec = {d: RECALL_N_PER_ABOVE for d in RECALL_ABOVE_DEPTS}
    samp_fn = sample_stratified(fn, over_spec, RECALL_N_TOTAL, rng)
    feats = [square_feature(r['lon'], r['lat'],
                             {'sample_id': f'fn_{i:04d}', 'dpt': r['dpt'],
                              'stratum': r['stratum'], 'source': r['source']})
             for i, r in samp_fn.iterrows()]
    json.dump({'type': 'FeatureCollection', 'features': feats},
              open('to_relabel_recall_fn.geojson', 'w'))
    samp_fn.assign(sample_id=[f'fn_{i:04d}' for i in range(len(samp_fn))]) \
           .to_csv(paths.KEY_RECALL_FN, index=False)
    print(f"to_relabel_recall_fn.geojson : {len(samp_fn)} carres "
          f"({(samp_fn['stratum'] == 'oversampled').sum()} from over-reported units)")


if __name__ == '__main__':
    main()
