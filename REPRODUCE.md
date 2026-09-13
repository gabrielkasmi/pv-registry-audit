# Reproducing the results

Read this alongside the "What reproduces, and what does not" section of the
README, which sets out four categories and puts every file in one of them. This
document gives the order.

## Before anything

Three steps: clone the code, fetch the data, check the wiring.

```bash
git clone https://github.com/gabrielkasmi/pv-registry-audit.git
cd pv-registry-audit
conda env create -f environment.yml
conda activate pv-registry-audit
```

The replication dataset is a **separate Zenodo deposit** with its own DOI,
distinct from the code archive: **<https://doi.org/10.5281/zenodo.22729786>**. Download and
unpack it, either at `data/` inside the clone, or anywhere with a variable
pointing at it.

```bash
unzip pv-registry-audit-data.zip -d /somewhere
export PV_AUDIT_DATA=/somewhere/data

python paths.py
```

Every path prints `ok` or `MISSING`. Exactly two should be missing, the
OpenStreetMap dump and the IGN boundary set, and nothing below depends on them.
If anything else is missing, stop here: the rest will fail later and less
legibly.

## One important habit

**Notebooks write into the data directory.** Running them in place overwrites the
released files with your own regeneration, which is fine until you want to
compare the two and no longer can.

Point `PV_AUDIT_DATA` at a copy before re-running anything:

```bash
cp -r data /tmp/replication
export PV_AUDIT_DATA=/tmp/replication
```

Then diff the result against the release. That is the whole reason the location
is an environment variable rather than a hard-coded path.

## The short path: check the headline result

Two minutes, no re-running.

```bash
python code/1_audit/gen_comparison.py
```

Regenerates the registry comparison table from the released intermediate data.
Expect the corrected national estimate at 4,033 MWp against 3,901 MWp in the
connection registry, a 3.4% surplus, and verdicts of 60 within, 25 under-reported
and 8 over-reported.

## The full order

Each step consumes the previous one's output. Every output already ships, so any
step can be run on its own.

### 1. Sources

| | |
|---|---|
| `code/0_pipeline/registries/build_rni.py` | Flattens the nine annual public-registry snapshots, which ship, into one table. |
| `code/0_pipeline/postprocess/filtering/filter_detections.py` | Filters the detection product to the perimeter. Needs the unfiltered product, published separately. |

### 2. Evaluation: what the detector gets right

| | |
|---|---|
| `evaluation/precision/precision.ipynb` | Aggregates the annotation exports into precision points. |
| `evaluation/recall/recall.ipynb` | Builds the recall ground truth and matches it against the detections. |
| `evaluation/quality_assessment/kappa.ipynb` | Label agreement, and the re-review of missed installations. **Run before `score_and_aggregation`**: it writes the validity correction that the recall counts depend on. |
| `evaluation/quality_assessment/gt_bias_tests.ipynb` | Source bias and the vintage effect. |
| `evaluation/score_and_aggregation.ipynb` | Produces the calibration table, which is the interface to everything downstream. |

The OpenStreetMap extraction scripts sit upstream of `recall.ipynb` and need the
France dump. Their outputs ship; see README category 2.

### 3. Postprocess: from detections to comparable quantities

| | |
|---|---|
| `postprocess/dates/datation.py` | Imagery date per municipality. Needs the IGN mosaicking graphs; output ships. |
| `postprocess/recalibration/recalibrate.py` | Corrects each unit's raw totals. This is where the 4,033 MWp comes from. |
| `postprocess/merge_sources/merge.py` | Date-matches the registries against the detections. **Cannot be replayed**; see README category 3. Its outputs ship, and everything below reads them. |
| `postprocess/visualization.ipynb` | Sanity checks on the dates and on the correction. |

### 4. Audit

| | |
|---|---|
| `1_audit/1_comparisons_rte.ipynb` | The audit against the connection registry, and the specification battery. |
| `1_audit/2_comparisons_rni_agg.ipynb` | Same audit, public registry departmental totals. |
| `1_audit/2_comparisons_rni_bottomup.ipynb` | Same audit, public registry municipal view. |
| `1_audit/3_truncation_bias.ipynb` | What the difference between those two measures. |
| `1_audit/gen_comparison.py` | The three side by side. |
| `sanity_checks/battery_specification.ipynb` | The battery, and the closed-form conversion threshold. |
| `sanity_checks/estimation_convergence_scales.ipynb` | What forming the factor at a coarser scale costs. Doubles as a regression test. |
| `sanity_checks/conversion_factor_anchor.ipynb` | Anchors the conversion coefficient. **Needs the restricted extract**; see README category 4. |

Section 6 of each audit notebook is the conversion anchoring and skips itself with
a message when the extract is absent. The notebook still runs to completion.

### 5. Mechanisms

| | |
|---|---|
| `2_mechanisms/1_temporal_lag.ipynb` | Can connection lag explain the gaps? Bounded in both directions. |
| `2_mechanisms/2_eld.ipynb` | Distribution-chain fragmentation, with the econometric frame. |
| `2_mechanisms/3_tagging.ipynb` | Unit-by-unit attribution. Runs last: it reads the other two. |

## What to expect, and what a difference means

Re-running the chain on the released data reproduces the shipped intermediate
files, with three exceptions that are understood rather than mysterious.

`temporal_lag_departements.csv` and `temporal_sensitivity_departements.csv` differ
from the values first computed for the paper by at most 0.5 MWp and 1.4 MWp
respectively. The released aggregates use day-level cutoffs per municipality,
while the original notebook took the departmental mode of the imagery date. The
aggregate is the more precise of the two, and the shipped files carry its values.

One consequence is worth knowing before you compare against the manuscript: in
the dating sweep, hard-core survival at the +1 month offset is 14 rather than 15.
The range the paper reports, 18 of 18 down to 14 of 18, is unchanged.

`temporal_lag_eld_departements.csv` carries ten columns rather than thirteen. The
three missing ones describe a month-by-month build-up that needs a connection date
per installation, which the release does not carry. The verdict itself is computed
from the endpoints and is unaffected.

Everything else regenerates identically, up to floating-point formatting. The
calibration table matches on all 33 columns to within 1e-9.

## If a number does not match

In rough order of likelihood:

1. **The data directory is stale.** A partial re-run leaves a mix of your outputs
   and the released ones. Start from a fresh copy.
2. **The prior.** Everything downstream inherits it from the alpha and beta
   columns of the calibration table. If those changed, every number changes, and
   the aggregation-scales notebook will say so in its first assertion.
3. **Notebook order.** `kappa.ipynb` before `score_and_aggregation.ipynb`, and
   `3_tagging.ipynb` last.
4. **Fractional counts.** The recall counts in the calibration table are not
   integers and must not be rounded: they carry the ground-truth validity
   correction, and rounding them moves the national total by 0.4%.
