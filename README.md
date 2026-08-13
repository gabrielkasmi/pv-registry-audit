# Auditing photovoltaic registries with remote sensing

Code and data for Kasmi et al. (2026), *Joule*.

Nationwide aerial imagery is processed by a detection pipeline, the detections
are corrected for the detector's own precision and recall, and the corrected
estimate is used to audit the two official registries of rooftop photovoltaic
capacity in France. The national estimate is 4,033 MWp; 18 reporting units are
under-reported under every specification, against 7 in the symmetric negative
control.

## Layout

```
code/
  0_pipeline/     detections -> annotations -> per-unit correction
  1_audit/        corrected estimate vs each registry, and the robustness battery
  2_mechanisms/   what explains the flagged gaps
data/             not in this repository; see below
figures/          every figure the notebooks produce, as PDF and PNG.
                  22 of them appear in the paper; the rest are diagnostics
                  kept so that re-running creates no untracked files.

paths.py          every data location, in one module
style.py          shared figure style
units.py          reporting units: departments, with Paris and its ring merged
dept_posteriors.py  regenerates a unit's posterior, identically to the main chain

REPRODUCE.md      the order in which to run things
COMPARISON.md     generated: the three registry references side by side
```

28 files under `code/`, plus the four shared modules at the root.

## Setting up

The code lives here; the data does not. It is deposited separately at Zenodo,
under its own DOI, because it is 546 MB of mostly binary geospatial files and
because it carries terms this repository does not.

```bash
git clone https://github.com/gabrielkasmi/pv-registry-audit.git
cd pv-registry-audit

conda env create -f environment.yml
conda activate pv-registry-audit
```

Then fetch the replication dataset from **<https://doi.org/[DOI-DATA]>** and
unpack it. Either put it at `data/` inside the clone, which needs no further
configuration, or put it anywhere and point one variable at it:

```bash
unzip pv-registry-audit-data.zip -d /somewhere
export PV_AUDIT_DATA=/somewhere/data
```

Check the wiring before running anything:

```bash
python paths.py
```

Every declared path prints with `ok` or `MISSING`. Two are expected to be
missing: the OpenStreetMap dump and the IGN boundary set, which are third-party
bulk downloads this release does not carry and which nothing you are likely to
run depends on. Anything else missing means the dataset is not where the code
thinks it is, and it is worth fixing here rather than three notebooks later.

`data/MANIFEST.csv` describes every file in the dataset: what it holds, what
produced it, and whether it can be regenerated. It is versioned in this
repository even though the data is not, so the contents of the deposit can be
inspected before downloading it.

## What reproduces, and what does not

Reproducibility is not a yes or no here, and the distinctions matter more than a
single claim would. Four categories, and every file falls in exactly one.

### 1. Runs from the release

Twenty-three of the twenty-eight files under `code/`, and with them every
published number and every figure. The national estimate, the per-unit verdicts,
the specification battery and its hard core, the aggregation-scale comparison,
the truncation analysis, the fragmentation and connection-lag mechanisms, the
annotation-quality tests.

Three of those twenty-three carry one section that does not run: the conversion
anchoring at the end of each audit notebook. It skips itself with a message
rather than raising, so the notebook still completes and the audit above it is
unaffected.

Start from `code/1_audit/` to check the headline result;
`gen_comparison.py` regenerates the registry comparison table in one call.

### 2. Upstream steps whose reproducibility starts from a precomputed file

Three scripts consume bulk third-party data that is open but too large to
redistribute: `extract_osm_pbf.py` and `extract_osm_sources.py` read the France
OpenStreetMap dump, and `datation.py` reads the IGN mosaicking graphs. Each
source is named exactly in `data/MANIFEST.csv`, edition included.

Their outputs ship, so reproduction starts from those outputs rather than from
the dumps, and nothing downstream is blocked. Re-running them is possible but is
not the intended path: it means fetching several gigabytes from a provider, and
in the OpenStreetMap case the dump is mutable, so a download today would not
return the snapshot this study read. That step is re-runnable rather than
reproducible in the strict sense, and saying so is more useful than implying
otherwise.

The municipal geometries needed by the truncation map are an exception: they are
shipped, simplified to a 200 m tolerance, which is below one pixel at the scale
that map is drawn. That removes the only case where a *published figure* would
have required a download.

### 3. Preparation steps that cannot be replayed, whose outputs ship

The municipality-level date matching in `code/0_pipeline/postprocess/merge_sources/merge.py`
compares each installation's connection date against its municipality's imagery
date. That comparison needs the operator's per-installation extract, which is not
ours to redistribute, and no aggregate can stand in for it: the granularity being
withheld is exactly the granularity the step operates at.

Its outputs are released, and every published result derives from them, so no
reported number becomes unverifiable. What is lost is the ability to re-run this
preparation step and check it independently.

The same is true of the 31,853 manual annotations, for a different reason. They
are released, they are the input to everything, and no amount of computation
would recreate them: they are human judgements. The raw exports from the
labelling platform ship alongside the aggregated points, so at least the
aggregation is checkable.

### 4. Results that cannot be reproduced, and what stands in for each

Two, and each has a documented substitute. `conversion_factor_anchor.ipynb` is
the only file that stops rather than skipping, and it stops with a message saying
why and what stands in for it.

**The municipality-level fixed-effects regression** of the distribution-chain
test needs municipality-level registry values, which are withheld: at that
granularity they would disclose what the public registry itself censors under its
privacy rule.

Its non-parametric counterpart reproduces exactly. The departmental medians by
operator type are released, giving 16 of 19 departments negative and a Wilcoxon
signed-rank *p* = 0.0003. The mechanism finding therefore rests on a test any
reader can re-run.

**The external anchoring of the surface-to-capacity coefficient** pools the
capacity distribution of every consistent unit, which the departmental aggregates
do not carry. Capacity deciles for three illustrative units are released instead.
They reproduce the congruence between the two size distributions, including the
pile-up at the 3 kWp connection tier on both sides, and place the implied
coefficient inside the 5.0 to 6.0 m²/kWp range, without recovering the anchoring
value itself.

This limits the justification for the retained coefficient, not the conclusions
that depend on it. Specification **C** varies the coefficient across that entire
range, and since it is a purely multiplicative rescaling, the survival of the
hard core under it reproduces in full from the released data.

Sections in categories 3 and 4 carry a banner at the top of the relevant notebook
cell or script, so this is visible where it bites rather than only here.

## The statistical core, as a package

The estimator itself is released separately as a standalone Python package,
`bayesian-pv-census`, with its own tests and documentation. It knows nothing about
photovoltaics or France: a *unit* is anything with a raw total and a validation
sample.

```bash
pip install bayesian-pv-census
```

## Related releases

| | |
|---|---|
| Detection product | OpenPVMapper v3, deposited at Zenodo |
| Mapping algorithm | <https://github.com/gabrielkasmi/deeppvmapper> |
| Annotation tooling | <https://github.com/gabrielkasmi/pv-annotation> |
| Statistical core | <https://github.com/gabrielkasmi/bayesian-pv-census>, <https://doi.org/10.5281/zenodo.21921771> |

## Citation

See `CITATION.cff`. Cite the paper for the results, the package DOI for the
estimator, and this repository's DOI for the analysis itself.

## Licence

MIT for the code, see `LICENSE`. The data release carries its own terms: see the
Zenodo deposit, and `data/source/rte_derived/README.md` for the restrictions
attached to the files derived from the operator's registry.
