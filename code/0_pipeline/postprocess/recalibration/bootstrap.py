"""Reusable core of the probabilistic precision/recall correction.

Isolated here so that any script or notebook in the postprocess stage can call
it without duplicating the sampling logic: any raw quantity derived from the
detections can be corrected by the same mechanism.

Convention, identical to the evaluation helpers: precision and recall each carry
a ``Beta(ALPHA_PRIOR + tp, BETA_PRIOR + fail)`` posterior, with a Jeffreys prior
``Beta(0.5, 0.5)``. The correction factor is precision over recall, so a
corrected quantity is the raw quantity times P/R. The full justification, why
P/R rather than F1 and what the homogeneity assumption buys, is in the
evaluation README.

The convention is restated here rather than imported, to keep this module
self-contained (no cross-dependency between the postprocess and evaluation
stages) and light (no matplotlib, geopandas or scipy).
"""

import hashlib

import numpy as np

ALPHA_PRIOR = 0.5
BETA_PRIOR = 0.5
DEFAULT_N_BOOT = 10000
DEFAULT_CI_LEVELS = (0.95, 0.99)


def beta_params(tp, fail, alpha_prior=ALPHA_PRIOR, beta_prior=BETA_PRIOR):
    """Posterior Beta parameters from raw counts.

    ``tp`` is true positives; ``fail`` is false positives for precision, false
    negatives for recall.
    """
    return alpha_prior + tp, beta_prior + fail


def _seed_for(key, seed):
    """Deterministic seed from a key (a department code, say) and a global seed.

    The same (key, seed) pair always reproduces the same draw, whatever the
    order the keys are processed in. That matters when only a subset of units is
    corrected: the subset's numbers match the full run exactly.
    """
    return int(hashlib.md5(f"{seed}_{key}".encode()).hexdigest()[:8], 16)


def draw_correction_factor(alpha_precision, beta_precision, alpha_recall, beta_recall,
                            n_boot=DEFAULT_N_BOOT, seed=42, key=None):
    """Draw ``n_boot`` samples of the correction factor P/R.

    P and R are drawn independently from their respective Beta posteriors.

    The seed is deterministic: derived from ``key`` through ``_seed_for`` when
    one is given, typically the department code, otherwise the raw ``seed``.

    Returns an array of length ``n_boot``, NaN wherever R was drawn at zero.
    """
    rng = np.random.default_rng(_seed_for(key, seed) if key is not None else seed)
    p = rng.beta(alpha_precision, beta_precision, size=n_boot)
    r = rng.beta(alpha_recall, beta_recall, size=n_boot)
    return np.divide(p, r, out=np.full(n_boot, np.nan), where=r > 0)


def summarize_draws(draws, ci_levels=DEFAULT_CI_LEVELS):
    """Mean and credible intervals at several levels, from an existing set of draws.

    Returns ``{'mean': ..., 'ci95_low': ..., 'ci95_high': ..., 'ci99_low': ...,
    'ci99_high': ...}``, one low/high pair per level in ``ci_levels``, the number
    in the key being the level as a rounded percentage.
    """
    out = {'mean': float(np.nanmean(draws))}
    for ci in ci_levels:
        lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
        lo, hi = np.nanquantile(draws, [lo_q, hi_q])
        label = f"ci{round(ci * 100)}"
        out[f'{label}_low'] = float(lo)
        out[f'{label}_high'] = float(hi)
    return out


def bootstrap_recalibration(raw_values, alpha_precision, beta_precision, alpha_recall, beta_recall,
                             n_boot=DEFAULT_N_BOOT, seed=42, key=None, ci_levels=DEFAULT_CI_LEVELS,
                             return_draws=False):
    """Correct one or more raw quantities with the same set of factor draws.

    Installation counts and installed capacity for a given unit share a single
    correction factor rather than drawing independently, because they share the
    same detection uncertainty. This is the homogeneity assumption in practice:
    the correction is a uniform multiplicative factor, and it is not tested here.

    ``raw_values`` maps a name to a raw value, for instance
    ``{'n_inst': 1500, 'p_inst': 4200.0}``.

    Returns ``{name: {'raw': ..., 'mean': ..., 'ci95_low': ..., ...}}``. With
    ``return_draws=True`` it also carries ``'draws'`` and ``'factor_draws'``,
    which are what you need to plot the full distribution rather than its
    summary.
    """
    factor = draw_correction_factor(alpha_precision, beta_precision, alpha_recall, beta_recall,
                                     n_boot=n_boot, seed=seed, key=key)
    result = {}
    draws_out = {}
    for name, raw in raw_values.items():
        draws = raw * factor
        result[name] = summarize_draws(draws, ci_levels=ci_levels)
        result[name]['raw'] = raw
        if return_draws:
            draws_out[name] = draws

    if return_draws:
        result['draws'] = draws_out
        result['factor_draws'] = factor
    return result


def weighted_national_factor(alpha_precision, beta_precision, alpha_recall, beta_recall, weights,
                              n_boot=DEFAULT_N_BOOT, seed=42, key='national_weighted'):
    """One national correction factor, from departmental rates weighted by raw volume.

    This is a sanity check against the main pipeline, which corrects each unit
    and sums. It is not the estimator: the published numbers come from the
    per-unit route in ``recalibrate.py``.

    Why weight by raw volume rather than pool the national tp/fp/tp/fn counts,
    which is the other naive way to define a national precision and recall.
    Pooling the counts weights each department by the size of *its annotation
    sample*, a quantity with no relation to that department's share of the
    thing being corrected. Two departments can carry the same number of
    annotated points and very different fleets: a dense department weighs more
    in the national total than a rural one, however many points were annotated
    in each. Pooling by annotation ignores that asymmetry; weighting by raw
    volume respects it. Empirically the volume-weighted version departs from the
    per-unit sum by about 1%, against 12 to 14% for the annotation-pooled one.

    It remains an approximation, not an exact rewriting of the per-unit sum. The
    weighted mean of P over the weighted mean of R is not the weighted mean of
    P/R (Jensen's inequality). Only the weighted mean of the ratio would
    reproduce the per-unit sum exactly, and it would then provide no independent
    check at all, being that sum rewritten.

    The four Beta parameters and ``weights`` are 1-D arrays, one element per
    department, on the same Jeffreys convention as ``draw_correction_factor``.
    That convention is always defined, unlike a raw ``Beta(tp, fp)``, which
    fails as soon as a department records zero false positives or zero false
    negatives.

    Returns an array of length ``n_boot``.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n_dept = len(weights)

    rng = np.random.default_rng(_seed_for(key, seed) if key is not None else seed)
    precision_draws = rng.beta(alpha_precision, beta_precision, size=(n_boot, n_dept))
    recall_draws = rng.beta(alpha_recall, beta_recall, size=(n_boot, n_dept))

    national_precision = precision_draws @ weights
    national_recall = recall_draws @ weights
    return np.divide(national_precision, national_recall,
                     out=np.full(n_boot, np.nan), where=national_recall > 0)
