"""
Shared functions for score_and_aggregation.ipynb, which keeps that notebook readable.

Three families:

- aggregation by reporting unit, turning the annotation point files into a table
  of counts and their Beta calibration
- frequentist calibration (Wilson), as a companion to the Beta
- visualisation (cartes precision/rappel/F1, carte + scatter d'incertitude relative)

Beta convention: a Jeffreys prior, Beta(0.5, 0.5). The standard choice for
calibrating a proportion (Brown, Cai and DasGupta 2001), with good coverage near
the 0 and 1 boundaries and on small samples, both of which occur here.

Wilson convention: the standard frequentist interval for a binomial proportion,
with no prior. It covers better than the naive Wald interval at small n or with p
near the boundaries, which is common here on thinly sampled units. It is kept
alongside the Beta because it reads as a margin of error, the way a poll does:
precision is 85% plus or minus 5 points, rather than a credible width relative to
a posterior mean.
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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.stats import norm

# Shared figure style, found by walking up from *this* file rather than from the
# caller's working directory, so it resolves whatever notebook imports helpers.
# Applied on import: `import helpers` is enough to inherit the rcParams.
def _find_style(start, marker='paths.py'):
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"{marker} introuvable en remontant depuis {start}")

sys.path.insert(0, str(_find_style(Path(__file__).resolve().parent)))
import style
style.apply()

ALPHA_PRIOR = 0.5
BETA_PRIOR = 0.5
DEFAULT_CI = 0.95

# Offline basemap. This pipeline has no network access to tile servers, so
# OSM/CartoDB bloques depuis l'environnement d'execution) -- world_lowres.geojson (Natural Earth
# a static land/sea silhouette is used instead (Natural Earth 1:110m, same CRS as
# the departmental boundaries). Enough at the national and departmental scale
# this work is drawn at; it carries no sub-departmental detail.
WORLD_LOWRES_PATH = Path(__file__).parent / paths.WORLD


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_precision_points(path):
    """precision_points.csv : array_id, lat, lon, dpt, pred (1 = vrai positif, 0 = faux positif).
    The unconditional zfill(2) is a no-op on a code that already has two
    characters, and only fixes '5' into '05'. It cannot over-pad, and the Corsican
    codes are unaffected."""
    df = pd.read_csv(path, dtype={'dpt': str})
    df['dpt'] = df['dpt'].str.zfill(2)
    return df


def load_recall_points(path):
    """Recall points: 1 means the installation was found, 0 means it was missed."""
    df = pd.read_csv(path, dtype={'dpt': str})
    df['dpt'] = df['dpt'].str.zfill(2)
    return df


# ---------------------------------------------------------------------------
# Aggregation by reporting unit
# ---------------------------------------------------------------------------

def beta_posterior(n_success, n_fail, alpha_prior=ALPHA_PRIOR, beta_prior=BETA_PRIOR, ci=DEFAULT_CI):
    """Posterieure Beta(alpha_prior + succes, beta_prior + echecs) : renvoie
    (alpha, beta, moyenne, ci_low, ci_high, incertitude_relative)."""
    a = alpha_prior + n_success
    b = beta_prior + n_fail
    mean = a / (a + b)
    lo, hi = beta_dist.ppf([(1 - ci) / 2, 1 - (1 - ci) / 2], a, b)
    rel_uncertainty = (hi - lo) / mean if mean > 0 else np.nan
    return a, b, mean, lo, hi, rel_uncertainty


def dept_counts(df, dept_col='dpt', pred_col='pred'):
    """Raw counts per unit: n, n_positive, n_negative.

    When a ``weight`` column is present the counts are weighted, n being the sum
    of the weights. This is the channel through which the ground-truth validity
    correction is applied: false negatives that were not re-reviewed carry a
    weight of one minus the measured false-FN rate, true positives carry one.

    The resulting counts are not integers, and that is deliberate. Beta
    posteriors accept fractional pseudo-counts without complaint, and rounding
    them would silently discard the correction.
    """
    if 'weight' in df.columns:
        tmp = df.assign(_w=df['weight'], _pw=df[pred_col] * df['weight'])
        g = tmp.groupby(dept_col).agg(n=('_w', 'sum'), n_positive=('_pw', 'sum')).reset_index()
    else:
        g = df.groupby(dept_col).agg(n=(pred_col, 'size'), n_positive=(pred_col, 'sum')).reset_index()
    g['n_negative'] = g['n'] - g['n_positive']
    return g


def fit_beta_prior(success, fail):
    """Fit an empirical-Bayes Beta prior across units, by the method of moments.

    The expected binomial sampling noise is subtracted from the observed
    variance, so only genuine between-unit variation ends up in the prior. This
    is standard partial pooling, a hierarchical Beta-binomial without the MCMC:
    each unit's posterior becomes Beta(a + tp, b + fail), which amounts to
    kappa = a + b virtual observations at the national mean. Thinly sampled units
    are pulled toward that mean; well sampled ones barely move. The Jeffreys
    baseline is the kappa -> 0 limit.

    Why exchangeability is defensible here: precision and recall are properties
    of the *instrument*, the imagery and the building stock, and are a priori
    independent of the registry completeness this work sets out to measure.
    Pooling therefore stabilises the measurement without touching the signal.
    That argument is what licenses the prior, and it is worth restating whenever
    the method is reused somewhere the two are not independent.

    ``success`` and ``fail`` are aligned series of per-unit counts.
    Renvoie (a, b, diagnostics_dict)."""
    mask = success.notna() & fail.notna()
    s, f = success[mask].astype(float), fail[mask].astype(float)
    n = s + f
    r = s / n
    m = float(np.average(r, weights=n))
    var_obs = float(np.average((r - m) ** 2, weights=n))
    noise = float(np.average(r * (1 - r) / n, weights=n))
    var_true = max(var_obs - noise, 1e-6)
    kappa = float(np.clip(m * (1 - m) / var_true - 1, 2.0, 1000.0))
    diag = {'mean': m, 'sd_true': float(np.sqrt(var_true)), 'kappa': kappa, 'n_depts': int(mask.sum())}
    return m * kappa, (1 - m) * kappa, diag


def build_scores_table(precision_df, recall_df, alpha_prior=ALPHA_PRIOR, beta_prior=BETA_PRIOR,
                        ci=DEFAULT_CI, prior='jeffreys'):
    """Per-unit table: precision and recall counts, point estimates, and Beta calibration.

    Carries alpha, beta, the posterior mean, the interval and the relative
    uncertainty for each metric.

    `prior` :
      - 'jeffreys' (baseline): each unit estimated on its own.
      - 'eb': a prior fitted across all units, separately for precision and
        recall.

    Everything downstream (F1, the correction factor, the bootstrap in
    recalibrate.py) inherits the choice through the alpha_* and beta_* columns of
    this table. Nothing else needs changing, which is what keeps the prior
    consistent across the whole chain.
    """
    prec = dept_counts(precision_df).rename(columns={
        'n': 'n_precision', 'n_positive': 'tp_precision', 'n_negative': 'fp_precision',
    })
    prec['precision'] = prec['tp_precision'] / prec['n_precision']

    rec = dept_counts(recall_df).rename(columns={
        'n': 'n_recall', 'n_positive': 'tp_recall', 'n_negative': 'fn_recall',
    })
    rec['recall'] = rec['tp_recall'] / rec['n_recall']

    table = prec.merge(rec, on='dpt', how='outer')
    table['f1'] = 2 * table['precision'] * table['recall'] / (table['precision'] + table['recall'])

    if prior == 'eb':
        a_p, b_p, diag_p = fit_beta_prior(table['tp_precision'], table['fp_precision'])
        a_r, b_r, diag_r = fit_beta_prior(table['tp_recall'], table['fn_recall'])
        print(f"prior EB precision : Beta({a_p:.1f}, {b_p:.1f}) -- mean={diag_p['mean']:.3f}, "
              f"sd_true={diag_p['sd_true']:.3f}, kappa={diag_p['kappa']:.0f}")
        print(f"prior EB recall    : Beta({a_r:.1f}, {b_r:.1f}) -- mean={diag_r['mean']:.3f}, "
              f"sd_true={diag_r['sd_true']:.3f}, kappa={diag_r['kappa']:.0f}")
        priors_p, priors_r = (a_p, b_p), (a_r, b_r)
    elif prior == 'jeffreys':
        priors_p, priors_r = (alpha_prior, beta_prior), (alpha_prior, beta_prior)
    else:
        raise ValueError(f"unknown prior: {prior!r} (expected 'jeffreys' or 'eb')")

    # calibration Beta -- precision
    beta_cols_p = table.apply(
        lambda r: beta_posterior(r['tp_precision'], r['fp_precision'], priors_p[0], priors_p[1], ci)
        if pd.notna(r.get('n_precision')) else (np.nan,) * 6,
        axis=1, result_type='expand',
    )
    beta_cols_p.columns = ['alpha_precision', 'beta_precision', 'precision_mean',
                            'precision_ci_low', 'precision_ci_high', 'precision_rel_uncertainty']
    table = pd.concat([table, beta_cols_p], axis=1)

    # calibration Beta -- recall
    beta_cols_r = table.apply(
        lambda r: beta_posterior(r['tp_recall'], r['fn_recall'], priors_r[0], priors_r[1], ci)
        if pd.notna(r.get('n_recall')) else (np.nan,) * 6,
        axis=1, result_type='expand',
    )
    beta_cols_r.columns = ['alpha_recall', 'beta_recall', 'recall_mean',
                            'recall_ci_low', 'recall_ci_high', 'recall_rel_uncertainty']
    table = pd.concat([table, beta_cols_r], axis=1)

    return table


def add_f1_uncertainty(table, n_mc=20000, seed=42, ci=DEFAULT_CI):
    """F1 has no closed Beta form, being a non-linear combination of two
    independent proportions, so its distribution comes from Monte-Carlo draws of
    the two posteriors. Adds the f1_* columns.
    """
    rng = np.random.default_rng(seed)
    has_both = table.dropna(subset=['alpha_precision', 'beta_precision', 'alpha_recall', 'beta_recall'])

    rows = []
    for _, row in has_both.iterrows():
        p_samples = rng.beta(row['alpha_precision'], row['beta_precision'], size=n_mc)
        r_samples = rng.beta(row['alpha_recall'], row['beta_recall'], size=n_mc)
        denom = p_samples + r_samples
        f1_samples = np.divide(2 * p_samples * r_samples, denom, out=np.zeros(n_mc), where=denom > 0)
        rows.append({
            'dpt': row['dpt'],
            'f1_mean': f1_samples.mean(),
            'f1_ci_low': np.quantile(f1_samples, (1 - ci) / 2),
            'f1_ci_high': np.quantile(f1_samples, 1 - (1 - ci) / 2),
        })

    f1_df = pd.DataFrame(rows)
    if len(f1_df):
        f1_df['f1_rel_uncertainty'] = (f1_df['f1_ci_high'] - f1_df['f1_ci_low']) / f1_df['f1_mean']

    return table.merge(f1_df, on='dpt', how='left')


def national_summary(table, alpha_prior=ALPHA_PRIOR, beta_prior=BETA_PRIOR, ci=DEFAULT_CI):
    """National summary: counts summed across units, then one Beta on the total.

    That order matters. Averaging the per-unit estimates instead biases the
    national figure as soon as sample sizes differ, and they differ a lot here: a
    unit with 500 annotations should not weigh the same as one with 100.

    The mean and median of the per-unit values are reported alongside, as
    descriptive summaries rather than as the national estimator. The median is
    the robust one to read when a few units behave atypically.
    """
    rows = []
    for metric, tp_col, fail_col, n_col in [
        ('precision', 'tp_precision', 'fp_precision', 'n_precision'),
        ('recall', 'tp_recall', 'fn_recall', 'n_recall'),
    ]:
        sub = table.dropna(subset=[n_col])
        tp_total, fail_total = sub[tp_col].sum(), sub[fail_col].sum()
        _, _, mean, lo, hi, rel_unc = beta_posterior(tp_total, fail_total, alpha_prior, beta_prior, ci)
        rows.append({
            'metric': metric,
            'pooled_mean': mean, 'pooled_ci_low': lo, 'pooled_ci_high': hi,
            'pooled_rel_uncertainty': rel_unc,
            'dept_mean': sub[metric].mean(), 'dept_median': sub[metric].median(),
            'n_depts': len(sub), 'n_total': int(sub[n_col].sum()),
        })
    return pd.DataFrame(rows)


def add_capacity_correction_uncertainty(table, n_mc=20000, seed=42, ci=DEFAULT_CI):
    """Propagate the precision and recall uncertainty to a corrected quantity.

    The correction is ``corrected = raw * precision / recall``: multiplying by a
    precision below one removes the false positives, dividing by a recall below
    one scales up to compensate for the real installations that were missed.

    This assumes a **uniform multiplicative correction**: one factor applied to
    the whole raw quantity, independent of installation size. That is the
    homogeneity assumption, and it is exactly what the size-weighted
    specification exists to bound. Check it against whatever correction is
    actually applied downstream if the two ever diverge.

    The raw quantity is a known constant here, not a random variable, so it
    cancels out of the *relative* uncertainty, which is all this computes. The
    correction factor carries the whole relative uncertainty of the corrected
    quantity, and the result therefore applies unchanged to any raw quantity,
    capacity or count, whatever its value.
    absolue : incertitude_relative(grandeur_corrigee) == incertitude_relative(precision/recall).

    Monte-Carlo rather than the delta method: exact up to sampling noise, with no
    first-order approximation. That matters here because the variance of P/R is
    harder to approximate than that of F1 when recall approaches zero.

    Ajoute correction_factor_mean, correction_factor_ci_low/high, correction_factor_rel_uncertainty."""
    rng = np.random.default_rng(seed)
    has_both = table.dropna(subset=['alpha_precision', 'beta_precision', 'alpha_recall', 'beta_recall'])

    rows = []
    for _, row in has_both.iterrows():
        p_samples = rng.beta(row['alpha_precision'], row['beta_precision'], size=n_mc)
        r_samples = rng.beta(row['alpha_recall'], row['beta_recall'], size=n_mc)
        factor_samples = np.divide(p_samples, r_samples, out=np.full(n_mc, np.nan), where=r_samples > 0)
        rows.append({
            'dpt': row['dpt'],
            'correction_factor_mean': np.nanmean(factor_samples),
            'correction_factor_ci_low': np.nanquantile(factor_samples, (1 - ci) / 2),
            'correction_factor_ci_high': np.nanquantile(factor_samples, 1 - (1 - ci) / 2),
        })

    factor_df = pd.DataFrame(rows)
    if len(factor_df):
        factor_df['correction_factor_rel_uncertainty'] = (
            (factor_df['correction_factor_ci_high'] - factor_df['correction_factor_ci_low'])
            / factor_df['correction_factor_mean']
        )

    return table.merge(factor_df, on='dpt', how='left')


def samples_needed_for_target(table, target_rel_uncertainty, dpts=None,
                               n_grid=(0, 25, 50, 100, 200, 400, 800, 1600, 3200),
                               n_mc=5000, seed=42, ci=DEFAULT_CI):
    """Where to spend the next annotation, unit by unit.

    Grid-searches how many *additional* samples are needed to bring the
    correction factor's relative uncertainty under the target: first along
    precision alone, then along recall alone. The cheaper axis is the one to
    reinforce first for that unit.

    There is a ceiling, and it is the useful part of this function. Reinforcing
    one axis can never push the factor's uncertainty below the *other* axis's own
    uncertainty at its current n. Infinitely many precision samples still leave
    the total bounded by recall, if recall is not improved too. When neither axis
    alone reaches the target, ``n_both_add`` gives the number to add to both at
    once, which is then the only option.

    The projection assumes added samples succeed at the current rate, so only the
    interval narrows with n. That is optimistic, since a real campaign can reveal
    a different rate, but it is the standard assumption for sample planning and
    it is stated rather than hidden.

    ``dpts`` defaults to every unit with a computed correction factor. The table
    comes back sorted by current uncertainty, worst first. A NaN in an ``n_*_add``
    column means ``n_grid`` did not reach far enough to hit the target; extend it.
    """
    rng = np.random.default_rng(seed)
    sub = table if dpts is None else table[table['dpt'].isin(dpts)]
    sub = sub.dropna(subset=['alpha_precision', 'beta_precision', 'alpha_recall', 'beta_recall',
                              'correction_factor_rel_uncertainty'])

    def factor_rel_unc(a_p, b_p, a_r, b_r):
        p_samples = rng.beta(a_p, b_p, size=n_mc)
        r_samples = rng.beta(a_r, b_r, size=n_mc)
        factor = np.divide(p_samples, r_samples, out=np.full(n_mc, np.nan), where=r_samples > 0)
        mean = np.nanmean(factor)
        if not (mean > 0):
            return np.nan
        lo = np.nanquantile(factor, (1 - ci) / 2)
        hi = np.nanquantile(factor, 1 - (1 - ci) / 2)
        return (hi - lo) / mean

    def search_axis(n_current, p_hat, alpha_fixed, beta_fixed, precision_axis):
        for add in n_grid:
            n_new = n_current + add
            tp_new, fail_new = n_new * p_hat, n_new * (1 - p_hat)
            a, b = ALPHA_PRIOR + tp_new, BETA_PRIOR + fail_new
            unc = (factor_rel_unc(a, b, alpha_fixed, beta_fixed) if precision_axis
                   else factor_rel_unc(alpha_fixed, beta_fixed, a, b))
            if pd.notna(unc) and unc <= target_rel_uncertainty:
                return add
        return np.nan

    def search_both(n_prec_current, p_hat, n_rec_current, r_hat):
        """The same increment added to both axes at once.

        The fallback for when neither axis alone ever suffices, each being
        capped by the other's own uncertainty.
        """
        for add in n_grid:
            n_p_new, n_r_new = n_prec_current + add, n_rec_current + add
            tp_p, fail_p = n_p_new * p_hat, n_p_new * (1 - p_hat)
            tp_r, fail_r = n_r_new * r_hat, n_r_new * (1 - r_hat)
            unc = factor_rel_unc(ALPHA_PRIOR + tp_p, BETA_PRIOR + fail_p,
                                  ALPHA_PRIOR + tp_r, BETA_PRIOR + fail_r)
            if pd.notna(unc) and unc <= target_rel_uncertainty:
                return add
        return np.nan

    rows = []
    for _, row in sub.iterrows():
        p_hat = row['tp_precision'] / row['n_precision']
        r_hat = row['tp_recall'] / row['n_recall']

        n_prec_add = search_axis(row['n_precision'], p_hat, row['alpha_recall'], row['beta_recall'], True)
        n_rec_add = search_axis(row['n_recall'], r_hat, row['alpha_precision'], row['beta_precision'], False)

        if pd.isna(n_prec_add) and pd.isna(n_rec_add):
            # Neither axis alone can reach the target: each is capped by the
            # other's uncertainty at its current n. Both must grow together.
            axe = 'neither alone (cross ceiling); increase both, see n_both_add'
            n_both_add = search_both(row['n_precision'], p_hat, row['n_recall'], r_hat)
        elif pd.isna(n_prec_add):
            axe, n_both_add = 'recall', np.nan
        elif pd.isna(n_rec_add):
            axe, n_both_add = 'precision', np.nan
        elif n_prec_add < n_rec_add:
            axe, n_both_add = 'precision', np.nan
        elif n_rec_add < n_prec_add:
            axe, n_both_add = 'recall', np.nan
        else:
            axe, n_both_add = 'egal', np.nan

        rows.append({
            'dpt': row['dpt'],
            'current_rel_uncertainty': row['correction_factor_rel_uncertainty'],
            'n_precision': row['n_precision'], 'n_recall': row['n_recall'],
            'n_precision_add': n_prec_add, 'n_recall_add': n_rec_add, 'n_both_add': n_both_add,
            'axe_prioritaire': axe,
        })

    return pd.DataFrame(rows).sort_values('current_rel_uncertainty', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Frequentist calibration (Wilson), alongside the Beta above
# ---------------------------------------------------------------------------

def wilson_interval(n_success, n_trials, ci=DEFAULT_CI):
    """Wilson confidence interval for a binomial proportion.

    Frequentist, so no prior, unlike ``beta_posterior``. Closed form, and better
    coverage than Wald at small n or with p near the boundaries. Returns
    (centre, ci_low, ci_high, relative uncertainty).
    """
    if pd.isna(n_trials) or n_trials == 0:
        return np.nan, np.nan, np.nan, np.nan
    p_hat = n_success / n_trials
    z = norm.ppf(1 - (1 - ci) / 2)
    denom = 1 + z**2 / n_trials
    center = (p_hat + z**2 / (2 * n_trials)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n_trials + z**2 / (4 * n_trials**2))) / denom
    lo, hi = max(center - margin, 0.0), min(center + margin, 1.0)
    rel_uncertainty = (hi - lo) / center if center > 0 else np.nan
    return center, lo, hi, rel_uncertainty


def add_frequentist_uncertainty(table, ci=DEFAULT_CI):
    """Add the frequentist Wilson precision and recall, alongside the Beta ones.
    Colonnes ajoutees : precision_wilson_*, recall_wilson_*.

    For F1, Wilson gives only an interval, not a distribution to sample from, so
    no Monte-Carlo is available. The uncertainty is propagated by the delta method
    place -- developpement de Taylor au premier ordre de F1 = 2PR/(P+R) autour de (P_hat, R_hat) :

        Var(F1) ~= (dF1/dP)^2 * Var(P) + (dF1/dR)^2 * Var(R),   P et R independants
        dF1/dP = 2*R^2 / (P+R)^2      dF1/dR = 2*P^2 / (P+R)^2
        Var(P) ~= P_hat*(1-P_hat) / n_precision   (and likewise for R)

    puis IC normal-approxime F1_hat +/- z*sqrt(Var(F1)). Approximation standard (delta method),
    which is less exact than the Beta Monte-Carlo at very small n, but consistent
    with the frequentist framing and much faster.
    """
    out = table.copy()

    prec_w = out.apply(
        lambda r: wilson_interval(r['tp_precision'], r['n_precision'], ci)
        if pd.notna(r.get('n_precision')) else (np.nan,) * 4,
        axis=1, result_type='expand',
    )
    prec_w.columns = ['precision_wilson_mean', 'precision_wilson_ci_low',
                       'precision_wilson_ci_high', 'precision_wilson_rel_uncertainty']
    out = pd.concat([out, prec_w], axis=1)

    rec_w = out.apply(
        lambda r: wilson_interval(r['tp_recall'], r['n_recall'], ci)
        if pd.notna(r.get('n_recall')) else (np.nan,) * 4,
        axis=1, result_type='expand',
    )
    rec_w.columns = ['recall_wilson_mean', 'recall_wilson_ci_low',
                      'recall_wilson_ci_high', 'recall_wilson_rel_uncertainty']
    out = pd.concat([out, rec_w], axis=1)

    z = norm.ppf(1 - (1 - ci) / 2)
    P, R = out['precision_wilson_mean'], out['recall_wilson_mean']
    var_p = P * (1 - P) / out['n_precision']
    var_r = R * (1 - R) / out['n_recall']
    denom = (P + R) ** 2
    df1_dp = 2 * R**2 / denom
    df1_dr = 2 * P**2 / denom
    se_f1 = np.sqrt(df1_dp**2 * var_p + df1_dr**2 * var_r)

    out['f1_wilson_mean'] = 2 * P * R / (P + R)
    out['f1_wilson_ci_low'] = (out['f1_wilson_mean'] - z * se_f1).clip(lower=0)
    out['f1_wilson_ci_high'] = (out['f1_wilson_mean'] + z * se_f1).clip(upper=1)
    out['f1_wilson_rel_uncertainty'] = (
        (out['f1_wilson_ci_high'] - out['f1_wilson_ci_low']) / out['f1_wilson_mean']
    )

    return out


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1',
    'precision_rel_uncertainty': 'Relative uncertainty -- precision',
    'recall_rel_uncertainty': 'Relative uncertainty -- recall',
    'f1_rel_uncertainty': 'Relative uncertainty -- F1',
    'correction_factor_mean': 'Correction factor (precision/recall)',
    'correction_factor_rel_uncertainty': 'Relative uncertainty -- correction factor',
}


def _dept_map(departements_gdf, table):
    return departements_gdf.merge(table, left_on='code', right_on='dpt', how='left')


def _hide_axis_chrome(ax):
    """Hide ticks, labels and spines without calling ``ax.set_axis_off()``.

    ``set_axis_off()`` also disables the background patch, which would erase the
    sea drawn by ``add_basemap``. Use this instead on any axis that has a
    basemap.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_basemap(ax, bounds, margin=0.08, land_color=style.BASEMAP_LAND, sea_color=style.BASEMAP_SEA,
                 edge_color=style.BASEMAP_EDGE):
    """Static land and sea basemap, with no tiles downloaded.

    The axis background is the sea, the country silhouettes are the land. Call it
    *before* plotting the data layer so it stays underneath.

    ``bounds`` is the extent of the layer to be drawn on top, used to frame the
    view with a margin rather than showing the whole world.
    """
    world = gpd.read_file(WORLD_LOWRES_PATH)
    ax.set_facecolor(sea_color)
    world.plot(ax=ax, color=land_color, edgecolor=edge_color, linewidth=0.4, zorder=0)

    xmin, ymin, xmax, ymax = bounds
    dx, dy = (xmax - xmin) * margin, (ymax - ymin) * margin
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)


def plot_metric_map(departements_gdf, table, column, ax=None, cmap=None, vmin=0, vmax=1, title=None):
    """One choropleth of ``column``.

    Creates its own figure when no ``ax`` is given, otherwise draws on the one
    provided, which is how ``plot_score_maps`` builds a row of panels.

    The default colormap is the higher-is-better sequential one. Never a
    red-to-green scale: it is unreadable for a colour-blind reader and the
    journal's figure guide rules it out. Pass the higher-is-worse colormap
    explicitly for a metric like relative uncertainty.
    """
    cmap = cmap or style.CMAP_SEQUENTIAL
    dept_map = _dept_map(departements_gdf, table)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 8))

    add_basemap(ax, departements_gdf.total_bounds)

    dept_map.plot(
        column=column, ax=ax, legend=True,
        legend_kwds={'label': title or METRIC_LABELS.get(column, column), 'orientation': 'vertical', 'shrink': 0.7},
        missing_kwds={'color': 'lightgrey', 'label': 'No data'},
        cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.2, zorder=1,
    )
    _hide_axis_chrome(ax)
    ax.set_title(title or METRIC_LABELS.get(column, column), fontsize=style.FONT_SIZE_PANEL_LABEL, fontweight='bold')

    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax


def plot_score_maps(departements_gdf, table, metrics=('precision', 'recall', 'f1'), mode='all', **kwargs):
    """``mode='all'`` gives one figure with one panel per metric.

    ``mode='solo'`` gives one independent figure per metric, which is what you
    want to export them separately. Returns a figure, or a list of figures.
    """
    if mode == 'all':
        fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 8))
        if len(metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, metrics):
            plot_metric_map(departements_gdf, table, m, ax=ax, **kwargs)
        plt.tight_layout()
        return fig

    if mode == 'solo':
        figs = []
        for m in metrics:
            fig, _ = plot_metric_map(departements_gdf, table, m, ax=None, **kwargs)
            figs.append(fig)
        return figs

    raise ValueError(f"unknown mode: {mode!r} (expected 'all' or 'solo')")


def plot_halfwidth_maps(departements_gdf, table, q=0.95, figsize=(15, 5.5), ci=DEFAULT_CI):
    """Three maps in a row: the interval half-width on precision, on recall, and
    on the correction factor. The ``*_pm`` columns are derived here, nothing needs
    preparing upstream.

    Two colour scales, not three, and the choice carries meaning.

    Panels (a) and (b) **share** a scale, running from zero to the quantile ``q``
    of the two columns pooled. They are two uncertainties on detection rates,
    directly comparable, and a common scale is the only way to see which of the
    two dominates in a given unit.
    (c) a sa PROPRE echelle et sa propre colormap (style.CMAP_SEQUENTIAL_BAD_ALT) : l'incertitude
    Panel (c) gets its own colormap, because the uncertainty on P/R is about a
    corrected capacity, not about a rate. Keeping the same colormap on a different
    scale would invite the reader to compare two things that are not comparable.

    Values above the quantile are clipped by the colorbar, not dropped from the
    computation. Without that, one or two thinly sampled units flatten the whole
    dynamic range of the other ninety.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    t = table.copy()
    for base in ('precision', 'recall', 'correction_factor'):
        t[f'{base}_pm'] = t[f'{base}_rel_uncertainty'] / 2

    vmax_rate = max(t['precision_pm'].quantile(q), t['recall_pm'].quantile(q))
    vmax_factor = t['correction_factor_pm'].quantile(q)

    panels = (
        ('a', 'precision_pm', style.CMAP_SEQUENTIAL_BAD, vmax_rate, 'Precision'),
        ('b', 'recall_pm', style.CMAP_SEQUENTIAL_BAD, vmax_rate, 'Recall'),
        ('c', 'correction_factor_pm', style.CMAP_SEQUENTIAL_BAD_ALT, vmax_factor,
         'Correction factor (P/R)'),
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    dept_map = _dept_map(departements_gdf, t)

    for ax, (letter, col, cmap, vmax, name) in zip(axes, panels):
        add_basemap(ax, departements_gdf.total_bounds)
        dept_map.plot(
            column=col, ax=ax, cmap=cmap, vmin=0, vmax=vmax, legend=False,
            missing_kwds={'color': 'lightgrey'},
            edgecolor='black', linewidth=0.2, zorder=1,
        )
        ax.set_title(f"{letter}   {name}", fontsize=style.FONT_SIZE_PANEL_LABEL,
                     fontweight='bold', loc='left')
        _hide_axis_chrome(ax)

    pct = int(round(ci * 100))

    def _cbar(target_axes, cmap, vmax, label, shrink):
        cb = fig.colorbar(
            ScalarMappable(norm=Normalize(0, vmax), cmap=cmap),
            ax=target_axes, orientation='horizontal',
            fraction=0.06, pad=0.02, shrink=shrink, extend='max',
        )
        cb.set_label(label, fontsize=style.FONT_SIZE_SMALL)
        # Ticks on multiples of five percentage points: the default locator
        # lands on 2%, 8%, 12% and reads badly.
        cb.ax.xaxis.set_major_locator(MultipleLocator(0.05))
        cb.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        cb.ax.tick_params(labelsize=style.FONT_SIZE_SMALL)
        return cb

    _cbar(list(axes[:2]), style.CMAP_SEQUENTIAL_BAD, vmax_rate,
          f"$\\pm$ half-width of the {pct}% CI on the detection rate", 0.6)
    _cbar(axes[2], style.CMAP_SEQUENTIAL_BAD_ALT, vmax_factor,
          f"$\\pm$ half-width of the {pct}% CI on the corrected capacity", 0.9)

    return fig, axes


def plot_uncertainty_scatter(table, mean_col, unc_col, xlabel=None, ylabel=None, title=None, ax=None):
    """A metric's mean against its relative uncertainty.

    Picks out the units that are both poorly performing and poorly measured, the
    ones to annotate next. Works for any pair of
    colonnes *_mean/*_rel_uncertainty de `table` (f1, correction_factor, ...)."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    sub = table.dropna(subset=[mean_col, unc_col])
    ax.scatter(sub[mean_col], sub[unc_col], alpha=0.7, color=style.COLOR_SCATTER_DEFAULT)
    for _, row in sub.iterrows():
        ax.annotate(row['dpt'], (row[mean_col], row[unc_col]), fontsize=7, alpha=0.6)

    ax.set_xlabel(xlabel or METRIC_LABELS.get(mean_col, mean_col))
    ax.set_ylabel(ylabel or METRIC_LABELS.get(unc_col, unc_col))
    ax.set_title(title or f"{METRIC_LABELS.get(mean_col, mean_col)} vs relative uncertainty, by department")

    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax


def plot_f1_vs_uncertainty(table, ax=None):
    """Backwards compatibility: F1 posterior mean against its relative uncertainty."""
    return plot_uncertainty_scatter(
        table, 'f1_mean', 'f1_rel_uncertainty',
        xlabel='F1 (posterior mean)',
        ylabel='Relative F1 uncertainty (95% CI width / mean)',
        title='F1 vs relative uncertainty, by department',
        ax=ax,
    )


def save_fig(fig, name, figs_dir, dpi=None, also_pdf=True):
    """Save a figure, creating the folder if needed.

    ``dpi=None`` uses the style's export resolution. ``also_pdf=True`` writes a
    vector PDF beside the PNG, under the same name, which is the format wanted at
    submission because the text stays editable (``style.apply()`` already sets
    pdf.fonttype=42)."""
    dpi = dpi or style.EXPORT_DPI
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if also_pdf and path.suffix != '.pdf':
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    return path


def highest_uncertainty(table, column='f1_rel_uncertainty', n=10, n_col=None):
    """The units with the highest uncertainty, the first candidates for more annotation.

    Pass ``n_col`` to show the sample size alongside, which is what tells you
    whether the uncertainty comes from a small n or from a genuinely hard unit.
    """
    cols = ['dpt', column] + ([n_col] if n_col else [])
    return table.dropna(subset=[column]).sort_values(column, ascending=False)[cols].head(n)


# ---------------------------------------------------------------------------
# Part 3: descriptive statistics. Point counts, a per-unit illustration, and the
# comparison between recall ground-truth sources.
# ---------------------------------------------------------------------------

def points_summary(table):
    """Descriptive summary of the annotation effort, per unit.

    Precision and recall are summarised separately: they are two distinct
    campaigns with different sample sizes, and conflating them hides which of the
    two is the binding constraint. One row per metric, with the total, the number
    of units covered, and the min, median, max and mean per unit.
    """
    rows = []
    for metric, n_col in [('precision', 'n_precision'), ('recall', 'n_recall')]:
        sub = table.dropna(subset=[n_col])
        n = sub[n_col]
        rows.append({
            'metric': metric,
            'n_total': int(n.sum()),
            'n_depts': len(sub),
            'n_min': int(n.min()) if len(n) else 0,
            'n_median': n.median() if len(n) else float('nan'),
            'n_mean': n.mean() if len(n) else float('nan'),
            'n_max': int(n.max()) if len(n) else 0,
        })
    return pd.DataFrame(rows)


def plot_department_points(dept, precision_points, recall_points, departements_gdf, figsize=(14, 7)):
    """Illustrative map of a single unit: precision points beside recall points.

    Precision points are the model's detections, coloured true or false positive;
    recall points are the ground truth, coloured found or missed. Only meaningful
    at this scale. Nationally the point density resolves into nothing readable,
    which is what the aggregated choropleths are for.

    `precision_points`/`recall_points` : DataFrames charges via load_precision_points /
    ``departements_gdf`` supplies the outline.
    """
    dept = str(dept).zfill(2) if len(str(dept)) == 1 else str(dept)
    dept_geom = departements_gdf[departements_gdf['code'] == dept]
    if dept_geom.empty:
        raise ValueError(f"unit {dept!r} not found in departements_gdf")

    sub_p = precision_points[precision_points['dpt'] == dept]
    sub_r = recall_points[recall_points['dpt'] == dept]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    bounds = dept_geom.total_bounds

    add_basemap(axes[0], bounds, margin=0.15)
    dept_geom.boundary.plot(ax=axes[0], color='black', linewidth=1, zorder=1)
    for val, color, label in [(1, style.COLOR_TP, 'TP'), (0, style.COLOR_FP, 'FP')]:
        s = sub_p[sub_p['pred'] == val]
        axes[0].scatter(s['lon'], s['lat'], c=color, s=18, label=f'{label} (n={len(s)})', alpha=0.75, zorder=2)
    axes[0].set_title(f"Precision -- department {dept} (n={len(sub_p)})")
    axes[0].legend(loc='best', fontsize=style.FONT_SIZE_SMALL)
    _hide_axis_chrome(axes[0])

    add_basemap(axes[1], bounds, margin=0.15)
    dept_geom.boundary.plot(ax=axes[1], color='black', linewidth=1, zorder=1)
    for val, color, label in [(1, style.COLOR_TP, 'TP (detected)'), (0, style.COLOR_FN, 'FN (missed)')]:
        s = sub_r[sub_r['pred'] == val]
        axes[1].scatter(s['lon'], s['lat'], c=color, s=18, label=f'{label} (n={len(s)})', alpha=0.75, zorder=2)
    axes[1].set_title(f"Recall -- department {dept} (n={len(sub_r)})")
    axes[1].legend(loc='best', fontsize=style.FONT_SIZE_SMALL)
    _hide_axis_chrome(axes[1])

    plt.tight_layout()
    return fig, axes


def plot_region_points(
    region,
    precision_points,
    recall_points,
    departements_gdf,
    figsize=(16, 11),
    max_cities=14,
):
    """
    Detailed map of one region: the precision and recall points of every unit in
    it, on a single canvas.

    Encoding:
        - couleur = type d'evaluation
            * precision
            * recall
        - forme = statut
            * cercle = vrai positif (TP)
            * cross = error (false positive, or missed installation)

    The basemap is the paper's static one, with more detailed Natural Earth
    layers added on top, suited to the regional scale:
        - frontieres nationales
        - limites regionales
        - the region's units: a thick outline for the region, obtained by
          dissolving them, and neighbouring units in light grey for context
        - cours d'eau
        - lacs
        - populated places, at most ``max_cities`` of them

    `precision_points` / `recall_points` :
        DataFrames with columns dpt, lat, lon, pred.

    `departements_gdf` :
        GeoDataFrame of the units, carrying a ``code`` column.
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    # ------------------------------------------------------------------
    # Region to unit mapping
    # ------------------------------------------------------------------
    region_departements = {
        "Auvergne-Rhone-Alpes": [
            "01", "03", "07", "15", "26", "38",
            "42", "43", "63", "69", "73", "74"
        ],
        "Bourgogne-Franche-Comte": [
            "21", "25", "39", "58", "70", "71", "89", "90"
        ],
        "Bretagne": [
            "22", "29", "35", "56"
        ],
        "Centre-Val de Loire": [
            "18", "28", "36", "37", "41", "45"
        ],
        "Corse": [
            "2A", "2B"
        ],
        "Grand Est": [
            "08", "10", "51", "52", "54", "55", "57",
            "67", "68", "88"
        ],
        "Hauts-de-France": [
            "02", "59", "60", "62", "80"
        ],
        "Ile-de-France": [
            "75", "77", "78", "91", "92", "93", "94", "95"
        ],
        "Normandie": [
            "14", "27", "50", "61", "76"
        ],
        "Nouvelle-Aquitaine": [
            "16", "17", "19", "23", "24", "33", "40",
            "47", "64", "79", "86", "87"
        ],
        "Occitanie": [
            "09", "11", "12", "30", "31", "32", "34",
            "46", "48", "65", "66", "81", "82"
        ],
        "Pays de la Loire": [
            "44", "49", "53", "72", "85"
        ],
        "Provence-Alpes-Cote d'Azur": [
            "04", "05", "06", "13", "83", "84"
        ],
    }

    if region not in region_departements:
        raise ValueError(
            f"Region {region!r} introuvable. "
            f"Regions disponibles : {list(region_departements)}"
        )

    region_dpts = region_departements[region]

    # ------------------------------------------------------------------
    # Department geometries
    # ------------------------------------------------------------------
    dept_codes = departements_gdf["code"].astype(str)

    region_geom = departements_gdf[
        dept_codes.isin(region_dpts)
    ].copy()

    if region_geom.empty:
        raise ValueError(
            f"No geometry found for units "
            f"de {region!r}"
        )

    # Make sure geometries are in lon/lat for the basemap and Cartopy
    if region_geom.crs is not None and region_geom.crs.to_epsg() != 4326:
        region_geom = region_geom.to_crs(4326)

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------
    def normalize_dpt(x):
        x = str(x)
        return x if x in ["2A", "2B"] else x.zfill(2)

    precision_dpt = precision_points["dpt"].apply(normalize_dpt)
    recall_dpt = recall_points["dpt"].apply(normalize_dpt)

    sub_p = precision_points[
        precision_dpt.isin(region_dpts)
    ].copy()

    sub_r = recall_points[
        recall_dpt.isin(region_dpts)
    ].copy()

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(),
    )

    # ------------------------------------------------------------------
    # Extent
    # ------------------------------------------------------------------
    bounds = region_geom.total_bounds

    xmin, ymin, xmax, ymax = bounds

    dx = (xmax - xmin) * 0.08
    dy = (ymax - ymin) * 0.08

    ax.set_extent(
        [
            xmin - dx,
            xmax + dx,
            ymin - dy,
            ymax + dy,
        ],
        crs=ccrs.PlateCarree(),
    )

    # ------------------------------------------------------------------
    # Existing paper basemap
    # ------------------------------------------------------------------
    # `add_basemap` provides the same land / sea background as the
    # other figures in the paper.
    add_basemap(
        ax,
        bounds,
        margin=0.08,
    )

    # ------------------------------------------------------------------
    # Higher-resolution contextual layers
    # ------------------------------------------------------------------

    # Oceans / land are already provided by add_basemap().
    # The following layers provide additional regional context.

    def _ne_path(category, name, resolution="10m"):
        """Local path of a Natural Earth layer, or None if unavailable.

        Cartopy downloads these lazily, so without this pre-fetch an offline
        machine would not fail until render time, inside savefig, which is a
        confusing place to discover a network problem. Resolve the path first
        and skip the layer if unreachable: these layers are decorative, and the
        paper's own data is local.
        """
        try:
            return shpreader.natural_earth(
                resolution=resolution,
                category=category,
                name=name,
            )
        except Exception:
            return None

    # Lakes
    if _ne_path("physical", "lakes") is not None:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "lakes",
                "10m",
                facecolor="none",
                edgecolor="#7f9db5",
                linewidth=0.6,
            ),
            zorder=1,
        )

    # Rivers
    if _ne_path("physical", "rivers_lake_centerlines") is not None:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "rivers_lake_centerlines",
                "10m",
                facecolor="none",
                edgecolor="#7f9db5",
                linewidth=0.55,
            ),
            zorder=1,
        )

    # Country boundaries
    if _ne_path("cultural", "admin_0_boundary_lines_land") is not None:
        ax.add_feature(
            cfeature.BORDERS.with_scale("10m"),
            edgecolor="#888888",
            linewidth=0.7,
            zorder=2,
        )

    # ------------------------------------------------------------------
    # Neighbouring departments (context)
    # ------------------------------------------------------------------
    # Departments outside the region but inside the frame, in light grey:
    # the region then reads as a cut-out of the national maps.
    neighbours = departements_gdf[~dept_codes.isin(region_dpts)].copy()

    if neighbours.crs is not None and neighbours.crs.to_epsg() != 4326:
        neighbours = neighbours.to_crs(4326)

    neighbours.plot(
        ax=ax,
        facecolor="#e3ded3",
        edgecolor="#c9c2b4",
        linewidth=0.4,
        zorder=1,
    )

    # ------------------------------------------------------------------
    # Region and department boundaries
    # ------------------------------------------------------------------
    # The regional outline is the dissolved union of its departments (IGN
    # geometries, cf. src/departements.geojson) rather than a Natural Earth
    # admin-1 layer. Same source as every other map in the paper, and no
    # dependency on cartopy's shapefile reader -- shpreader.Reader() returns
    # FionaRecord objects when Fiona is installed, and those are incompatible
    # with cartopy's own Record.geometry property (AttributeError: '_shape').
    # The unit geometries are simplified, so the internal borders of two
    # neighbours do not coincide exactly. A plain union leaves around 200
    # lens-shaped holes along them, every one of which renders as a spurious
    # thick outline. A 500 m open-close in Lambert-93 closes them, changing the
    # area by less than 0.1% and leaving the outer boundary untouched.
    _CLOSE_M = 500

    outline = region_geom.to_crs(2154).geometry
    outline = outline.buffer(_CLOSE_M).union_all().buffer(-_CLOSE_M)
    outline = gpd.GeoSeries([outline], crs=2154).to_crs(4326)

    outline.plot(
        ax=ax,
        facecolor="white",
        alpha=0.45,
        edgecolor="none",
        zorder=2,
    )

    region_geom.boundary.plot(
        ax=ax,
        color="#8a8a8a",
        linewidth=0.7,
        zorder=3,
    )

    outline.boundary.plot(
        ax=ax,
        color="#1a1a1a",
        linewidth=1.8,
        zorder=4,
    )

    # Unit codes, drawn above the points with a white stroke. Placed underneath
    # they vanish in dense areas and the reader takes them for missing labels.
    for _, dpt_row in region_geom.iterrows():
        label_point = dpt_row.geometry.representative_point()

        ax.text(
            label_point.x,
            label_point.y,
            str(dpt_row["code"]),
            fontsize=15,
            fontweight="bold",
            color="#6a6a6a",
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
            zorder=12,
            path_effects=[
                pe.withStroke(linewidth=3.5, foreground="white"),
            ],
        )

    # ------------------------------------------------------------------
    # Cities / populated places
    # ------------------------------------------------------------------
    populated_places_shp = _ne_path("cultural", "populated_places")

    places = (
        gpd.read_file(populated_places_shp)
        if populated_places_shp is not None
        else gpd.GeoDataFrame(geometry=[], crs=4326)
    )

    # Keep only places inside / close to the regional extent
    places = places.to_crs(4326)

    region_union = outline.union_all()

    places = places[
        places.geometry.intersects(region_union)
    ].copy()

    # Prefer larger settlements.
    # Natural Earth provides `SCALERANK`: lower values = larger places.
    # Few labels: the map is about the evaluation points, cities are only
    # there to let the reader situate them.
    if "SCALERANK" in places.columns:
        places = places.sort_values("SCALERANK").head(max_cities)
    else:
        places = places.head(max_cities)

    # Point symbols for cities
    ax.scatter(
        places.geometry.x,
        places.geometry.y,
        s=16,
        color="#3a3a3a",
        marker="o",
        alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # City labels
    name_col = None

    for candidate in ["NAME", "name", "NAMEASCII"]:
        if candidate in places.columns:
            name_col = candidate
            break

    if name_col is not None:
        for _, city in places.iterrows():
            ax.text(
                city.geometry.x + 0.04,
                city.geometry.y + 0.02,
                str(city[name_col]),
                fontsize=style.FONT_SIZE_SMALL,
                color="#1a1a1a",
                ha="left",
                va="bottom",
                transform=ccrs.PlateCarree(),
                zorder=6,
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground="white"),
                ],
            )

    # ------------------------------------------------------------------
    # Colors = evaluation type
    # Shapes = status
    # ------------------------------------------------------------------
    # Precision and recall have distinct colors.
    # TP = filled circle
    # Error = x
    precision_color = style.COLOR_TP
    recall_color = style.COLOR_FN

    # ------------------------------------------------------------------
    # Precision points
    # ------------------------------------------------------------------
    # TP
    s = sub_p[sub_p["pred"] == 1]

    if len(s) > 0:
        ax.scatter(
            s["lon"],
            s["lat"],
            c=precision_color,
            s=34,
            marker="o",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
            transform=ccrs.PlateCarree(),
            zorder=8,
        )

    # FP
    s = sub_p[sub_p["pred"] == 0]

    if len(s) > 0:
        ax.scatter(
            s["lon"],
            s["lat"],
            c=precision_color,
            s=45,
            marker="x",
            alpha=0.85,
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
            zorder=9,
        )

    # ------------------------------------------------------------------
    # Recall points
    # ------------------------------------------------------------------
    # TP
    s = sub_r[sub_r["pred"] == 1]

    if len(s) > 0:
        ax.scatter(
            s["lon"],
            s["lat"],
            c=recall_color,
            s=40,
            marker="o",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

    # FN
    s = sub_r[sub_r["pred"] == 0]

    if len(s) > 0:
        ax.scatter(
            s["lon"],
            s["lat"],
            c=recall_color,
            s=50,
            marker="x",
            alpha=0.85,
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
            zorder=11,
        )

    # ------------------------------------------------------------------
    # Legends
    # ------------------------------------------------------------------
    # First legend: what does the color mean?
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=precision_color,
            markeredgecolor="white",
            markersize=8,
            label=f"Precision (n={len(sub_p):,})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=recall_color,
            markeredgecolor="white",
            markersize=8,
            label=f"Recall (n={len(sub_r):,})",
        ),
    ]

    legend_color = ax.legend(
        handles=color_handles,
        loc="lower left",
        fontsize=11,
        frameon=True,
        framealpha=0.9,
        title="Evaluation",
        title_fontsize=12,
    )

    ax.add_artist(legend_color)

    # Second legend: what does the shape mean?
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=8,
            label="True positive",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="#555555",
            markersize=9,
            markeredgewidth=1.5,
            label="False positive / negative",
        ),
    ]

    ax.legend(
        handles=shape_handles,
        loc="lower right",
        fontsize=11,
        frameon=True,
        framealpha=0.9,
        title="Status",
        title_fontsize=12,
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.set_title(
        f"{region} — annotated points for precision and recall",
        fontsize=style.FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        pad=12,
    )

    ax.text(
        0.5,
        -0.02,
        f"{len(region_dpts)} departments · "
        f"{len(sub_p) + len(sub_r):,} annotations",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=style.FONT_SIZE_SMALL,
        color="#555555",
    )

    _hide_axis_chrome(ax)

    plt.tight_layout()

    return fig, ax

def recall_by_source(recall_df, alpha_prior=ALPHA_PRIOR, beta_prior=BETA_PRIOR, ci=DEFAULT_CI):
    """Compare recall on OSM ground truth against manually annotated ground truth.

    This answers a question the audit depends on: does recall differ with the
    *origin* of the ground-truth point? If it did, the mix of sources in a unit
    would leak into its correction factor.

    ``national`` pools by source, the same method as the national summary but
    split by source rather than by unit. ``by_dept`` gives the point estimate per
    unit and source, so the two distributions can be seen rather than one number.
    """
    if 'source' not in recall_df.columns:
        raise ValueError("recall_df has no 'source' column; it is propagated from "
                          "recall_src.csv by recall.ipynb")

    national_rows = []
    for source, sub in recall_df.groupby('source'):
        tp = int(sub['pred'].sum())
        fn = int((sub['pred'] == 0).sum())
        _, _, mean, lo, hi, rel_unc = beta_posterior(tp, fn, alpha_prior, beta_prior, ci)
        national_rows.append({
            'source': source, 'n': len(sub), 'tp': tp, 'fn': fn,
            'recall_mean': mean, 'recall_ci_low': lo, 'recall_ci_high': hi,
            'recall_rel_uncertainty': rel_unc,
        })
    national = pd.DataFrame(national_rows)

    by_dept = (
        recall_df.groupby(['dpt', 'source'])
        .agg(n=('pred', 'size'), recall=('pred', 'mean'))
        .reset_index()
    )

    return national, by_dept


def plot_recall_by_source_boxplot(by_dept, min_n=10, ax=None):
    """Recall by unit, grouped by ground-truth source.

    Only unit-source pairs with at least ``min_n`` samples are shown, to keep out
    the noise from units where one source is nearly absent.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    sub = by_dept[by_dept['n'] >= min_n]
    sources = sorted(sub['source'].unique())
    data = [sub.loc[sub['source'] == s, 'recall'].values for s in sources]
    n_depts = [int((sub['source'] == s).sum()) for s in sources]

    # `labels=` rather than `tick_labels=`, which arrived in matplotlib 3.9, to
    # stay compatible with older versions.
    ax.boxplot(data, labels=[f"{s}\n(n={n} units)" for s, n in zip(sources, n_depts)])
    ax.set_ylabel('Recall by department')
    ax.set_title(f'Recall by department, OSM vs manual\n(departments with ≥{min_n} points per source)')

    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax


# ---------------------------------------------------------------------------
# Partie 4 -- dependance temporelle (millesime de l'imagerie BDORTHO)
# ---------------------------------------------------------------------------

def load_dept_year_mapping(dates_path):
    """Dominant imagery vintage per unit, from the dating step's output.

    Every unit in this dataset is covered by a single vintage: checked when this
    function was written, 96 of 96 units at 100% purity on the dominant year. A
    plain mode per unit is therefore enough, and nothing finer, such as weighting
    by municipality count, would change the answer.
    """
    dates = gpd.read_file(dates_path)
    years = pd.to_datetime(dates['imagedate']).dt.year
    dept_year = (
        pd.DataFrame({'dpt': dates['dpt'], 'year': years})
        .groupby('dpt')['year']
        .agg(lambda s: s.value_counts().idxmax())
        .rename('year')
        .reset_index()
    )
    return dept_year


def precision_recall_by_year(precision_points, recall_points, dept_year,
                              alpha_prior=ALPHA_PRIOR, beta_prior=BETA_PRIOR, ci=DEFAULT_CI):
    """Pooled precision and recall by imagery vintage.
    millesime -- teste si l'erreur de detection depend de l'annee de prise de vue (qualite
    This tests whether the *instrument* behaves differently across campaigns:
    sensor, exposure, resolution. It is a different question from whether the
    installation existed when the photograph was taken, which the temporal
    alignment already handles. Conflating the two is easy and would be wrong.
    """
    prec = precision_points.merge(dept_year, on='dpt', how='left')
    rec = recall_points.merge(dept_year, on='dpt', how='left')

    rows = []
    for year, sub in prec.groupby('year'):
        tp = int(sub['pred'].sum())
        fp = int((sub['pred'] == 0).sum())
        _, _, mean, lo, hi, rel_unc = beta_posterior(tp, fp, alpha_prior, beta_prior, ci)
        rows.append({'year': int(year), 'metric': 'precision', 'n': len(sub), 'tp': tp, 'fail': fp,
                      'mean': mean, 'ci_low': lo, 'ci_high': hi, 'rel_uncertainty': rel_unc})
    for year, sub in rec.groupby('year'):
        tp = int(sub['pred'].sum())
        fn = int((sub['pred'] == 0).sum())
        _, _, mean, lo, hi, rel_unc = beta_posterior(tp, fn, alpha_prior, beta_prior, ci)
        rows.append({'year': int(year), 'metric': 'recall', 'n': len(sub), 'tp': tp, 'fail': fn,
                      'mean': mean, 'ci_low': lo, 'ci_high': hi, 'rel_uncertainty': rel_unc})

    return pd.DataFrame(rows).sort_values(['metric', 'year']).reset_index(drop=True)


def plot_precision_recall_by_year_barplot(by_year, ax=None):
    """Grouped bar chart of precision and recall by vintage, with intervals."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    years = sorted(by_year['year'].unique())
    width = 0.35
    x = np.arange(len(years))

    for i, (metric, color) in enumerate([('precision', style.COLOR_TP), ('recall', style.COLOR_SOURCE_OSM)]):
        sub = by_year[by_year['metric'] == metric].set_index('year').reindex(years)
        yerr = np.array([sub['mean'] - sub['ci_low'], sub['ci_high'] - sub['mean']])
        ax.bar(x + (i - 0.5) * width, sub['mean'], width, yerr=yerr, capsize=3,
               label=metric.capitalize(), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel('Acquisition year (BD ORTHO imagery)')
    ax.set_ylabel('Value (95% CI)')
    ax.set_title('Precision / recall by acquisition year', fontweight='bold')
    ax.legend()

    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax


def bootstrap_year_effect_by_department_draw(precision_points, recall_points, dept_year,
                                               depts_per_year=1, n_draws=1000, seed=42):
    """Is the vintage effect real, or is it geography in disguise?

    Units sharing a vintage are often geographically contiguous, imagery
    campaigns being flown zone by zone. A precision or recall gap between
    vintages can therefore reflect terrain or urbanisation rather than anything
    about the sensor.

    This resamples, keeping only ``depts_per_year`` unit(s) per vintage on each
    draw, which dilutes the contribution of any single geographic cluster. If the
    gap survives the repeated subsampling, the vintage effect is robust. If it
    dissolves, what looked like a vintage effect on the full set was geography.

    A plain point estimate per draw, with no Beta posterior. What is wanted here
    is the dispersion due to the random *choice* of units, not a second layer of
    Monte-Carlo uncertainty on top; stacking the two would complicate the reading
    without
    ajouter d'information utile a ce diagnostic.

    One caveat to carry into the reading: 2022 has only two units in this dataset,
    so at ``depts_per_year=2`` that group has essentially no sampling variability,
    both units being drawn every time. Its box is narrow for a reason that has
    nothing to do with the effect being tested.

    Returns a DataFrame of (draw, year, metric, value).
    """
    prec = precision_points.merge(dept_year, on='dpt', how='left')
    rec = recall_points.merge(dept_year, on='dpt', how='left')
    years = sorted(dept_year['year'].unique())
    depts_by_year = {y: dept_year.loc[dept_year['year'] == y, 'dpt'].tolist() for y in years}

    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(n_draws):
        for year in years:
            pool = depts_by_year[year]
            k = min(depts_per_year, len(pool))
            chosen = rng.choice(pool, size=k, replace=False)

            sub_p = prec[prec['dpt'].isin(chosen)]
            if len(sub_p):
                rows.append({'draw': draw, 'year': year, 'metric': 'precision', 'value': sub_p['pred'].mean()})

            sub_r = rec[rec['dpt'].isin(chosen)]
            if len(sub_r):
                rows.append({'draw': draw, 'year': year, 'metric': 'recall', 'value': sub_r['pred'].mean()})

    return pd.DataFrame(rows)


def plot_year_effect_draws_boxplot(draws_df, metric='precision', ax=None):
    """Box plot of the resampling draws, one box per vintage.

    Read it against the pooled bar chart. Boxes that overlap heavily from one
    vintage to the next mean the pooled gap is not robust to which units happen
    to represent each year; boxes that stay apart mean the vintage effect holds.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    sub = draws_df[draws_df['metric'] == metric]
    years = sorted(sub['year'].unique())
    data = [sub.loc[sub['year'] == y, 'value'].values for y in years]

    ax.boxplot(data, labels=[str(y) for y in years])
    ax.set_xlabel('Acquisition year')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} by year -- representative department draws", fontweight='bold')

    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax
