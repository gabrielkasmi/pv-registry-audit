"""Regenerate and plot the per-unit posterior distributions of corrected quantities.

Shared module, used by the postprocess visualisation notebook and by the audit
notebooks, so that one implementation produces one rendering.

Two reasons it exists rather than being inlined.

The draws are regenerated with **the same Beta parameters as the main chain**,
read from the alpha_* and beta_* columns of ``table.csv``, and with the same
deterministic per-unit seed. The histogram displayed is therefore exactly the
distribution whose mean and intervals ``recalibration_departements.geojson``
summarises, not an approximation of it. An earlier version rebuilt a Jeffreys
prior by hand and drifted out of step once the chain moved to empirical Bayes;
that is the failure this module prevents.

The plot accepts an external reference value, overlaid as a vertical line. That
line is the audit verdict read visually: is it inside the interval or not?

    import dept_posteriors as dp
    draws = dp.compute_dept_draws(['29', '86', '2A'], calibration, recal)
    fig = dp.plot_depts(draws, ref_values={'29': 41000}, quantity='p_inst')
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'code' / '0_pipeline' / 'postprocess' / 'recalibration'))
import style          # noqa: E402  (applies the rcParams on import)
import bootstrap      # noqa: E402

N_BOOT = 10000
SEED = 42

QUANTITY_LABELS = {
    'p_inst': ("Corrected installed capacity (kWp)", 'p_inst_raw'),
    'n_inst': ("Corrected installation count", 'n_inst_raw'),
}


def compute_dept_draws(depts, calibration, recal, quantities=('p_inst',), n_boot=N_BOOT, seed=SEED):
    """Regenerate the bootstrap draws for a list of reporting units.

    ``calibration`` is ``table.csv``. The Beta parameters come from the
    alpha_* and beta_* columns when present, which is the main chain's prior;
    older tables without them fall back to reconstructing Jeffreys from the
    counts.

    ``recal`` is the correction output, and supplies the raw values.

    Returns ``{unit: bootstrap_recalibration(..., return_draws=True)}``.
    """
    beta_cols = ['alpha_precision', 'beta_precision', 'alpha_recall', 'beta_recall']
    has_beta = all(col in calibration.columns for col in beta_cols)

    out = {}
    for dept in depts:
        c = calibration.loc[calibration['dpt'] == dept]
        r = recal.loc[recal['dpt'] == dept]
        if c.empty or r.empty:
            raise KeyError(f"unit {dept!r} missing from calibration or from recal")
        c, r = c.iloc[0], r.iloc[0]

        if has_beta:
            alpha_p, beta_p = c['alpha_precision'], c['beta_precision']
            alpha_r, beta_r = c['alpha_recall'], c['beta_recall']
        else:
            alpha_p, beta_p = bootstrap.ALPHA_PRIOR + c['tp_precision'], bootstrap.BETA_PRIOR + c['fp_precision']
            alpha_r, beta_r = bootstrap.ALPHA_PRIOR + c['tp_recall'], bootstrap.BETA_PRIOR + c['fn_recall']

        raw_values = {q: r[QUANTITY_LABELS[q][1]] for q in quantities}
        out[dept] = bootstrap.bootstrap_recalibration(
            raw_values, alpha_p, beta_p, alpha_r, beta_r,
            n_boot=n_boot, seed=seed, key=dept, return_draws=True,
        )
    return out


def plot_dept_posterior(ax, dept, result, quantity='p_inst', ref_value=None,
                         ref_label='RTE (date-matched)', bins=60, unit_scale=1.0):
    """Draw one unit's posterior on ``ax``.

    Histogram of the draws, the raw value as a dashed line, the corrected mean,
    shaded 95 and 99% bands, and, when given, the reference value as a vertical
    line. Pass ``unit_scale=1e-3`` to display in MWp.
    """
    res = result[quantity]
    draws = result['draws'][quantity] * unit_scale

    ax.hist(draws, bins=bins, color=style.COLOR_SOURCE_OSM, alpha=0.7)
    ax.axvline(res['raw'] * unit_scale, color=style.COLOR_RAW, linestyle='--', linewidth=1.2, label='Raw')
    ax.axvline(res['mean'] * unit_scale, color=style.COLOR_MEAN, linewidth=1.5, label='Corrected mean')
    ax.axvspan(res['ci95_low'] * unit_scale, res['ci95_high'] * unit_scale,
               color=style.COLOR_MEAN, alpha=0.15, label='95% CI')
    ax.axvspan(res['ci99_low'] * unit_scale, res['ci99_high'] * unit_scale,
               color=style.COLOR_MEAN, alpha=0.07, label='99% CI')
    if ref_value is not None and np.isfinite(ref_value):
        ax.axvline(ref_value * unit_scale, color=style.COLOR_REGISTRY, linewidth=2.0, label=ref_label)

    ax.set_title(f"Department {dept}", fontweight='bold')
    xlabel = QUANTITY_LABELS[quantity][0]
    if unit_scale == 1e-3:
        xlabel = xlabel.replace('(kWp)', '(MWp)')
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper right', fontsize=8)
    return ax


def plot_depts(draws_by_dept, ref_values=None, quantity='p_inst', unit_scale=1.0,
                figsize_per_panel=(5, 4.5)):
    """A 1xN grid of per-unit posteriors.

    ``ref_values`` maps a unit to its reference value, or is None. Returns the
    figure; saving it is the caller's business.
    """
    depts = list(draws_by_dept)
    fig, axes = plt.subplots(1, len(depts),
                              figsize=(figsize_per_panel[0] * len(depts), figsize_per_panel[1]),
                              sharey=False)
    if len(depts) == 1:
        axes = [axes]
    for ax, dept in zip(axes, depts):
        ref = (ref_values or {}).get(dept)
        plot_dept_posterior(ax, dept, draws_by_dept[dept], quantity=quantity,
                             ref_value=ref, unit_scale=unit_scale)
    axes[0].set_ylabel('Bootstrap draws')
    fig.tight_layout()
    return fig
