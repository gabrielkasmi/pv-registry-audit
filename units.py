"""Reporting units for the correction and the evaluation metrics.

The unit is the department, with one exception: Paris and the three inner-ring
departments (75, 92, 93, 94) are merged into a single unit, ``75PC``.

Why. Across the continuous urban fabric of Greater Paris the departmental
boundary is an administrative artefact. The four share a distributor (Enedis),
they share an orthoimagery campaign (August 2024 for all four), and their
building morphology is comparable. Merging pools their annotation samples,
roughly 480 precision points and 350 recall points, where Paris on its own was
the worst measured unit in the country: label agreement of 0.435 before
re-labelling, and the widest uncertainty on its correction factor. The four
landed on the same side of the audit anyway, below or within, so the merge
changes no verdict.

The merge is applied uniformly from the calibration step onwards: annotation
pooling in ``score_and_aggregation``, raw totals and dissolved geometry in
``recalibrate.py``, the aggregated reference in the merge and comparison steps,
and every map. Upstream steps (ground-truth construction, matching, dating)
stay at the department: this is an aggregation, not a change of data.

To go back to plain departments, set ``DEPT_AGGREGATION = {}``. One line.
"""

# Department code -> merged unit. Any code absent from the mapping is unchanged.
DEPT_AGGREGATION = {
    '75': '75PC',
    '92': '75PC',
    '93': '75PC',
    '94': '75PC',
}

# The label is stored as-is in the published tables, so it stays in French: it
# is a data value, not display text, and translating it here would silently
# break the join with every file that carries it.
UNIT_LABELS = {'75PC': 'Paris + petite couronne'}


def to_unit(dpt_series):
    """Department codes -> reporting units. Identity outside DEPT_AGGREGATION."""
    return dpt_series.map(lambda d: DEPT_AGGREGATION.get(d, d))


def apply_units(df, col='dpt'):
    """Return a copy of ``df`` with column ``col`` replaced by the reporting unit."""
    out = df.copy()
    out[col] = to_unit(out[col])
    return out


def dissolve_units(gdf, code_col='code'):
    """Departmental geometries -> unit geometries.

    Dissolves the merged departments into one polygon, keeping the first value
    of every other column. The merged unit's name comes from UNIT_LABELS.
    """
    g = gdf.copy()
    g['_unit'] = to_unit(g[code_col])
    dissolved = g.dissolve(by='_unit', as_index=False, aggfunc='first')
    dissolved[code_col] = dissolved['_unit']
    if 'nom' in dissolved.columns:
        dissolved['nom'] = dissolved.apply(
            lambda r: UNIT_LABELS.get(r[code_col], r['nom']), axis=1)
    return dissolved.drop(columns='_unit')
