"""Assemble the sources: detections against each registry, date-matched.

Produces:

- ``comparison_communes.csv`` -- raw detections and date-matched connection data,
  by municipality
- ``comparison_departements.geojson`` -- corrected estimate and connection data,
  93 reporting units
- ``comparison_communes_rni.csv`` -- the public registry variants, by municipality
- ``comparison_departements_rni_bottomup.geojson`` -- the public registry as
  published at municipal level, and therefore *carrying* the truncation bias
  - comparison_departements_rni_total.geojson     (RNI total -- troncation corrigee)

Includes the dating patch, which dates municipalities carrying no detection
directly from the mosaicking graphs, and the crosswalk that folds the
arrondissements of Paris, Lyon and Marseille into their city.

Consumed by the three audit notebooks.

Usage : python merge.py   (depuis 0_data/postprocess/merge-sources/)
"""

# ---------------------------------------------------------------------------
# NOT REPRODUCIBLE FROM THE RELEASED DATA
#
# The municipality-level date matching below needs the operator's
# per-installation extract, with a connection date per installation. That
# extract is not redistributable, and no aggregate can stand in for it: the
# matching is a per-installation comparison against a per-municipality imagery
# date, which is exactly the granularity that is withheld.
#
# The outputs of this script ship in data/intermediate/postprocess, so
# everything downstream of it runs. This step alone cannot be replayed.
# ---------------------------------------------------------------------------


# ======================================================================
# ======================================================================
import pandas as pd
import geopandas as gpd
import json
import os
import time

# ======================================================================
# ======================================================================
# STEP 1 - File imports
dpvm_uncalibrated = gpd.read_file(paths.IMAGERY_DATES)
dpvm_calibrated = gpd.read_file(paths.RECALIBRATION)
rte = pd.read_csv('../../src/rte/installations_rte.csv', dtype={'codeinseecommune': str, 'codedepartement': str}, low_memory=False)
rni = pd.read_csv(paths.RNI)

# administrative_boundaries
SRC_PATH = '../../src/admin-boundaries/ADMIN-EXPRESS_3-2__SHP_LAMB93_FXX_2025-02-03/ADMIN-EXPRESS/1_DONNEES_LIVRAISON_2025-02-00021/ADE_3-2_SHP_LAMB93_FXX-ED2025-02-03/'
cities_shp= SRC_PATH + "/COMMUNE.shp"
cities_shp=gpd.read_file(cities_shp)

# arrondissements to handle the cases of Paris, Lyon and Marseille
arrdt_shp=  SRC_PATH + "/ARRONDISSEMENT_MUNICIPAL.shp"
arrdt_shp=gpd.read_file(arrdt_shp)

cities_shp=cities_shp[['NOM', 'INSEE_COM', "POPULATION", "INSEE_DEP",'geometry']]
cities_shp = cities_shp.to_crs("EPSG:4326")  # Convert to WGS84

arrdt_shp=arrdt_shp[['NOM', 'INSEE_COM', "POPULATION", "INSEE_ARM", "INSEE_COM",'geometry']]
arrdt_shp = arrdt_shp.loc[:, ~arrdt_shp.columns.duplicated()]
arrdt_shp = arrdt_shp.to_crs("EPSG:4326")  # Convert to WGS84

# ======================================================================
# ======================================================================
# --- Base ---
cities_all = cities_shp.rename(columns={'INSEE_COM': 'code_ville'})

# --- Crosswalk arrondissement -> ville (attributaire seulement) ---
crosswalk = (
    arrdt_shp[['INSEE_ARM', 'INSEE_COM']]
    .drop_duplicates()
    .rename(columns={'INSEE_ARM': 'code_source', 'INSEE_COM': 'code_ville'})
)
insee_valides = set(cities_shp['INSEE_COM'])

# ---- Step 1: spatial join, detections onto municipalities ----
dpvm_joined = gpd.sjoin(
    dpvm_uncalibrated, cities_all[['code_ville', 'NOM', 'geometry']],
    how='left', predicate='within'
)
mismatch = dpvm_joined[dpvm_joined['insee'] != dpvm_joined['code_ville']]
print(f"{len(mismatch)} mismatches / {len(dpvm_joined)}")

# ---- Step 2: prefer the detection's own INSEE code where the two disagree ----
def resolve_code_ville(row):
    if pd.notna(row['code_ville']) and row['insee'] == row['code_ville']:
        return row['code_ville']  # join spatial ok
    if row['insee'] in insee_valides:
        return row['insee']  # geometric join failed (border, island): trust the code
    return row['code_ville']  # stale code after a municipal merger: trust the geometry

dpvm_fix = dpvm_joined.copy()
dpvm_fix['code_ville_final'] = dpvm_fix.apply(resolve_code_ville, axis=1)

unresolved = dpvm_fix[dpvm_fix['code_ville_final'].isna()]
print(f"{len(unresolved)} unresolved cases")

# ---- Step 3: fold arrondissements into their city, then aggregate ----
dpvm_fix['code_ville_final'] = dpvm_fix['code_ville_final'].map(
    crosswalk.set_index('code_source')['code_ville']
).fillna(dpvm_fix['code_ville_final'])

dpvm_agg = (
    dpvm_fix.groupby('code_ville_final')
    .agg(
        n_installations=('n_installations', 'sum'),
        kWp_total=('kWp_total', 'sum'),
        year_median=('year_median', 'first'),
        imagedate=('imagedate', 'first'),
    )
    .reset_index()
    .rename(columns={'code_ville_final': 'code_ville'})
)
print(dpvm_agg['code_ville'].duplicated().sum())  # must be 0

# ---- Step 4: attach the aggregated detections to the municipalities ----
cities_all = cities_all.merge(dpvm_agg, on='code_ville', how='left')


# ---- Dating patch: municipalities with no detection get a day-level date ----
# A municipality carrying no detection has no imagery date of its own, and was
# falling back to its department's year. Instead, its polygon is dated directly
# against the department's mosaicking graph, taking the tile that dominates by
# area. Cached, and recomputed automatically when a municipality is missing.
import re as _re
from pathlib import Path as _Path
_ASSEMBLAGE = _Path('../dates/assemblage')
_CACHE = _Path(paths.DATES_PATCH)

cities_all['imagedate'] = pd.to_datetime(cities_all['imagedate'])
_missing = cities_all[cities_all['imagedate'].isna()][['code_ville', 'INSEE_DEP', 'geometry']]
_patch = None
if _CACHE.exists():
    _p = pd.read_csv(_CACHE, dtype={'code_ville': str})
    if set(_missing['code_ville']) <= set(_p['code_ville']):
        _patch = _p
if _patch is None:
    def _vector_file(folder):
        vf = sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.gpkg', '.shp'))
        return vf[0] if vf else None
    _folders = {}
    for _f in sorted(_ASSEMBLAGE.iterdir()):
        _m = _re.search(r'D([0-9]{2}[0-9AB])_(\d{4})', _f.name) if _f.is_dir() else None
        if _m:
            _dpt, _yr = _m.group(1).lstrip('0').zfill(2), int(_m.group(2))
            if _dpt not in _folders or _yr > _folders[_dpt][0]:
                _folders[_dpt] = (_yr, _f)
    _rows = []
    for _dpt, _grp in _missing.groupby('INSEE_DEP'):
        if _dpt not in _folders or _vector_file(_folders[_dpt][1]) is None:
            continue
        # Date of the tile containing the municipality's representative point.
        # This equals the dominant tile in nearly every case; an exact overlay
        # at shot level is too heavy across 89 departments.
        import pyogrio as _pyogrio
        _g = _grp.to_crs(2154)
        _tiles = _pyogrio.read_dataframe(str(_vector_file(_folders[_dpt][1])),
                                         columns=['DATE'], bbox=tuple(_g.total_bounds))
        if _tiles.crs is None:
            _tiles = _tiles.set_crs(2154)
        _pts = _g.copy(); _pts['geometry'] = _pts.representative_point()
        _j = gpd.sjoin(_pts, _tiles.to_crs(2154)[['DATE', 'geometry']],
                       how='left', predicate='within')
        _j = _j[~_j.index.duplicated(keep='first')]
        _rows.append(_j[['code_ville', 'DATE']].dropna())
    _patch = (pd.concat(_rows, ignore_index=True)
                .rename(columns={'DATE': 'imagedate_patch'}))
    _patch.to_csv(_CACHE, index=False)
_patch.columns = ['code_ville', 'imagedate_patch']
cities_all = cities_all.merge(_patch, on='code_ville', how='left')
cities_all['imagedate'] = cities_all['imagedate'].fillna(pd.to_datetime(cities_all['imagedate_patch']))
print(f"patch graphes : {cities_all['imagedate_patch'].notna().sum()} communes datees au jour ; "
      f"still undated, falling back to the departmental year: {cities_all['imagedate'].isna().sum()}")
cities_all = cities_all.drop(columns=['imagedate_patch'])

# ---- Fallback: departmental reference year for whatever is still undated ----
dpt_year = (
    dpvm_uncalibrated.groupby('dpt')['year_median']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index().rename(columns={'year_median': 'year_ref'})
)
cities_all = cities_all.merge(dpt_year, left_on='INSEE_DEP', right_on='dpt', how='left')
cities_all['imagedate_effective'] = cities_all['imagedate'].fillna(
    pd.to_datetime(cities_all['year_ref'].astype('Int64').astype(str) + '-12-31')
)

# ---- Step 5: attach the connection data ----
rte_norm = rte.merge(crosswalk, left_on='code_insee_commune', right_on='code_source', how='left')
rte_norm['code_ville_final'] = rte_norm['code_ville'].fillna(rte_norm['code_insee_commune'])

rte_norm = rte_norm.merge(
    cities_all[['code_ville', 'imagedate_effective']],
    left_on='code_ville_final', right_on='code_ville', how='left'
)
rte_norm['date_raccordement'] = pd.to_datetime(rte_norm['date_raccordement'])
rte_filtered = rte_norm[rte_norm['date_raccordement'] <= rte_norm['imagedate_effective']]

rte_agg = (
    rte_filtered.groupby('code_ville_final')
    .agg(n_installations_rte=('p_inst', 'count'), kWp_total_rte=('p_inst', 'sum'))
    .reset_index().rename(columns={'code_ville_final': 'code_ville'})
)

cities_all = cities_all.merge(rte_agg, on='code_ville', how='left')

# ---- Step 6: final clean-up ----
cities_all_clean = cities_all[[
    'NOM', 'code_ville', 'POPULATION', 'INSEE_DEP', 'geometry',
    'imagedate', 'year_ref', 'n_installations', 'kWp_total',
    'n_installations_rte', 'kWp_total_rte'
]].rename(columns={
    'NOM': 'nom',
    'code_ville': 'code_insee',
    'POPULATION': 'population',
    'INSEE_DEP': 'dpt',
    'year_ref': 'year',
    'n_installations': 'n_installations_dpvm',
    'kWp_total': 'kWp_dpvm',
    'kWp_total_rte': 'kWp_rte',
})

cities_all_clean[['n_installations_dpvm', 'kWp_dpvm', 'n_installations_rte', 'kWp_rte']] = (
    cities_all_clean[['n_installations_dpvm', 'kWp_dpvm', 'n_installations_rte', 'kWp_rte']].fillna(0)
)

cities_all_clean = gpd.GeoDataFrame(cities_all_clean, geometry='geometry', crs=cities_shp.crs)

# ======================================================================
# ======================================================================
# Municipality export: attributes only. The geometry is heavy and can be
# re-joined on the INSEE code from the boundary file if a map is needed.
cities_export = cities_all_clean.drop(columns='geometry')
cities_export.to_csv(paths.COMPARISON_COMMUNES, index=False)
print(f"Ecrit : comparison_communes.csv ({len(cities_export)} communes)")


# ======================================================================
# ======================================================================
# Sanity check: national totals in MWp, detections still uncorrected.
print(f"dpvm brut : {cities_all_clean['kWp_dpvm'].sum()/1000:,.1f} MWc | "
      f"RTE date-matche : {cities_all_clean['kWp_rte'].sum()/1000:,.1f} MWc")


# ======================================================================
# ======================================================================
# Reporting units: 75, 92, 93 and 94 merge into 75PC. See units.py.
import sys
from pathlib import Path
def _find_root(start, marker='paths.py'):
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(marker)
sys.path.insert(0, str(_find_root(Path.cwd())))
import paths
import units

cities_all_clean['unit'] = units.to_unit(cities_all_clean['dpt'])
dpt_agg = (
    cities_all_clean.groupby('unit')
    .agg(
        n_installations_rte=('n_installations_rte', 'sum'),
        kWp_rte=('kWp_rte', 'sum'),
    )
    .reset_index()
    .rename(columns={'unit': 'dpt'})
)

# Merge onto the corrected table, already expressed in reporting units.
comparison = dpvm_calibrated.merge(dpt_agg, on='dpt', how='left')

comparison[['n_installations_rte', 'kWp_rte']] = (
    comparison[['n_installations_rte', 'kWp_rte']].fillna(0)
)

comparison = gpd.GeoDataFrame(comparison, geometry='geometry', crs=dpvm_calibrated.crs)


# ======================================================================
# ======================================================================
import tempfile, shutil as _sh, os as _os
_tmp = _os.path.join(tempfile.mkdtemp(), paths.COMPARISON_DEPT)
comparison.to_file(_tmp, driver='GeoJSON')
_sh.copy(_tmp, paths.COMPARISON_DEPT)
print(f"wrote comparison_departements.geojson ({len(comparison)} units)")
comparison.drop(columns='geometry').head()


# ======================================================================
# ======================================================================
# --- Base ---
cities_all = cities_shp.rename(columns={'INSEE_COM': 'code_ville'})

# ---- Reporting unit, computed once here and carried through ----
# Rather than recomputed just before the final groupby, which is how the two
# can silently drift apart.
import sys
from pathlib import Path
def _find_root(start, marker='paths.py'):
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(marker)
sys.path.insert(0, str(_find_root(Path.cwd())))
import units

cities_all['unit'] = units.to_unit(cities_all['INSEE_DEP'])

# --- Crosswalk arrondissement -> ville (attributaire seulement) ---
crosswalk = (
    arrdt_shp[['INSEE_ARM', 'INSEE_COM']]
    .drop_duplicates()
    .rename(columns={'INSEE_ARM': 'code_source', 'INSEE_COM': 'code_ville'})
)
insee_valides = set(cities_shp['INSEE_COM'])

# ---- Step 1: spatial join, detections onto municipalities ----
dpvm_joined = gpd.sjoin(
    dpvm_uncalibrated, cities_all[['code_ville', 'NOM', 'geometry']],
    how='left', predicate='within'
)
mismatch = dpvm_joined[dpvm_joined['insee'] != dpvm_joined['code_ville']]
print(f"{len(mismatch)} mismatches / {len(dpvm_joined)}")

# ---- Step 2: prefer the detection's own INSEE code where the two disagree ----
def resolve_code_ville(row):
    if pd.notna(row['code_ville']) and row['insee'] == row['code_ville']:
        return row['code_ville']  # join spatial ok
    if row['insee'] in insee_valides:
        return row['insee']  # geometric join failed (border, island): trust the code
    return row['code_ville']  # stale code after a municipal merger: trust the geometry

dpvm_fix = dpvm_joined.copy()
dpvm_fix['code_ville_final'] = dpvm_fix.apply(resolve_code_ville, axis=1)

unresolved = dpvm_fix[dpvm_fix['code_ville_final'].isna()]
print(f"{len(unresolved)} unresolved cases")

# ---- Step 3: fold arrondissements into their city, then aggregate ----
dpvm_fix['code_ville_final'] = dpvm_fix['code_ville_final'].map(
    crosswalk.set_index('code_source')['code_ville']
).fillna(dpvm_fix['code_ville_final'])

dpvm_agg = (
    dpvm_fix.groupby('code_ville_final')
    .agg(
        n_installations=('n_installations', 'sum'),
        kWp_total=('kWp_total', 'sum'),
        year_median=('year_median', 'first'),
        imagedate=('imagedate', 'first'),
    )
    .reset_index()
    .rename(columns={'code_ville_final': 'code_ville'})
)
print(dpvm_agg['code_ville'].duplicated().sum())  # must be 0

# ---- Step 4: attach the aggregated detections to the municipalities ----
cities_all = cities_all.merge(dpvm_agg, on='code_ville', how='left')


# ---- Dating patch: municipalities with no detection ----
import re as _re
_ASSEMBLAGE = Path('../dates/assemblage')
_CACHE = Path(paths.DATES_PATCH)

cities_all['imagedate'] = pd.to_datetime(cities_all['imagedate'])
_missing = cities_all[cities_all['imagedate'].isna()][['code_ville', 'INSEE_DEP', 'geometry']]
_patch = None
if _CACHE.exists():
    _p = pd.read_csv(_CACHE, dtype={'code_ville': str})
    if set(_missing['code_ville']) <= set(_p['code_ville']):
        _patch = _p
if _patch is None:
    def _vector_file(folder):
        vf = sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.gpkg', '.shp'))
        return vf[0] if vf else None
    _folders = {}
    for _f in sorted(_ASSEMBLAGE.iterdir()):
        _m = _re.search(r'D([0-9]{2}[0-9AB])_(\d{4})', _f.name) if _f.is_dir() else None
        if _m:
            _dpt, _yr = _m.group(1).lstrip('0').zfill(2), int(_m.group(2))
            if _dpt not in _folders or _yr > _folders[_dpt][0]:
                _folders[_dpt] = (_yr, _f)
    _rows = []
    for _dpt, _grp in _missing.groupby('INSEE_DEP'):
        if _dpt not in _folders or _vector_file(_folders[_dpt][1]) is None:
            continue
        import pyogrio as _pyogrio
        _g = _grp.to_crs(2154)
        _tiles = _pyogrio.read_dataframe(str(_vector_file(_folders[_dpt][1])),
                                         columns=['DATE'], bbox=tuple(_g.total_bounds))
        if _tiles.crs is None:
            _tiles = _tiles.set_crs(2154)
        _pts = _g.copy(); _pts['geometry'] = _pts.representative_point()
        _j = gpd.sjoin(_pts, _tiles.to_crs(2154)[['DATE', 'geometry']],
                       how='left', predicate='within')
        _j = _j[~_j.index.duplicated(keep='first')]
        _rows.append(_j[['code_ville', 'DATE']].dropna())
    _patch = (pd.concat(_rows, ignore_index=True)
                .rename(columns={'DATE': 'imagedate_patch'}))
    _patch.to_csv(_CACHE, index=False)
_patch.columns = ['code_ville', 'imagedate_patch']
cities_all = cities_all.merge(_patch, on='code_ville', how='left')
cities_all['imagedate'] = cities_all['imagedate'].fillna(pd.to_datetime(cities_all['imagedate_patch']))
print(f"patch graphes : {cities_all['imagedate_patch'].notna().sum()} communes datees au jour ; "
      f"still undated, falling back to the departmental year: {cities_all['imagedate'].isna().sum()}")
cities_all = cities_all.drop(columns=['imagedate_patch'])

# ---- Fallback: departmental reference year for the still undated ----
# Used twice: for the municipal fallback below, and to filter the registry
# rows that are not attached to any municipality, which have no median year of
# their own.
dpt_year = (
    dpvm_uncalibrated.groupby('dpt')['year_median']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index().rename(columns={'year_median': 'year_ref'})
)
cities_all = cities_all.merge(dpt_year, left_on='INSEE_DEP', right_on='dpt', how='left')
cities_all['imagedate_effective'] = cities_all['imagedate'].fillna(
    pd.to_datetime(cities_all['year_ref'].astype('Int64').astype(str) + '-12-31')
)

# ---- Step 5: attach the public registry, at municipal level ----
# The registry is cumulative by year, so summing across years would count the
# same installations several times. Only the vintage matching the
# municipality's median detection year is kept. This is a selection, not a date
# comparison, which is what the connection data allows and this does not.
rni_communal = rni[rni['codeinseecommune'].notna()].copy()
print(f"registry: {len(rni_communal)}/{len(rni)} rows attached to a municipality "
      f"({len(rni) - len(rni_communal)} departmental or regional aggregates, handled separately)")

rni_communal = rni_communal.merge(crosswalk, left_on='codeinseecommune', right_on='code_source', how='left')
rni_communal['code_ville_final'] = rni_communal['code_ville'].fillna(rni_communal['codeinseecommune'])

rni_communal = rni_communal.merge(
    cities_all[['code_ville', 'year_median']],
    left_on='code_ville_final', right_on='code_ville', how='left'
)
# Note: a municipality with no detection has no median year, so no registry row
# matches and it carries no registry capacity. That is a direct consequence of
# selecting on the exact year, and it is deliberate.
rni_filtered = rni_communal[rni_communal['year'] == rni_communal['year_median']]

rni_agg = (
    rni_filtered.groupby('code_ville_final')
    .agg(n_installations_rni=('nbinstallations', 'sum'), kWp_total_rni=('p_inst', 'sum'))
    .reset_index().rename(columns={'code_ville_final': 'code_ville'})
)

cities_all = cities_all.merge(rni_agg, on='code_ville', how='left')

# ---- Step 6: final clean-up, municipal level ----
cities_all_clean = cities_all[[
    'NOM', 'code_ville', 'POPULATION', 'INSEE_DEP', 'unit', 'geometry',
    'imagedate', 'year_ref', 'n_installations', 'kWp_total',
    'n_installations_rni', 'kWp_total_rni'
]].rename(columns={
    'NOM': 'nom',
    'code_ville': 'code_insee',
    'POPULATION': 'population',
    'INSEE_DEP': 'dpt',
    'year_ref': 'year',
    'n_installations': 'n_installations_dpvm',
    'kWp_total': 'kWp_dpvm',
    'kWp_total_rni': 'kWp_rni',
})

cities_all_clean[['n_installations_dpvm', 'kWp_dpvm', 'n_installations_rni', 'kWp_rni']] = (
    cities_all_clean[['n_installations_dpvm', 'kWp_dpvm', 'n_installations_rni', 'kWp_rni']].fillna(0)
)

cities_all_clean = gpd.GeoDataFrame(cities_all_clean, geometry='geometry', crs=cities_shp.crs)

cities_export = cities_all_clean.drop(columns='geometry')
cities_export.to_csv(paths.COMPARISON_COMMUNES_RNI, index=False)
print(f"Ecrit : comparison_communes_rni.csv ({len(cities_export)} communes)")

# ---- Step 7: departmental aggregation, two variants ----

# Variant 1, bottom-up: the sum of the municipal capacities.
dpt_agg_bottomup = (
    cities_all_clean.groupby('unit')
    .agg(
        n_installations_rni=('n_installations_rni', 'sum'),
        kWp_rni=('kWp_rni', 'sum'),
    )
    .reset_index()
    .rename(columns={'unit': 'dpt'})
)

# Variant 2, total: bottom-up plus the registry capacity that is not attached to
# any municipality, which the registry publishes as departmental aggregates.
# Selected on the department's reference year, there being no municipal median
# to use here. The difference between the two variants is the truncation bias.
rni_dept_only = rni[rni['codeinseecommune'].isna() & rni['codedepartement'].notna()].copy()
print(f"registry: {len(rni_dept_only)} rows aggregated at department level")

rni_dept_only = rni_dept_only.merge(dpt_year, left_on='codedepartement', right_on='dpt', how='left')
rni_dept_only = rni_dept_only[rni_dept_only['year'] == rni_dept_only['year_ref']].copy()
rni_dept_only['unit'] = units.to_unit(rni_dept_only['codedepartement'])

dept_unattributed = (
    rni_dept_only.groupby('unit')
    .agg(n_installations_rni_nonattr=('nbinstallations', 'sum'),
         kWp_rni_nonattr=('p_inst', 'sum'))
    .reset_index()
    .rename(columns={'unit': 'dpt'})
)

dpt_agg_total = dpt_agg_bottomup.merge(dept_unattributed, on='dpt', how='outer')
dpt_agg_total[['n_installations_rni', 'kWp_rni', 'n_installations_rni_nonattr', 'kWp_rni_nonattr']] = (
    dpt_agg_total[['n_installations_rni', 'kWp_rni', 'n_installations_rni_nonattr', 'kWp_rni_nonattr']].fillna(0)
)
dpt_agg_total['n_installations_rni'] += dpt_agg_total['n_installations_rni_nonattr']
dpt_agg_total['kWp_rni'] += dpt_agg_total['kWp_rni_nonattr']
dpt_agg_total = dpt_agg_total.drop(columns=['n_installations_rni_nonattr', 'kWp_rni_nonattr'])

# Merge onto the corrected table, already expressed in reporting units.
import tempfile, shutil as _sh, os as _os

for _name, _dpt_agg in [('bottomup', dpt_agg_bottomup), ('total', dpt_agg_total)]:
    comparison = dpvm_calibrated.merge(_dpt_agg, on='dpt', how='left')
    comparison[['n_installations_rni', 'kWp_rni']] = comparison[['n_installations_rni', 'kWp_rni']].fillna(0)
    comparison = gpd.GeoDataFrame(comparison, geometry='geometry', crs=dpvm_calibrated.crs)

    _fname = f'comparison_departements_rni_{_name}.geojson'
    _tmp = _os.path.join(tempfile.mkdtemp(), _fname)
    comparison.to_file(_tmp, driver='GeoJSON')
    _sh.copy(_tmp, _fname)
    print(f"wrote {_fname} ({len(comparison)} units)")

comparison.drop(columns='geometry').head()

