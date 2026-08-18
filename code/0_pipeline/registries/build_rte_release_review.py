"""Bundle the operator-derived files into one JSON for the operator's review.

Everything in ``data/source/rte_derived/`` that we intend to publish, in a single
record, so the operator sees the whole redistribution request at once rather than
opening five files. The small tables travel in full; the one large table is
summarised, because what matters about it is what it does *not* contain.

Run it whenever anything in ``rte_derived/`` changes, so the review copy and the
release cannot drift apart.

    python build_rte_release_review.py
    python build_rte_release_review.py --output ~/Desktop/rte_release_review.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _find_root(start, marker="paths.py"):
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(marker)


ROOT = _find_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402

METADATA = {
    "objet": "Donnees derivees du registre de raccordement, destinees a la publication",
    "perimetre": "Photovoltaique en toiture <= 36 kVA, France continentale",
    "maille": ("Departement (unite de report), sauf operator_by_municipality "
               "(commune, sans volume)"),
    "absent": [
        "identifiant d'installation",
        "adresse",
        "coordonnees",
        "date de raccordement individuelle",
        "capacite a la maille communale",
        "nombre d'installations dans la serie temporelle",
    ],
    "date_extraction": "2025-10-13",
    "controle_divulgation": (
        "Serie de coupures: le nombre d'installations n'est pas publie, et toute "
        "valeur cumulee dont l'increment depuis la coupure precedente n'excede "
        "pas 36 kWp est retiree. Le perimetre de l'etude plafonnant chaque "
        "installation a 36 kWp, tout ecart entre deux coupures publiees recouvre "
        "necessairement au moins deux installations -- verifie de facon exhaustive "
        "sur les 10 633 paires de coupures, dont la plus petite difference "
        "strictement positive vaut 36,1 kWp. La valeur auditee (audit_reference) "
        "n'est jamais retiree. Regle appliquee par "
        "code/0_pipeline/registries/apply_disclosure_control.py."
    ),
}

DESCRIPTIONS = {
    "departmental_cutoffs": (
        "Capacite cumulee par unite de report a chaque date de coupure. Les dates "
        "sont des dates de prise de vue aerienne (metadonnee IGN publique) et non "
        "des dates de raccordement. Sans nombre d'installations, et increments "
        "fins retires (voir controle_divulgation)."
    ),
    "eld_within_department": (
        "Couverture mediane du registre par type de gestionnaire, au sein d'un "
        "meme departement. Aucun volume individuel."
    ),
    "capacity_deciles": "Deciles de puissance, trois departements a titre illustratif.",
    "operator_by_municipality": (
        "Gestionnaire de reseau desservant chaque commune. Ni capacite, ni "
        "nombre d'installations. Necessaire au test de couverture ELD / Enedis "
        "intra-departemental."
    ),
}


def _plain(obj):
    """Fallback for anything json refuses: numpy scalars, Timestamps, NA.

    ``value_counts().to_dict()`` returns ``numpy.int64`` on older pandas, which
    ``json.dumps`` rejects outright. Rather than depend on the pandas version, we
    coerce here and let the encoder call this for whatever it does not recognise.
    """
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if obj is pd.NaT or obj is pd.NA:
        return None
    return str(obj)


def table(df, key):
    return {
        "description": DESCRIPTIONS[key],
        "n_lignes": int(len(df)),
        "colonnes": [str(c) for c in df.columns],
        "donnees": json.loads(df.to_json(orient="records", force_ascii=False)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT.parent / "rte_release_review.json")
    args = parser.parse_args()

    for p in (paths.RTE_CUTOFFS, paths.RTE_ELD, paths.RTE_DECILES,
              paths.OPERATOR_BY_MUNICIPALITY):
        if not p.exists():
            raise SystemExit(f"missing: {p}\nRun `python paths.py` to check the wiring.")

    cutoffs = pd.read_csv(paths.RTE_CUTOFFS, dtype={"unit": str})
    eld = pd.read_csv(paths.RTE_ELD)
    deciles = pd.read_csv(paths.RTE_DECILES, dtype={"unit": str})
    operators = pd.read_csv(paths.OPERATOR_BY_MUNICIPALITY, dtype={"insee": str})

    bundle = {
        "metadata": METADATA,
        "departmental_cutoffs": table(cutoffs, "departmental_cutoffs"),
        "eld_within_department": table(eld, "eld_within_department"),
        "capacity_deciles": table(deciles, "capacity_deciles"),
        # Summarised rather than carried in full: 34,746 rows of operator labels
        # would bury the tables that need reading, and the point of this one is
        # the absence of any volume column.
        "operator_by_municipality": {
            "description": DESCRIPTIONS["operator_by_municipality"],
            "n_lignes": len(operators),
            "colonnes": list(operators.columns),
            "repartition": {str(k): int(v) for k, v
                            in operators["categorie"].value_counts().items()},
            "extrait_10_lignes": json.loads(
                operators.head(10).to_json(orient="records", force_ascii=False)),
            "note": "Fichier complet joint separement en CSV.",
        },
    }

    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, default=_plain),
        encoding="utf-8")

    blanked = int(cutoffs.kWp_cumulative.isna().sum())
    audited = int(cutoffs.loc[cutoffs.cutoff_label == "audit_reference",
                              "kWp_cumulative"].notna().sum())
    print(f"wrote {args.output}")
    for key in ("departmental_cutoffs", "eld_within_department", "capacity_deciles"):
        print(f"  {key:<24} {bundle[key]['n_lignes']:>6} rows, "
              f"{len(bundle[key]['colonnes'])} cols")
    print(f"  operator_by_municipality {len(operators):>6} rows, summarised")
    print(f"  blanked cells: {blanked} | audit_reference intact: {audited}/93")


if __name__ == "__main__":
    main()
