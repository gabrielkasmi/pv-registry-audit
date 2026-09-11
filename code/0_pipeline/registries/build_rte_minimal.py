"""Rebuild ``rte_minimal.json`` from the released CSVs.

``rte_minimal.json`` is a convenience wrapper: the three departmental tables in a
single record, with the metadata a reader needs to interpret them. It carries no
information the CSVs do not.

That makes it derived, and derived files must be regenerated rather than
maintained. The first version of this deposit shipped a ``rte_minimal.json``
built before the disclosure control existed, so it still carried the cumulative
values the control withholds and the installation counts it drops -- which would
have voided the control through the back door. This script exists so that cannot
recur: it rewrites the wrapper from whatever the CSVs currently hold.

``apply_disclosure_control.py`` calls it as its last step. Run it directly only
if a CSV in ``rte_derived/`` changed for some other reason.

    python build_rte_minimal.py
"""

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

MINIMAL = paths.RTE_DERIVED / "rte_minimal.json"


def build():
    """Rewrite the wrapper, keeping the descriptive blocks and replacing the data."""
    doc = json.loads(MINIMAL.read_text(encoding="utf-8"))

    cutoffs = pd.read_csv(paths.RTE_CUTOFFS, dtype={"unit": str})
    eld = pd.read_csv(paths.RTE_ELD)
    deciles = pd.read_csv(paths.RTE_DECILES, dtype={"unit": str})

    def records(df):
        return json.loads(df.to_json(orient="records", force_ascii=False))

    doc["departmental_cutoffs"]["data"] = records(cutoffs)
    doc["eld_within_department"]["data"] = records(eld)
    doc["capacity_deciles"]["data"] = records(deciles)

    # The field dictionary described a column the control has since removed, and
    # said nothing about withheld cells. Keep it truthful to the CSV.
    fields = doc["departmental_cutoffs"].get("fields", {})
    fields.pop("n_installations_cumulative", None)
    fields["kWp_cumulative"] = (
        "registry capacity connected on or before cutoff_date, kWp; empty where "
        "withheld by the disclosure control, or where the cutoff falls beyond the "
        "registry extraction date"
    )
    doc["departmental_cutoffs"]["fields"] = fields

    doc["metadata"]["disclosure_control"] = (
        "The installation count is not published for the temporal series, and any "
        "cumulative value whose increment from the previous cutoff was at or below "
        "36 kWp is withheld. The study perimeter caps every installation at 36 kWp, "
        "so any interval between two published cutoffs necessarily covers at least "
        "two installations; verified across all 10,633 pairs of cutoffs, the "
        "smallest strictly positive difference being 36.1 kWp. The audited value "
        "(audit_reference) is never withheld. Applied by "
        "code/0_pipeline/registries/apply_disclosure_control.py."
    )

    MINIMAL.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    withheld = int(cutoffs.kWp_cumulative.isna().sum())
    audited = int(cutoffs.loc[cutoffs.cutoff_label == "audit_reference",
                              "kWp_cumulative"].notna().sum())
    return {"rows": len(cutoffs), "withheld": withheld, "audit_reference": audited,
            "has_counts": "n_installations_cumulative" in cutoffs.columns}


def main():
    r = build()
    print(f"wrote {MINIMAL}")
    print(f"  cutoffs rows {r['rows']} | withheld cells {r['withheld']} | "
          f"audit_reference {r['audit_reference']}/93 | count column {r['has_counts']}")
    assert not r["has_counts"], "the count column leaked into the wrapper"
    assert r["audit_reference"] == 93, "audit_reference incomplete"


if __name__ == "__main__":
    main()
