"""Suppress the cutoff cells that could disclose a single installation.

The cutoff file gives cumulative connected capacity at a grid of dates, so any
two consecutive cutoffs imply an increment. Five increments in the unprotected
file cover exactly one installation, four of them inside windows of one to eight
days, which would expose that system's capacity, its department and a near-exact
connection date. The collisions come from the two cutoff families overlapping:
offsets from the imagery date (``-3m`` ... ``+12m``) and month ends
(``me-3`` ... ``me+3``) land days apart whenever a flight falls near a month end.

**The rule.** Walk each unit's series in date order and blank a cutoff whose
increment from the last surviving one is at or below ``FLOOR``.

``FLOOR`` is the study perimeter, not a value fitted to the data. Every
installation in scope is at or below 36 kWp, so an increment of D kWp needs at
least ceil(D / 36) of them: an increment above 36 kWp cannot be a single system,
*as arithmetic*. The guarantee never mentions a count, so a reader holding only
the published file can check it.

**What is never blanked**, and why each is load-bearing:

- ``audit_reference`` is the audited value itself, the registry stopped at each
  municipality's own imagery date. It sums to the 3.90 GWp the paper reports and
  matches ``kWp_rte`` in the comparison tables to 0.05 kWp. It is also held out
  of the walk: it shares a date with ``t0``, so the two differ by which
  municipalities fall on a non-modal date, not by elapsed time, and feeding it
  into a date-ordered series would read a spatial difference as a temporal one.
- ``t0`` anchors every window; ``extraction`` closes the absolute transit bound;
  ``eoy`` carries the year-end catch-up test. All three are endpoints, so
  blanking them has no wider window to fall back on.
- ``me-3``, ``me-2``, ``me+1``, ``me+2``, ``me+3`` are the outer points of the
  dating sweep behind Figure S11.

``me+0`` and ``me-1`` are deliberately *not* protected. They are the two sweep
points that can sit days from ``t0``, and they are where the last residual thin
increments live. Letting them go closes the leak completely, and costs nothing:
every published quantity reproduces unchanged, because a blanked cell widens the
window rather than removing it.

Suppression is safe against reconstruction. The series is cumulative, so blanking
a cutoff does not hide the interval containing it -- the wider window spanning the
gap stays computable. Resolution degrades, nothing breaks.

    python apply_disclosure_control.py --dry-run
    python apply_disclosure_control.py
"""

import argparse
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

PERIMETER_KWP = 36.0

#: Held out of the walk entirely, and never blanked. See the module docstring.
REFERENCE = "audit_reference"

#: Blanking these would lose information no wider window can recover.
PROTECTED = {"t0", "extraction", "eoy",
             "me-3", "me-2", "me+1", "me+2", "me+3"}


def suppress(df, floor=PERIMETER_KWP):
    """Blank ``kWp_cumulative`` wherever an increment could cover one system.

    Two passes. The forward pass carries the last *surviving* value, so a run of
    thin increments merges into one wide one rather than each being tested
    against its immediate neighbour.

    The backward pass handles the step *into* a protected cutoff, which the
    forward pass cannot test because protected cells are kept whatever their
    increment. Where that step is thin, the predecessor is blanked instead, and
    the walk continues back until the step clears the floor.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["cutoff_date"])
    blanked = set()

    for _, g in df.groupby("unit", sort=False):
        g = g[g["cutoff_label"] != REFERENCE].sort_values("_date")
        g = g[g["kWp_cumulative"].notna()]     # cells past the extraction date
        index, values = list(g.index), g["kWp_cumulative"].tolist()
        labels = g["cutoff_label"].tolist()

        last = None
        for i, label in enumerate(labels):
            if last is None or label in PROTECTED:
                last = values[i]
            elif values[i] - last <= floor:
                blanked.add(index[i])
            else:
                last = values[i]

        for i, label in enumerate(labels):
            if label not in PROTECTED or i == 0:
                continue
            j = i - 1
            while j >= 0:
                if index[j] in blanked:
                    j -= 1
                elif values[i] - values[j] > floor or labels[j] in PROTECTED:
                    break
                else:
                    blanked.add(index[j])
                    j -= 1

    df.loc[sorted(blanked), "kWp_cumulative"] = pd.NA
    return df.drop(columns="_date"), sorted(blanked)


def residual(df, floor=PERIMETER_KWP):
    """Increments still small enough to cover one installation. Must be zero."""
    df = df.assign(_date=pd.to_datetime(df["cutoff_date"]))
    n = 0
    for _, g in df[df["cutoff_label"] != REFERENCE].groupby("unit"):
        g = g[g["kWp_cumulative"].notna()].sort_values("_date")
        v = g["kWp_cumulative"].tolist()
        n += sum(0 < b - a <= floor for a, b in zip(v, v[1:]))
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=paths.RTE_CUTOFFS)
    parser.add_argument("--output", type=Path, default=paths.RTE_CUTOFFS)
    parser.add_argument("--floor", type=float, default=PERIMETER_KWP)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"unit": str})

    # The installation count is what makes an increment legible as a singleton,
    # and nothing in the repository reads it: every quantity the temporal
    # notebook forms is a difference of cumulative capacities.
    if "n_installations_cumulative" in df.columns:
        df = df.drop(columns="n_installations_cumulative")
        print("dropped n_installations_cumulative")

    cells = int(df.loc[df.cutoff_label != REFERENCE, "kWp_cumulative"].notna().sum())
    out, blanked = suppress(df, floor=args.floor)

    print(f"floor              : {args.floor:.0f} kWp "
          f"(>= {int(args.floor // PERIMETER_KWP) + 1} installations per increment)")
    print(f"series cells       : {cells}")
    print(f"blanked            : {len(blanked)} ({100 * len(blanked) / cells:.1f}%)")
    print(f"residual <= floor  : {residual(out, args.floor)}")
    print(f"{REFERENCE} rows intact: "
          f"{int(out.loc[out.cutoff_label == REFERENCE, 'kWp_cumulative'].notna().sum())}/93")

    by_label = out.loc[blanked, "cutoff_label"].value_counts()
    print("by label           : " + ", ".join(f"{k} {v}" for k, v in by_label.items()))

    if args.dry_run:
        print("dry run, nothing written")
        return

    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
