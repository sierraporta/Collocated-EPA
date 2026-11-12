#!/usr/bin/env python3
"""
04_resolution_limit.py — Is the record limited by the instrument or by the
reporting resolution?
==========================================================================

A relative CV is hard to read when a large share of the concurrent hours carry
the **same value** in both monitors (the case of ozone). A CV computed on such
data does not measure disagreement between instruments; it measures the
increment with which values are reported.

The paired difference is therefore also expressed in **units of the reporting
increment**, taken as the smallest positive step between distinct values in the
series. The measure is independent of unit and level and answers directly:

  sd in increments ~ 1 or less   the two monitors agree within the resolution
                                 of the published value. The record is limited
                                 by **resolution**; the instrument precision is
                                 better than the record can show and can only
                                 be bounded from above.

  sd in increments >> 1          the disagreement spans many increments. The
                                 record is limited by the **instrument**, and
                                 the CV of step 3 is not a quantisation
                                 artefact.

The same Appendix A section 4(c) floor as in step 3 is applied, because without
it the hours near zero dominate any ratio.

Usage
-----
  python3 04_resolution_limit.py
  python3 04_resolution_limit.py --all-types
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FLOOR = {"pm25": 2.0, "pm10": 3.0, "pmc": 3.0}
MIN_HOURS = 200
RESOLUTION_LIMITED_SD = 1.5  # sd in increments at or below which -> resolution


def reporting_step(*series: np.ndarray) -> float:
    """Smallest positive increment between the distinct observed values.

    It is measured on both monitors together: if one reports one more decimal
    than the other, the increment that limits the **comparison** is the coarse
    one, and taking the fine one would give a large disagreement in increments
    that is an artefact of the pairing.
    """
    v = np.unique(np.concatenate(series))
    d = np.diff(v)
    d = d[d > 0]
    return float(np.min(d)) if len(d) else np.nan


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(
        description="Paired difference in units of the reporting increment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--concurrency", type=Path,
                   default=root / "Results_Collocated" / "concurrency.parquet")
    p.add_argument("--hourly-dir", type=Path,
                   default=root / "Data_Collocated" / "hourly_pocs")
    p.add_argument("--designations", type=Path,
                   default=root / "Data_Collocated"
                   / "method_designations.csv")
    p.add_argument("--out", type=Path,
                   default=root / "Results_Collocated"
                   / "resolution_limit.parquet")
    p.add_argument("--arm", default="precision",
                   choices=["precision", "comparability", "reprocessing"])
    p.add_argument("--all-types", action="store_true",
                   help="include pairs with non-SLAMS monitors")
    a = p.parse_args(argv)

    F = pd.read_csv(a.designations, dtype=str)
    F["method_code"] = F.method_code.str.strip().str.zfill(3)
    fam = dict(zip(F.method_code, F.family))

    Q = pd.read_parquet(a.concurrency)
    Q = Q[Q.arm == a.arm]
    if not a.all_types:
        Q = Q[(Q.type_a == "SLAMS") & (Q.type_b == "SLAMS")]

    rows = []
    for (var, year), d in Q.groupby(["variable", "year"]):
        f = a.hourly_dir / f"hourly_pocs_{var}_{year}.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        floor = FLOOR.get(var)
        for r in d.itertuples():
            ca, cb = f"{r.site}#{r.poc_a}", f"{r.site}#{r.poc_b}"
            if ca not in H.columns or cb not in H.columns:
                continue
            x = H[ca].to_numpy("float64")
            y = H[cb].to_numpy("float64")
            m = np.isfinite(x) & np.isfinite(y)
            if floor is not None:
                m &= (x >= floor) & (y >= floor)
            if m.sum() < MIN_HOURS:
                continue
            xx, yy = x[m], y[m]
            h = reporting_step(xx, yy)
            if not np.isfinite(h) or h <= 0:
                continue
            dif = xx - yy
            st = np.abs(dif) / h
            rows.append(dict(
                variable=var, family=fam.get(r.method_a), site=r.site,
                year=year, n=int(m.sum()), step=h,
                level=float(np.mean((xx + yy) / 2)),
                sd_steps=float(np.std(dif, ddof=1) / h),
                p95_steps=float(np.percentile(st, 95)),
                # Share of hours by number of increments separating the two
                # values. The resolution-limited case has almost everything at
                # zero and one increment, nothing beyond two.
                frac_0=float(np.mean(st < 0.5)),
                frac_1=float(np.mean((st >= 0.5) & (st < 1.5))),
                frac_ge2=float(np.mean(st >= 1.5)),
                sd_absolute=float(np.std(dif, ddof=1))))
        print(f"  {var} {year}: {len(d)} pairs")

    R = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    R.to_parquet(a.out, index=False)

    print("\n=== paired difference in units of the reporting increment ===")
    T = R.groupby(["variable", "family"]).agg(
        pairs=("site", "size"), sites=("site", "nunique"),
        step=("step", "median"), level=("level", "median"),
        sd_steps=("sd_steps", "median"), p95_steps=("p95_steps", "median"),
        sd_abs=("sd_absolute", "median"),
        h_0=("frac_0", "median"), h_1=("frac_1", "median"),
        h_ge2=("frac_ge2", "median"))
    T["limited_by"] = np.where(T.sd_steps <= RESOLUTION_LIMITED_SD,
                               "resolution", "instrument")
    print(T.round(3).to_string())
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
