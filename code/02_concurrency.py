#!/usr/bin/env python3
"""
02_concurrency.py — Concurrent hours, and whether a pair is two instruments
===========================================================================

Two things must be known before any agreement statistic is computed.

**The actual sample size.** Two monitors with 8000 observed hours each may
overlap in 8000 hours or in 2000. What counts is the number of hours in which
**both** report. Pairs with fewer than MIN_CONCURRENT concurrent hours are
dropped.

**Whether the pair is a replicate.** In 2024 EPA recalculated all unaligned
hourly T640/T640X PM2.5 data with the Network Data Alignment and added the
recalculated series **as a new POC at the same site** (docket
EPA-HQ-OAR-2023-0642). Such pairs are the same hours of the same instrument,
one of them transformed by a published equation: they measure the equation,
not agreement between instruments.

The distinction is made from the data rather than from metadata. Two physical
instruments give a high but imperfect correlation, with a lower tail, and lose
different hours. A deterministic transformation gives a correlation pinned at
one. The reprocessing pairs serve as a positive control: their signature shows
what the degenerate case looks like.

Usage
-----
  python3 02_concurrency.py
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# Below this a pair supports no agreement statistic.
MIN_CONCURRENT = 500


def designation_map(path: Path) -> dict[str, str]:
    D = pd.read_csv(path, dtype=str)
    D["method_code"] = D.method_code.str.strip().str.zfill(3)
    return dict(zip(D.method_code, D.designation))


def classify(mc_a: str, mc_b: str, des: dict) -> str:
    """Three categories, grouping by equivalent method designation."""
    if mc_a == mc_b:
        return "precision"
    da, db = des.get(mc_a), des.get(mc_b)
    if da is not None and da == db:
        return "reprocessing"
    return "comparability"


def pair_stats(x: np.ndarray, y: np.ndarray) -> dict | None:
    pa, pb = np.isfinite(x), np.isfinite(y)
    both = pa & pb
    n = int(both.sum())
    if n < MIN_CONCURRENT:
        return None
    xx, yy = x[both], y[both]
    r = (float(np.corrcoef(xx, yy)[0, 1])
         if np.std(xx) > 0 and np.std(yy) > 0 else np.nan)
    union = int((pa | pb).sum())
    d = xx - yy
    return dict(
        n_concurrent=n,
        n_union=union,
        disjoint_frac=float((union - n) / union) if union else np.nan,
        r=r,
        mean_a=float(np.mean(xx)), mean_b=float(np.mean(yy)),
        sd_diff=float(np.std(d, ddof=1)),
        mean_diff=float(np.mean(d)),
        p95_abs_diff=float(np.percentile(np.abs(d), 95)))


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(
        description="Concurrent hours and replicate diagnostic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--columns", type=Path,
                   default=root / "Data_Collocated" / "poc_columns.parquet")
    p.add_argument("--hourly-dir", type=Path,
                   default=root / "Data_Collocated" / "hourly_pocs")
    p.add_argument("--designations", type=Path,
                   default=root / "Data_Collocated"
                   / "method_designations.csv")
    p.add_argument("--monitors", type=Path,
                   default=root / "Data_Collocated" / "inputs" / "monitors_subset.csv")
    p.add_argument("--out", type=Path,
                   default=root / "Results_Collocated"
                   / "concurrency.parquet")
    a = p.parse_args(argv)

    des = designation_map(a.designations)
    I = pd.read_parquet(a.columns)
    I = I[I.method_code.notna()]

    mon = pd.read_csv(a.monitors, dtype=str)
    mon["site"] = (mon["State Code"].str.zfill(2) + "_"
                   + mon["County Code"].str.zfill(3) + "_"
                   + mon["Site Number"].str.zfill(4))
    PARAM = {"44201": "o3", "42401": "so2", "42101": "co", "42602": "no2",
             "88101": "pm25", "88502": "pm25n", "81102": "pm10",
             "86101": "pmc"}
    mon["variable"] = mon["Parameter Code"].map(PARAM)
    mon["poc"] = pd.to_numeric(mon["POC"], errors="coerce")
    # The monitor listing has more than one row per site, parameter and POC;
    # without dropping duplicates the index is not unique and `.get()` returns
    # a Series instead of a value.
    mon = mon.drop_duplicates(["site", "variable", "poc"])
    mtype = mon.set_index(["site", "variable", "poc"])["Monitor Type"]

    rows = []
    for (var, year), d in I.groupby(["variable", "year"]):
        f = a.hourly_dir / f"hourly_pocs_{var}_{year}.parquet"
        if not f.exists():
            print(f"  missing {f.name}, skipped")
            continue
        H = pd.read_parquet(f)
        for site, g in d.groupby("site"):
            recs = g.sort_values("poc").to_dict("records")
            if len(recs) < 2:
                continue
            for ra, rb in itertools.combinations(recs, 2):
                ca, cb = ra["column"], rb["column"]
                if ca not in H.columns or cb not in H.columns:
                    continue
                st = pair_stats(H[ca].to_numpy("float64"),
                                H[cb].to_numpy("float64"))
                if st is None:
                    continue
                rows.append(dict(
                    variable=var, year=year, site=site,
                    poc_a=ra["poc"], poc_b=rb["poc"],
                    method_a=ra["method_code"], method_b=rb["method_code"],
                    arm=classify(ra["method_code"], rb["method_code"], des),
                    type_a=mtype.get((site, var, ra["poc"])),
                    type_b=mtype.get((site, var, rb["poc"])),
                    **st))
        print(f"  {var} {year}: {H.shape[1]} columns")

    Q = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Q.to_parquet(a.out, index=False)
    Q["both_slams"] = (Q.type_a == "SLAMS") & (Q.type_b == "SLAMS")

    print(f"\npairs with {MIN_CONCURRENT} or more concurrent hours: {len(Q)}")
    print("\n=== concurrent hours and replicate diagnostic, by category ===")
    T = Q.groupby("arm").agg(
        pairs=("site", "size"), sites=("site", "nunique"),
        conc_hours_median=("n_concurrent", "median"),
        r_median=("r", "median"), r_p05=("r", lambda s: s.quantile(0.05)),
        r_max=("r", "max"),
        disjoint_median=("disjoint_frac", "median"))
    print(T.round(4).to_string())

    print("\n=== pairs that are practically deterministic (r > 0.9999) ===")
    print(Q.groupby("arm").r.apply(
        lambda s: f"{int((s > 0.9999).sum())} of {int(s.notna().sum())}"
        ).to_string())

    print("\n=== pairs with both monitors SLAMS ===")
    S = Q[Q.both_slams]
    print(S.groupby(["arm", "variable"]).agg(
        pairs=("site", "size"), sites=("site", "nunique"),
        conc_hours_median=("n_concurrent", "median"),
        r_median=("r", "median"), r_p05=("r", lambda s: s.quantile(0.05)),
        disjoint_median=("disjoint_frac", "median")).round(4).to_string())
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
