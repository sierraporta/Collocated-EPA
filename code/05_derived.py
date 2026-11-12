#!/usr/bin/env python3
"""
05_derived.py — What the disagreement does to the quantities actually used
==========================================================================

Single hours are rarely used. What is used is the annual mean, the 98th
percentile of daily values, the daily maximum and exceedance counts, and with
them a county is declared in or out of attainment and the public is shown an
air quality index.

This step asks whether **two monitors of the same designation, sampling the
same air, give the same answer**. If not, the answer depends on which of the
two was installed, and the size of that dependence can be measured.

All quantities are computed on the **same set of hours** in both monitors.
Letting each monitor use the hours it has would mix instrument disagreement
with differences in coverage, which is a separate matter.

The results are not official design values: those need three years,
completeness rules and the data substitution performed by AQS. They are
one-year analogues and are reported as such. The quantity of interest is not
the value itself but the **difference between two monitors that should
agree**.

Usage
-----
  python3 05_derived.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FLOOR = {"pm25": 2.0, "pm10": 3.0, "pmc": 3.0}
MIN_HOURS_PER_DAY = 18
MIN_DAYS = 200
# Thresholds with a regulatory or public-communication meaning. The annual
# PM2.5 standard was lowered from 12 to 9 ug/m3 in 2024; both are reported
# because the record spans the revision.
THRESHOLDS = {"pm25": {"daily_35": 35.0, "annual_12": 12.0, "annual_9": 9.0},
              "pm10": {"daily_150": 150.0},
              "o3": {"8h_0.070": 0.070}}
# The ozone threshold applies to the maximum daily 8-hour average (MDA8), not
# to the 24-hour mean. Rules of 40 CFR 50 Appendix U, simplified to one year:
# an 8-hour mean is valid with at least 6 hours; the daily MDA8 is taken over
# the 17 windows starting from 07:00 to 23:00 local time and is valid with at
# least 13 of them; it is truncated to 0.001 ppm.
O3_NAAQS = 0.070


def mda8(x: np.ndarray, y: np.ndarray, hours: pd.DatetimeIndex,
         offset: int) -> pd.DataFrame:
    """Daily MDA8 of both monitors over the same hours."""
    local = hours + pd.Timedelta(hours=int(offset))
    ok = np.isfinite(x) & np.isfinite(y)
    s = pd.DataFrame({"x": np.where(ok, x, np.nan),
                      "y": np.where(ok, y, np.nan)}, index=local)
    s = s[~s.index.duplicated()].asfreq("h")
    # forward window: the mean starting at hour t covers t..t+7
    m8 = s.rolling(8, min_periods=6).mean().shift(-7)
    m8 = m8[(m8.index.hour >= 7)]
    g = m8.groupby(m8.index.floor("D"))
    n = g.x.count()
    keep = n[n >= 13].index
    out = pd.DataFrame({"x": g.x.max().reindex(keep),
                        "y": g.y.max().reindex(keep)}).dropna()
    return np.floor(out * 1000 + 1e-9) / 1000


def daily_table(x: np.ndarray, y: np.ndarray, hours: pd.DatetimeIndex,
                offset: int) -> pd.DataFrame:
    """Daily means and maxima in local time, over the same hours."""
    local = hours + pd.Timedelta(hours=int(offset))
    ok = np.isfinite(x) & np.isfinite(y)
    df = pd.DataFrame({"day": local.floor("D"), "x": x, "y": y})[ok]
    g = df.groupby("day")
    n = g.size()
    keep = n[n >= MIN_HOURS_PER_DAY].index
    out = pd.DataFrame({
        "mean_x": g.x.mean().reindex(keep), "mean_y": g.y.mean().reindex(keep),
        "max_x": g.x.max().reindex(keep), "max_y": g.y.max().reindex(keep)})
    return out.dropna()


def rel(a: float, b: float) -> float:
    """Relative difference in percent of the mean of the two."""
    m = (a + b) / 2.0
    return (a - b) / m * 100.0 if m else np.nan


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(
        description="Propagation of the disagreement to derived quantities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--concurrency", type=Path,
                   default=root / "Results_Collocated" / "concurrency.parquet")
    p.add_argument("--columns", type=Path,
                   default=root / "Data_Collocated" / "poc_columns.parquet")
    p.add_argument("--hourly-dir", type=Path,
                   default=root / "Data_Collocated" / "hourly_pocs")
    p.add_argument("--designations", type=Path,
                   default=root / "Data_Collocated"
                   / "method_designations.csv")
    p.add_argument("--out", type=Path,
                   default=root / "Results_Collocated" / "derived.parquet")
    p.add_argument("--arm", default="precision",
                   choices=["precision", "comparability", "reprocessing"])
    a = p.parse_args(argv)

    F = pd.read_csv(a.designations, dtype=str)
    F["method_code"] = F.method_code.str.strip().str.zfill(3)
    fam = dict(zip(F.method_code, F.family))

    Q = pd.read_parquet(a.concurrency)
    Q = Q[(Q.arm == a.arm) & (Q.type_a == "SLAMS") & (Q.type_b == "SLAMS")]
    I = pd.read_parquet(a.columns).set_index(
        ["variable", "year", "site", "poc"])

    rows = []
    for (var, year), d in Q.groupby(["variable", "year"]):
        f = a.hourly_dir / f"hourly_pocs_{var}_{year}.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        hours = pd.DatetimeIndex(H.index)
        for r in d.itertuples():
            ca, cb = f"{r.site}#{r.poc_a}", f"{r.site}#{r.poc_b}"
            if ca not in H.columns or cb not in H.columns:
                continue
            try:
                off = I.loc[(var, year, r.site, r.poc_a), "gmt_offset"]
                off = int(off) if pd.notna(off) else 0
            except KeyError:
                off = 0
            T = daily_table(H[ca].to_numpy("float64"),
                            H[cb].to_numpy("float64"), hours, off)
            if len(T) < MIN_DAYS:
                continue
            rec = dict(variable=var, year=year, site=r.site,
                       family=fam.get(r.method_a), method=r.method_a,
                       n_days=len(T))
            # Annual mean: the statistic of the annual PM2.5 standard.
            ax, ay = float(T.mean_x.mean()), float(T.mean_y.mean())
            rec.update(annual_x=ax, annual_y=ay, annual_abs=ax - ay,
                       annual_rel=rel(ax, ay))
            # 98th percentile of daily values: the 24-hour standard statistic.
            px = float(np.percentile(T.mean_x, 98))
            py = float(np.percentile(T.mean_y, 98))
            rec.update(p98_x=px, p98_y=py, p98_abs=px - py, p98_rel=rel(px, py))
            # Hourly maximum of the day, averaged over the year: closest to the
            # public index and the most sensitive to single-hour noise.
            mx, my = float(T.max_x.mean()), float(T.max_y.mean())
            rec.update(dmax_x=mx, dmax_y=my, dmax_rel=rel(mx, my))
            if var == "o3":
                M = mda8(H[ca].to_numpy("float64"), H[cb].to_numpy("float64"),
                         hours, off)
                if len(M) >= 4:
                    fx = float(np.sort(M.x.to_numpy())[-4])
                    fy = float(np.sort(M.y.to_numpy())[-4])
                    rec.update(n_mda8=len(M), mda8_4th_x=fx, mda8_4th_y=fy,
                               mda8_4th_abs=fx - fy,
                               mda8_mean_abs=float((M.x - M.y).mean()),
                               disagree_mda8_4th=int((fx > O3_NAAQS)
                                                     != (fy > O3_NAAQS)))
                    nx = int((M.x > O3_NAAQS).sum())
                    ny = int((M.y > O3_NAAQS).sum())
                    rec.update(**{"n_8h_0.070_x": nx, "n_8h_0.070_y": ny,
                                  "disagree_8h_0.070": int(nx != ny)})
            # Counts of days above each threshold.
            for name, thr in THRESHOLDS.get(var, {}).items():
                if name.startswith("8h"):
                    continue          # already counted on the MDA8
                if name.startswith("daily"):
                    nx = int((T.mean_x > thr).sum())
                    ny = int((T.mean_y > thr).sum())
                else:
                    nx = int(ax > thr)
                    ny = int(ay > thr)
                rec[f"n_{name}_x"] = nx
                rec[f"n_{name}_y"] = ny
                rec[f"disagree_{name}"] = int(nx != ny)
            rows.append(rec)
        print(f"  {var} {year}: {len(d)} pairs")

    R = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    R.to_parquet(a.out, index=False)

    print(f"\npair-years with {MIN_DAYS} or more days: {len(R)}")
    for var, g in R.groupby("variable"):
        print(f"\n=== {var} ===")
        T = g.groupby("family").agg(
            pairs=("site", "size"), sites=("site", "nunique"),
            level=("annual_x", "median"),
            annual_abs=("annual_abs", lambda s: np.median(np.abs(s))),
            annual_p95=("annual_abs", lambda s: np.percentile(np.abs(s), 95)),
            annual_rel=("annual_rel", lambda s: np.median(np.abs(s))),
            p98_abs=("p98_abs", lambda s: np.median(np.abs(s))),
            p98_rel=("p98_rel", lambda s: np.median(np.abs(s))),
            dmax_rel=("dmax_rel", lambda s: np.median(np.abs(s))))
        print(T.round(3).to_string())
        for name in THRESHOLDS.get(var, {}):
            col = f"disagree_{name}"
            if col in g.columns:
                k = int(g[col].sum())
                print(f"  pairs that disagree on threshold {name}: "
                      f"{k} of {len(g)} ({k/len(g):.0%})")

    print("\nAll values are one-year analogues, not design values, which need "
          "three years\nand the AQS completeness and substitution rules.")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
