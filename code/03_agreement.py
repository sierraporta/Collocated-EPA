#!/usr/bin/env python3
"""
03_agreement.py — Hourly and daily agreement of two identical instruments
=========================================================================

Regulatory collocated precision is assessed **at 24 hours**: the PM2.5
quality-control samplers required by 40 CFR 58 Appendix A are filter based and
run one day in twelve, and for gases precision is assessed with one-point
checks against a calibration standard rather than against a second analyser.
The hourly record therefore has no published precision figure. It is computed
here from the duplicate monitors the network already operates.

The statistic is the regulatory one (Appendix A section 4.2.1; Eqs. 1-2 of the
article). For each pair of concurrent values the relative percent difference is

    d_i = (X_i - Y_i) / ((X_i + Y_i)/2) * 100

and precision is summarised by the upper bound of the coefficient of variation

    CV_ub = sqrt( [ sum(d_i^2) - (sum d_i)^2 / k ] / (2 * chi2_{0.1, k-1}) )

with k valid pairs. The factor 2 accounts for both values carrying error. CV_ub
does not depend on the order of X and Y (swapping them only flips the sign of
every d_i), so no monitor has to be designated as primary; the mean difference
does depend on the order and is reported separately.

**The concentration floor matters.** Appendix A section 4(c) admits a pair only
when both values are at or above a minimum: 2 ug/m3 for PM2.5 and 3 ug/m3 for
low-volume PM10. At hourly resolution the denominator of d_i approaches zero at
low concentrations, and relative differences become uninformative. No minimum
is specified for gases; ozone is analysed without one.

For the precision category, the script also writes the sensitivity of the
family ordering to the estimator and to the floor (Methods, section 3.2):
CV_ub with floors of 0, 2 and 5 ug/m3, a plain CV sqrt(sum d^2 / 2k), a robust
CV from the median absolute deviation of d, and the median |d|.

Usage
-----
  python3 03_agreement.py                               # precision, SLAMS
  python3 03_agreement.py --arm comparability           # different designations
  python3 03_agreement.py --arm comparability --all-types
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Minima of Appendix A section 4(c). Gases have none.
FLOOR = {"pm25": 2.0, "pm10": 3.0, "pmc": 3.0, "no2": None, "o3": None,
         "so2": None, "co": None}
FLOOR_SOURCE = {"pm25": "40 CFR 58 App. A 4(c)(5)",
                "pm10": "40 CFR 58 App. A 4(c)(4)",
                "pmc": "40 CFR 58 App. A 4(c)(4), by analogy"}
SENSITIVITY_FLOORS = (0.0, 2.0, 5.0)  # ug/m3, PM2.5 sensitivity analysis
MIN_PAIRS = 30          # minimum k for the chi-square bound to be meaningful
MIN_HOURS_PER_DAY = 18  # usual completeness rule for a daily mean


def chi2_ppf(q: float, df: int) -> float:
    """Chi-square quantile; scipy if available, otherwise Wilson-Hilferty."""
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(q, df))
    except ImportError:
        # Wilson-Hilferty: accurate to third order and sufficient for
        # df >= 30, the minimum required here anyway.
        from math import sqrt
        z = _norm_ppf(q)
        return df * (1.0 - 2.0 / (9.0 * df) + z * sqrt(2.0 / (9.0 * df))) ** 3


def _norm_ppf(q: float) -> float:
    """Inverse standard normal (Acklam's algorithm, error < 1.15e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, pu = 0.02425, 1 - 0.02425
    if q < pl:
        r = (-2 * np.log(q)) ** 0.5
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
               ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q > pu:
        r = (-2 * np.log(1 - q)) ** 0.5
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    r = q - 0.5
    s = r * r
    return (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r / \
           (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1)


def family_map(path: Path) -> dict[str, str]:
    """Method code -> instrument family.

    Grouping by pollutant alone would mix instruments whose hourly precision
    differs by almost an order of magnitude (T640 optical and beta attenuation
    monitors for PM2.5), so all results are reported by family.
    """
    D = pd.read_csv(path, dtype=str)
    D["method_code"] = D.method_code.str.strip().str.zfill(3)
    return dict(zip(D.method_code, D.family))


def rpd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Relative percent difference (Eq. 1 of the article)."""
    m = (x + y) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m != 0, (x - y) / m * 100.0, np.nan)


def cv_upper(d: np.ndarray) -> tuple[float, int]:
    """Upper bound of the CV (Eq. 2 of the article). Symmetric in X and Y."""
    d = d[np.isfinite(d)]
    k = len(d)
    if k < MIN_PAIRS:
        return np.nan, k
    num = float(np.sum(d ** 2) - np.sum(d) ** 2 / k)
    return float(np.sqrt(num / (2.0 * chi2_ppf(0.10, k - 1)))), k


def alternative_estimators(d: np.ndarray) -> dict:
    """Plain CV, robust CV from the MAD, and median |d| of one pair."""
    d = d[np.isfinite(d)]
    k = len(d)
    if k < MIN_PAIRS:
        return dict(cv_plain=np.nan, cv_mad=np.nan, median_abs_d=np.nan)
    mad = float(np.median(np.abs(d - np.median(d))))
    return dict(cv_plain=float(np.sqrt(np.sum(d ** 2) / (2.0 * k))),
                cv_mad=1.4826 * mad / np.sqrt(2.0),
                median_abs_d=float(np.median(np.abs(d))))


def daily_means(x: np.ndarray, y: np.ndarray, hours: pd.DatetimeIndex,
                offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Daily means in local standard time, requiring completeness in both.

    Days are local, not UTC, because the local day is the unit of the
    regulatory statistic. A day enters only if both monitors have at least
    MIN_HOURS_PER_DAY hours, and both means use the same set of hours, so that
    no difference arises from averaging different hours.
    """
    local = hours + pd.Timedelta(hours=int(offset))
    day = local.floor("D")
    both = np.isfinite(x) & np.isfinite(y)
    df = pd.DataFrame({"day": day, "x": x, "y": y, "ok": both})
    g = df[df.ok].groupby("day")
    n = g.size()
    keep = n[n >= MIN_HOURS_PER_DAY].index
    a = g.x.mean().reindex(keep).to_numpy()
    b = g.y.mean().reindex(keep).to_numpy()
    return a, b


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(
        description="Hourly and daily agreement of duplicate monitors.",
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
    p.add_argument("--out-dir", type=Path,
                   default=root / "Results_Collocated")
    p.add_argument("--arm", default="precision",
                   choices=["precision", "comparability", "reprocessing"])
    p.add_argument("--all-types", action="store_true",
                   help="include pairs with non-SLAMS monitors (special "
                        "purpose, tribal, industrial); outputs get the suffix "
                        "_alltypes")
    p.add_argument("--no-floor", action="store_true",
                   help="ignore the Appendix A section 4(c) minimum")
    a = p.parse_args(argv)
    a.slams_only = not a.all_types
    tag = "" if a.slams_only else "_alltypes"

    fam = family_map(a.designations)
    Q = pd.read_parquet(a.concurrency)
    Q = Q[Q.arm == a.arm]
    if a.slams_only:
        Q = Q[(Q.type_a == "SLAMS") & (Q.type_b == "SLAMS")]
    I = pd.read_parquet(a.columns).set_index(["variable", "year", "site", "poc"])

    rows, sens, hourly_d = [], [], []
    for (var, year), d in Q.groupby(["variable", "year"]):
        f = a.hourly_dir / f"hourly_pocs_{var}_{year}.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        hours = pd.DatetimeIndex(H.index)
        floor = None if a.no_floor else FLOOR.get(var)
        for r in d.itertuples():
            ca, cb = f"{r.site}#{r.poc_a}", f"{r.site}#{r.poc_b}"
            if ca not in H.columns or cb not in H.columns:
                continue
            x = H[ca].to_numpy("float64")
            y = H[cb].to_numpy("float64")
            try:
                off = I.loc[(var, year, r.site, r.poc_a), "gmt_offset"]
                off = int(off) if pd.notna(off) else 0
            except KeyError:
                off = 0

            both = np.isfinite(x) & np.isfinite(y)
            if floor is not None:
                both &= (x >= floor) & (y >= floor)
            dh = rpd(x[both], y[both])
            cvh, kh = cv_upper(dh)
            # Fraction of hours in which both monitors report the **same**
            # value. If large, the CV reflects the reporting increment rather
            # than disagreement between instruments.
            tie = float(np.mean(dh == 0)) if kh else np.nan

            xa, ya = daily_means(x, y, hours, off)
            if floor is not None and len(xa):
                kp = (xa >= floor) & (ya >= floor)
                xa, ya = xa[kp], ya[kp]
            dd = rpd(xa, ya) if len(xa) else np.array([])
            cvd, kd = cv_upper(dd)

            lvl = float(np.nanmean((x[both] + y[both]) / 2)) if both.any() else np.nan
            rows.append(dict(
                variable=var, year=year, site=r.site,
                poc_a=r.poc_a, poc_b=r.poc_b, method=r.method_a,
                n_hourly=kh, cv_hourly=cvh,
                n_daily=kd, cv_daily=cvd,
                ratio=cvh / cvd if (np.isfinite(cvh) and np.isfinite(cvd)
                                    and cvd > 0) else np.nan,
                mean_level=lvl, tie_frac=tie,
                family=fam.get(r.method_a),
                bias_hourly=float(np.nanmean(dh)) if kh else np.nan))
            if kh:
                hourly_d.append(pd.DataFrame(dict(
                    variable=var, year=year, site=r.site, d=dh,
                    level=(x[both] + y[both]) / 2,
                    hour=(hours[both] + pd.Timedelta(hours=off)).hour,
                    month=(hours[both] + pd.Timedelta(hours=off)).month)))

            # Sensitivity to estimator and floor (precision, PM2.5 only).
            if a.arm == "precision" and var == "pm25" and not a.no_floor:
                fin = np.isfinite(x) & np.isfinite(y)
                s = dict(variable=var, year=year, site=r.site,
                         poc_a=r.poc_a, poc_b=r.poc_b,
                         family=fam.get(r.method_a))
                for fl in SENSITIVITY_FLOORS:
                    m = fin & (x >= fl) & (y >= fl)
                    s[f"cv_ub_floor{fl:g}"] = cv_upper(rpd(x[m], y[m]))[0]
                s.update(alternative_estimators(dh))
                sens.append(s)
        print(f"  {var} {year}: {len(d)} pairs")

    R = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    R.to_parquet(a.out_dir / f"agreement_{a.arm}{tag}.parquet", index=False)
    D = (pd.concat(hourly_d, ignore_index=True) if hourly_d
         else pd.DataFrame(columns=["variable", "d", "level", "hour", "month"]))

    print(f"\ncategory: {a.arm}"
          + ("  (SLAMS only)" if a.slams_only else "  (all monitor types)")
          + ("  NO floor" if a.no_floor else ""))
    print("\n=== hourly vs daily CV, BY INSTRUMENT FAMILY ===")
    gg = R[R.cv_hourly.notna() & R.cv_daily.notna()]
    for var, g in gg.groupby("variable"):
        fl = FLOOR.get(var)
        src = FLOOR_SOURCE.get(var, "no floor")
        print(f"\n{var}   floor {fl} ({src})")
        T = g.groupby("family").agg(
            pairs=("site", "size"), sites=("site", "nunique"),
            level=("mean_level", "median"),
            cv_h=("cv_hourly", "median"), cv_d=("cv_daily", "median"),
            ratio=("ratio", "median"), ties=("tie_frac", "median"))
        print(T.round(3).to_string())
    print(f"\n'ties' is the fraction of hours in which both monitors report the "
          f"same value.\nFor errors independent between hours the hourly/daily "
          f"ratio would be {np.sqrt(24):.2f}.")

    if sens:
        S = pd.DataFrame(sens)
        S.to_parquet(a.out_dir / f"agreement_{a.arm}{tag}_sensitivity.parquet",
                     index=False)
        print("\n=== PM2.5: sensitivity to estimator and floor "
              "(family medians, %) ===")
        cols = [c for c in S.columns if c.startswith("cv_") or c == "median_abs_d"]
        print(S.groupby("family")[cols].median().round(2).to_string())
        try:
            from scipy.stats import spearmanr
            ok = S.cv_ub_floor2.notna() & S.cv_plain.notna()
            rho = spearmanr(S.cv_ub_floor2[ok], S.cv_plain[ok]).correlation
            print(f"rank correlation across pairs, CV_ub vs plain CV: {rho:.3f}")
        except ImportError:
            pass

    if not D.empty:
        D.to_parquet(a.out_dir / f"agreement_{a.arm}{tag}_hourly.parquet",
                     index=False)
    print(f"\n-> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
