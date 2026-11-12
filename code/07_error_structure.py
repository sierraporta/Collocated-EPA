#!/usr/bin/env python3
"""
07_error_structure.py — Error structure of each instrument family
=================================================================

Steps 3 and 4 give one agreement figure per pair. This step asks what shape
the disagreement has, because that shape decides what reaches each derived
statistic and what is inherited by anyone using the record as a reference.

Four analyses, all on the paired difference in absolute units (ug/m3 for PM,
ppb for ozone) over concurrent hours, with no concentration floor: in absolute
units a low value is not a problem.

1. Allan deviation of the difference against averaging time tau, in
   non-overlapping blocks with at least 75% of hours. White noise falls as
   tau^-1/2; a persistent component flattens it. Divided by sqrt(2) to express
   it per instrument, assuming two equivalent instruments.

2. Error model against concentration: standard deviation of the difference
   (after removing the pair's mean bias) by concentration bin, and a fit of
   sigma^2(c) = a^2 + (b c)^2, an additive and a proportional part.

3. Decomposition of the hourly difference into the pair's mean offset, a slow
   component (168-h moving mean minus the offset) and a fast component (the
   remainder). It shows which part reaches the annual mean.

4. Bound for sensor evaluation. If the reference carries an independent error
   sigma_ref, the observed RMSE of a sensor satisfies
   RMSE_obs^2 = RMSE_sensor^2 + sigma_ref^2, and a perfect sensor would reach
   at most R^2_max = 1 - sigma_ref^2 / var(reference). Computed per pair at
   1 h and at 24 h.

For ozone the Sheppard correction is added: two series rounded to an
increment h, without instrument error, would give a difference with standard
deviation h / sqrt(6). Whatever exceeds that belongs to the instrument.

All family summaries carry a 95% interval from a bootstrap over sites (pairs
at the same site are not independent).

Usage
-----
  python3 07_error_structure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TAUS = [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 168, 336]
MIN_BLOCK_COVER = 0.75
SLOW_WINDOW = 168           # hours, one week
BINS = {"pm25": [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 50],
        "pm10": [0, 5, 10, 15, 20, 30, 40, 60, 100],
        "o3": [0, 10, 20, 30, 40, 50, 60, 80]}
SCALE = {"pm25": 1.0, "pm10": 1.0, "o3": 1000.0}   # ozone in ppb
MIN_BIN_HOURS = 50
MIN_SITES_FAMILY = 4
B = 2000


def allan(d: pd.Series, tau: int) -> tuple[float, int]:
    """Non-overlapping Allan deviation of an hourly series with gaps."""
    blk = np.arange(len(d)) // tau
    g = d.groupby(blk)
    m = g.mean().where(g.count() >= MIN_BLOCK_COVER * tau)
    dd = m.diff().dropna()
    if len(dd) < 10:
        return np.nan, len(dd)
    return float(np.sqrt(0.5 * np.mean(dd ** 2))), len(dd)


def pair_analysis(x: pd.Series, y: pd.Series, var: str) -> dict:
    s = SCALE[var]
    ok = x.notna() & y.notna()
    x = (x * s).where(ok)
    y = (y * s).where(ok)
    d = x - y
    lvl = (x + y) / 2
    out = {"allan": [], "bins": []}

    for tau in TAUS:
        a, k = allan(d, tau)
        out["allan"].append(dict(tau=tau, adev=a / np.sqrt(2), n_blocks=k))

    mu = float(d.mean())
    resid = d - mu
    frame = pd.DataFrame({"r": resid[ok], "l": lvl[ok]})
    frame["bin"] = pd.cut(frame.l, BINS[var])
    for iv, g in frame.groupby("bin", observed=True):
        if len(g) >= MIN_BIN_HOURS:
            out["bins"].append(dict(lo=iv.left, hi=iv.right,
                                    c=float(g.l.median()),
                                    sd=float(g.r.std()) / np.sqrt(2),
                                    n=len(g)))

    slow = d.rolling(SLOW_WINDOW, min_periods=SLOW_WINDOW // 2,
                     center=True).mean() - mu
    fast = d - mu - slow
    day = x.index.floor("D")
    xd, yd = x.groupby(day).mean(), y.groupby(day).mean()
    nd = ok.groupby(day).sum()
    dd = (xd - yd)[nd >= 18]
    lvd = ((xd + yd) / 2)[nd >= 18]
    sd1 = float(resid.std()) / np.sqrt(2)
    sd24 = float((dd - dd.mean()).std()) / np.sqrt(2)
    out["summary"] = dict(
        n_hours=int(ok.sum()), level=float(lvl.mean()),
        bias=mu, abs_bias=abs(mu),
        sd_total=float(resid.std()), sd_slow=float(slow.std()),
        sd_fast=float(fast.std()),
        sd_ref_1h=sd1, sd_ref_24h=sd24,
        var_ref_1h=float(lvl.var()), var_ref_24h=float(lvd.var()),
        r2max_1h=1 - sd1 ** 2 / float(lvl.var()),
        r2max_24h=1 - sd24 ** 2 / float(lvd.var()))
    return out


def fit_model(b: pd.DataFrame) -> tuple[float, float]:
    """sigma(c) = sqrt(a^2 + b^2 c^2), residuals on a log scale.

    On a log scale each bin weighs by its relative error, so the
    high-concentration bins, with few hours and large sigma, do not dominate
    the fit. Weighted by the square root of the number of pairs in the bin.
    """
    from scipy.optimize import least_squares
    c = b.c.to_numpy()
    s = b.sd.to_numpy()
    w = np.sqrt(b.n_pairs.to_numpy())

    def res(p):
        return w * (0.5 * np.log(p[0] ** 2 + (p[1] * c) ** 2 + 1e-12)
                    - np.log(s))
    p0 = [max(s.min(), 1e-3), max(np.median(s / c), 1e-4)]
    r = least_squares(res, p0, bounds=([0, 0], [np.inf, np.inf]))
    return float(abs(r.x[0])), float(abs(r.x[1]))


def family_curve(bn: pd.DataFrame) -> pd.DataFrame:
    """Median per bin over pairs; only bins with 3 or more pairs."""
    c = bn.groupby(["lo", "hi"]).agg(c=("c", "median"), sd=("sd", "median"),
                                     n_pairs=("sd", "size"),
                                     n_sites=("site", "nunique")).reset_index()
    return c[c.n_pairs >= 3]


def boot(df: pd.DataFrame, fn, rng) -> tuple[float, float, float]:
    """Estimate and 2.5-97.5 percentiles, resampling sites."""
    sites = df.site.unique()
    groups = {s: df[df.site == s] for s in sites}
    est = fn(df)
    vals = [fn(pd.concat([groups[s] for s in rng.choice(sites, len(sites))]))
            for _ in range(B)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(description="Error structure by family.")
    p.add_argument("--agreement", type=Path,
                   default=root / "Results_Collocated"
                   / "agreement_precision.parquet")
    p.add_argument("--resolution", type=Path,
                   default=root / "Results_Collocated"
                   / "resolution_limit.parquet")
    p.add_argument("--hourly-dir", type=Path,
                   default=root / "Data_Collocated" / "hourly_pocs")
    p.add_argument("--out-dir", type=Path,
                   default=root / "Results_Collocated")
    p.add_argument("--seed", type=int, default=20261111)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    A = pd.read_parquet(a.agreement)
    A = A[A.family.notna()]
    allan_rows, bin_rows, sum_rows = [], [], []
    for (var, year), d in A.groupby(["variable", "year"]):
        f = a.hourly_dir / f"hourly_pocs_{var}_{year}.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        for r in d.itertuples():
            ca, cb = f"{r.site}#{r.poc_a}", f"{r.site}#{r.poc_b}"
            if ca not in H.columns or cb not in H.columns:
                continue
            key = dict(variable=var, year=year, site=r.site,
                       poc_a=r.poc_a, poc_b=r.poc_b, family=r.family)
            o = pair_analysis(H[ca].astype(float), H[cb].astype(float), var)
            allan_rows += [{**key, **x} for x in o["allan"]]
            bin_rows += [{**key, **x} for x in o["bins"]]
            sum_rows.append({**key, **o["summary"]})
        print(f"  {var} {year}: {len(d)} pairs")

    AL = pd.DataFrame(allan_rows)
    BN = pd.DataFrame(bin_rows)
    SM = pd.DataFrame(sum_rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    AL.to_parquet(a.out_dir / "error_allan.parquet", index=False)
    BN.to_parquet(a.out_dir / "error_bins.parquet", index=False)

    # --- error model by family, with site bootstrap -------------------------
    fits = []
    for (var, fam), g in BN.groupby(["variable", "family"]):
        if g.site.nunique() < MIN_SITES_FAMILY:
            continue
        a0, b0 = fit_model(family_curve(g))
        sites = g.site.unique()
        grp = {s: g[g.site == s] for s in sites}
        bs = []
        for _ in range(B):
            cc = family_curve(pd.concat([grp[s] for s in
                                         rng.choice(sites, len(sites))]))
            if len(cc) >= 3:
                bs.append(fit_model(cc))
        bs = np.array(bs)
        fits.append(dict(variable=var, family=fam, sites=len(sites),
                         a=a0, a_lo=np.percentile(bs[:, 0], 2.5),
                         a_hi=np.percentile(bs[:, 0], 97.5),
                         b=b0, b_lo=np.percentile(bs[:, 1], 2.5),
                         b_hi=np.percentile(bs[:, 1], 97.5)))
    FT = pd.DataFrame(fits)
    FT.to_parquet(a.out_dir / "error_model_fit.parquet", index=False)

    # --- ozone: Sheppard correction -----------------------------------------
    R = pd.read_parquet(a.resolution)
    O = R[R.variable == "o3"].copy()
    O["inst_sd_ppb"] = (np.sqrt(np.clip(O.sd_steps ** 2 - 1 / 6, 0, None))
                        * O.step * 1000 / np.sqrt(2))
    O = O.drop_duplicates(["site", "year"])
    SM = SM.merge(O[["variable", "site", "year", "sd_steps", "inst_sd_ppb"]],
                  on=["variable", "site", "year"], how="left")
    SM.to_parquet(a.out_dir / "error_summary.parquet", index=False)

    print("\n=== Allan deviation per instrument (median over pairs) ===")
    T = AL.groupby(["variable", "family", "tau"]).adev.median().unstack()
    print(T.round(3).to_string())

    print("\n=== Error model sigma^2(c) = a^2 + (b c)^2, 95% CI over sites ===")
    print(FT.round(3).to_string(index=False))

    print("\n=== Decomposition and sensor bound, median [95% CI over sites] ===")
    rows = []
    for (var, fam), g in SM.groupby(["variable", "family"]):
        if g.site.nunique() < MIN_SITES_FAMILY:
            continue
        rec = dict(variable=var, family=fam, pairs=len(g),
                   sites=g.site.nunique(), level=g.level.median())
        for col in ["abs_bias", "sd_total", "sd_slow", "sd_fast", "sd_ref_1h",
                    "sd_ref_24h", "r2max_1h", "r2max_24h", "inst_sd_ppb",
                    "sd_steps"]:
            if g[col].notna().sum() == 0:
                continue
            e, lo, hi = boot(g, lambda z, c=col: np.nanmedian(z[c]), rng)
            rec.update({col: e, f"{col}_lo": lo, f"{col}_hi": hi})
        for tau in (1, 24, 168):
            t = AL[(AL.variable == var) & (AL.family == fam) & (AL.tau == tau)]
            e, lo, hi = boot(t, lambda z: np.nanmedian(z.adev), rng)
            rec.update({f"adev_{tau}": e, f"adev_{tau}_lo": lo,
                        f"adev_{tau}_hi": hi})
        rows.append(rec)
    FS = pd.DataFrame(rows)
    FS.to_parquet(a.out_dir / "error_family.parquet", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(FS.set_index(["variable", "family"]).round(3).T.to_string())

    print("\nOzone: a difference due to rounding alone would give "
          f"{1/np.sqrt(6):.3f} increments; "
          f"{(O.sd_steps > 1/np.sqrt(6)).mean():.0%} of pairs exceed it.")
    print(f"\n-> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
