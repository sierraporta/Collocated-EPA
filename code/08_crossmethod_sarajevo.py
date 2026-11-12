#!/usr/bin/env python3
"""
08_crossmethod_sarajevo.py — Cross-method check in the Discussion
=========================================================================

Two numbers in the Discussion compare an optical monitor (T640) with a beta
attenuation monitor (BAM), i.e. instruments of *different* designation:

1. The network figure: median Appendix A coefficient of variation (Eq. 2 of the
   manuscript) of PM2.5 pairs in which one monitor is a T640 and the other a
   BAM, hourly and daily, from the comparability arm of 03_agreement.py.
   It is reported for pairs in which both monitors are SLAMS (the rule used
   for every other estimate) and, for reference, for all monitor types.

2. The Sarajevo figure: the same estimator applied to the daily data released
   with Hagler et al. (2022), doi:10.23719/1522400 (file Fig4-daily.csv,
   archived in Data_Collocated/hagler_sarajevo_daily.csv), using the 2 ug/m3
   floor of Appendix A section 4(c).

Inputs : Results_Collocated/agreement_comparability[_alltypes].parquet,
         Results_Collocated/concurrency.parquet,
         Data_Collocated/method_designations.csv,
         Data_Collocated/hagler_sarajevo_daily.csv
Output : Results_Collocated/crossmethod.csv (one row per subset)

Usage
-----
  python3 08_crossmethod_sarajevo.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["variable", "year", "site", "poc_a", "poc_b"]
PAIR = {"Beta attenuation", "T640 optical"}


def load_agreement(here: Path):
    """Import 03_agreement.py as a module (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("agreement",
                                                  here / "03_agreement.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def network(res_dir: Path, designations: Path, ag, suffix: str) -> dict | None:
    f = res_dir / f"agreement_comparability{suffix}.parquet"
    if not f.exists():
        print(f"  missing {f.name}; run 03_agreement.py --arm comparability"
              + (" --all-types" if suffix else ""))
        return None
    C = pd.read_parquet(f)
    Q = pd.read_parquet(res_dir / "concurrency.parquet")[
        KEY + ["method_a", "method_b"]]
    C = C.merge(Q, on=KEY, how="left")
    fam = ag.family_map(designations)
    fa, fb = C.method_a.map(fam), C.method_b.map(fam)
    same = np.array([{x, y} == PAIR for x, y in zip(fa, fb)], dtype=bool)
    sel = (C.variable == "pm25").to_numpy() & same
    P = C[sel]
    return dict(subset="network T640-BAM, " + ("all monitor types" if suffix
                                               else "both SLAMS"),
                pair_years=len(P), sites=P.site.nunique(),
                cv_hourly=P.cv_hourly.median(), cv_daily=P.cv_daily.median(),
                level=P.mean_level.median())


def sarajevo(path: Path, ag) -> dict:
    D = pd.read_csv(path).dropna(subset=["T640", "BAM"])
    x, y = D.T640.to_numpy(float), D.BAM.to_numpy(float)
    floor = ag.FLOOR["pm25"]
    m = (x >= floor) & (y >= floor)
    cv, k = ag.cv_upper(ag.rpd(x[m], y[m]))
    return dict(subset="Sarajevo T640-BAM (Hagler et al. 2022), daily",
                pair_years=np.nan, sites=1, cv_hourly=np.nan, cv_daily=cv,
                level=float(np.median((x[m] + y[m]) / 2)), n_days=k)


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--res-dir", type=Path, default=root / "Results_Collocated")
    p.add_argument("--designations", type=Path,
                   default=root / "Data_Collocated" / "method_designations.csv")
    p.add_argument("--sarajevo", type=Path,
                   default=root / "Data_Collocated" / "hagler_sarajevo_daily.csv")
    a = p.parse_args(argv)

    ag = load_agreement(here)
    rows = [r for r in (network(a.res_dir, a.designations, ag, ""),
                        network(a.res_dir, a.designations, ag, "_alltypes"))
            if r is not None]
    rows.append(sarajevo(a.sarajevo, ag))
    T = pd.DataFrame(rows)
    T.to_csv(a.res_dir / "crossmethod.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 140):
        print(T.round(2).to_string(index=False))
    print(f"\n-> {a.res_dir / 'crossmethod.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
