#!/usr/bin/env python3
"""
00_pairs_inventory.py — Which monitor pairs exist, and what each can measure
============================================================================

At some sites the EPA network measures the same air twice. Before using these
duplicates one has to establish **why** each pair exists, because that decides
what it can measure.

40 CFR 58 Appendix A requires 15% of primary PM2.5 monitors of each method
designation to be collocated, but those quality-control samplers are filter
based and run one day in twelve: they produce no hourly series. The hourly
duplicates in the archive exist for other reasons (SLAMS sites operating two
continuous instruments, special purpose monitors, industrial monitors), so the
sample is not the mandated collocation programme and has to be classified, and
compared with the network, before use.

Classification cannot use the raw method code. Several codes share one
**equivalent method designation**: PM2.5 codes 236, 636 and 736 are all
EQPM-0516-236 (Teledyne T640 at 5.0 L/min) and differ only in data processing
(unaligned, Network Data Alignment enabled in firmware, and recalculated). The
same holds for 238/638/738 (T640X) and for the NO2 codes 074/574 and 099/599.
Grouping by designation gives three categories:

  identical code              **precision**: two units of the same instrument
                              with the same processing, sampling the same air.

  same designation,           **reprocessing**: the same T640/T640X record with
  different code              and without the Network Data Alignment.

  different designation       **comparability**: different instruments.

The method code is taken from `poc_methods.csv` (AQS daily summaries), which
gives it **per year**. The `Last Method Code` field of the monitor listing is
not used: it is the last method of the monitor's life and misassigns every year
before a change of instrument.

Concurrent hours, the actual sample size of each pair, are computed in step 2.

Usage
-----
  python3 00_pairs_inventory.py
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# Monitors with a comparable regulatory role. Industrial or special purpose
# monitors may be operated for other reasons and under other maintenance
# schedules, so mixed pairs are counted but not used in the network estimates.
REGULATORY = {"SLAMS"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def designation_map(path: Path) -> dict[str, str]:
    """Method code -> equivalent method designation.

    The table comes from the AQS method code list, which is refreshed nightly:
    https://aqs.epa.gov/aqsweb/documents/codetables/methods_criteria.html
    It is archived with the code because the classification depends on it.
    """
    D = pd.read_csv(path, dtype=str)
    D["method_code"] = D.method_code.str.strip().str.zfill(3)
    return dict(zip(D.method_code, D.designation))


def load_inputs(a) -> tuple:
    sel = pd.read_parquet(a.poc_selection)
    met = pd.read_csv(a.methods, dtype={"site": str, "method_code": str})
    mon = pd.read_csv(a.monitors, dtype=str)
    st = pd.read_parquet(a.stations)

    # A POC can change method within a year; the method covering most days is
    # kept and the change is flagged.
    #
    # The flag counts **distinct method codes**, not rows: a POC appears in
    # several rows of the daily summary when it reports more than one sample
    # duration (hourly and 24-hour block, for example), which is not a change
    # of instrument.
    met = met.sort_values("n_days", ascending=False)
    key = ["variable", "site", "poc", "year"]
    changed = met.groupby(key).method_code.nunique() > 1
    met1 = met.drop_duplicates(key).copy()
    met1["method_changed"] = met1.set_index(key).index.map(
        changed).fillna(False)

    mon["site"] = (mon["State Code"].str.zfill(2) + "_"
                   + mon["County Code"].str.zfill(3) + "_"
                   + mon["Site Number"].str.zfill(4))
    mon["variable"] = mon["Parameter Code"].map(a.param_map)
    mon["poc"] = pd.to_numeric(mon["POC"], errors="coerce")
    mon = mon.rename(columns={"Monitor Type": "monitor_type",
                              "Networks": "networks",
                              "Measurement Scale": "scale"})
    keep = ["site", "variable", "poc", "monitor_type", "networks", "PQAO",
            "scale"]
    return sel, met1, mon[keep].drop_duplicates(["site", "variable", "poc"]), st


def build_pairs(sel, met, mon, min_hours: int, des: dict) -> tuple:
    """All POC pairs within each site, pollutant and year."""
    s = sel[sel.n_hours >= min_hours].copy()
    n_poc = s.groupby(["site", "variable", "year"]).poc.nunique()
    s = s.merge(n_poc[n_poc > 1].rename("n_poc"),
                on=["site", "variable", "year"])
    s = s.merge(met[["variable", "site", "poc", "year", "method_code",
                     "method_name", "method_changed"]],
                on=["variable", "site", "poc", "year"], how="left")
    s = s.merge(mon, on=["site", "variable", "poc"], how="left")

    rows = []
    for (site, var, year), d in s.groupby(["site", "variable", "year"]):
        for a, b in itertools.combinations(
                d.sort_values("poc").to_dict("records"), 2):
            mc_a, mc_b = a["method_code"], b["method_code"]
            known = pd.notna(mc_a) and pd.notna(mc_b)
            da, db = des.get(mc_a), des.get(mc_b)
            if not known or da is None or db is None:
                arm = None
            elif mc_a == mc_b:
                arm = "precision"
            elif da == db:
                arm = "reprocessing"
            else:
                arm = "comparability"
            rows.append(dict(
                site=site, variable=var, year=year,
                poc_a=a["poc"], poc_b=b["poc"],
                method_a=mc_a, method_b=mc_b,
                designation_a=da, designation_b=db, arm=arm,
                method_known=known,
                same_method=bool(known and mc_a == mc_b),
                method_changed=bool(a["method_changed"] or b["method_changed"]),
                type_a=a["monitor_type"], type_b=b["monitor_type"],
                both_regulatory=(a["monitor_type"] in REGULATORY
                                 and b["monitor_type"] in REGULATORY),
                same_pqao=a["PQAO"] == b["PQAO"],
                h_a=a["n_hours"], h_b=b["n_hours"],
                h_min=min(a["n_hours"], b["n_hours"]),
                selected_a=a["selected"], selected_b=b["selected"]))
    return pd.DataFrame(rows), s


def representativeness(sel, pairs, st, min_hours: int) -> pd.DataFrame:
    """Sites with a replicate compared with the network that contains them.

    Duplicate monitors are not placed at random. If the replicate sites are
    systematically urban, or concentrated in one region, precision estimated
    there cannot be attributed to the network. The table gives the shares by
    land use, location setting and NOAA climate region.
    """
    out = []
    univ = sel[sel.n_hours >= min_hours]
    cols = ["Land Use", "Location Setting", "noaa_region"]
    for var in sorted(pairs.variable.unique()):
        all_sites = set(univ.loc[univ.variable == var, "site"])
        rep_sites = set(pairs.loc[pairs.variable == var, "site"])
        for col in cols:
            a = st.reindex(sorted(all_sites))[col].fillna("(missing)")
            r = st.reindex(sorted(rep_sites))[col].fillna("(missing)")
            for lev in sorted(set(a.unique()) | set(r.unique())):
                out.append(dict(variable=var, attribute=col, level=lev,
                                n_network=int((a == lev).sum()),
                                n_replicate=int((r == lev).sum()),
                                p_network=float((a == lev).mean()) if len(a) else np.nan,
                                p_replicate=float((r == lev).mean()) if len(r) else np.nan))
    R = pd.DataFrame(out)
    R["difference"] = R.p_replicate - R.p_network
    return R


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(
        description="Inventory of collocated monitor pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--poc-selection", type=Path,
                   default=root / "Data_Collocated" / "inputs"
                   / "poc_selection.parquet")
    p.add_argument("--methods", type=Path,
                   default=root / "Data_Collocated" / "poc_methods.csv")
    p.add_argument("--monitors", type=Path,
                   default=root / "Data_Collocated" / "inputs" / "monitors_subset.csv")
    p.add_argument("--designations", type=Path,
                   default=root / "Data_Collocated"
                   / "method_designations.csv",
                   help="method code -> AQS equivalent method designation")
    p.add_argument("--stations", type=Path,
                   default=root / "Data_Collocated" / "inputs" / "stations.parquet")
    p.add_argument("--out-dir", type=Path,
                   default=root / "Results_Collocated")
    p.add_argument("--min-hours", type=int, default=2000,
                   help="minimum observed hours of each POC of the pair")
    a = p.parse_args(argv)
    a.param_map = {"44201": "o3", "42401": "so2", "42101": "co",
                   "42602": "no2", "88101": "pm25", "88502": "pm25n",
                   "81102": "pm10", "86101": "pmc", "61103": "wspd",
                   "62101": "temp", "62201": "rh"}

    des = designation_map(a.designations)
    sel, met, mon, st = load_inputs(a)
    pairs, poc = build_pairs(sel, met, mon, a.min_hours, des)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(a.out_dir / "pairs_inventory.parquet", index=False)
    poc.to_parquet(a.out_dir / "pairs_pocs.parquet", index=False)

    log(f"pairs with both POCs above {a.min_hours} h: {len(pairs)}")
    log(f"  with a method code for both: {int(pairs.method_known.sum())}")
    log(f"  with a change of method within the year in either: "
        f"{int(pairs.method_changed.sum())}")
    log()

    log("=== monitor types of the pair ===")
    tt = pairs.apply(lambda r: " + ".join(sorted(
        [str(r.type_a), str(r.type_b)])), axis=1)
    log(tt.value_counts().head(10).to_string())
    log()

    arms = {
        "PRECISION     (identical code: two units of the same instrument)":
            (pairs.arm == "precision") & pairs.both_regulatory,
        "REPROCESSING  (same designation, different code)":
            (pairs.arm == "reprocessing") & pairs.both_regulatory,
        "COMPARABILITY (different designation: different instruments)":
            (pairs.arm == "comparability") & pairs.both_regulatory,
        "unclassified (designation missing for a code)":
            pairs.arm.isna() & pairs.both_regulatory,
        "non-regulatory (at least one monitor is not SLAMS)":
            ~pairs.both_regulatory,
    }
    for name, sel_mask in arms.items():
        S = pairs[sel_mask]
        log(f"=== {name}: {len(S)} pairs, {S.site.nunique()} sites ===")
        if len(S):
            T = S.groupby("variable").agg(
                pairs=("site", "size"), sites=("site", "nunique"),
                years=("year", "nunique"), h_min_median=("h_min", "median"))
            log(T.sort_values("pairs", ascending=False).to_string())
        log()

    log("=== operated by the same agency (PQAO), SLAMS pairs ===")
    log(pairs[pairs.both_regulatory].same_pqao.value_counts().to_string())
    log()

    R = representativeness(sel, pairs[pairs.both_regulatory & pairs.arm.notna()],
                           st, a.min_hours)
    R.to_parquet(a.out_dir / "representativeness.parquet", index=False)
    log("=== representativeness: largest deviations from the network ===")
    big = R[R.n_replicate >= 5].copy()
    big = big.assign(_abs=big.difference.abs()).sort_values(
        "_abs", ascending=False).head(14)
    log(big[["variable", "attribute", "level", "n_network", "n_replicate",
             "p_network", "p_replicate", "difference"]].round(3).to_string(
                 index=False))
    log()
    log(f"-> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
