#!/usr/bin/env python3
"""
01_build_pocs.py — Hourly series for every monitor of every pair
================================================================

Analyses that treat the site as the unit keep **one POC per site** (the NAAQS
primary monitor, or the POC with most observations), because averaging two
instruments creates a discontinuity wherever one replaces the other. Here the
unit is the **pair**, so every parameter occurrence code (POC) at a site with
more than one monitor is kept as its own column, keyed `site#POC`.

The streaming download, the retry logic for the EPA server and the unit
conversion are imported from `aqs_io.py`. Only sites that have a replicate
(from `pairs_inventory.parquet`) are ingested, and no daily aggregates are
computed.

The **method code of each column** is also stored. The hourly files carry it on
every row, which makes it possible to detect a change of instrument within a
year, the one case in which a column cannot be treated as a single instrument.

The national hourly files are several hundred MB each and the full run takes
hours; it is only needed to extend the study period.

Usage
-----
  cd code
  python3 01_build_pocs.py --years 2023 2024 2025 2026
  python3 01_build_pocs.py --reindex      # rebuild the column index, no network

  # offline validation against a local server:
  python3 test_build_pocs.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_VARS = ["pm25", "o3", "no2", "pm10"]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def site_sets(pairs: Path) -> dict[str, set[str]]:
    """Sites with a replicate, **per pollutant**.

    Filtering by the union over pollutants would be wasteful: PM2.5 has about
    400 sites with replicates and ozone about 40, so ingesting ozone at every
    site of the union would store mostly empty columns.
    """
    P = pd.read_parquet(pairs)
    return {v: set(d.site.unique()) for v, d in P.groupby("variable")}


def ingest_pocs(aq, var: str, year: int, sites: set[str], a) -> dict | None:
    """Hourly matrix with one column per site and POC."""
    file_code, pcode, _ = aq.VARS[var]
    url = f"{a.base_url}/hourly_{file_code}_{year}.zip"
    hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    origin = pd.Timestamp(f"{year}-01-01")
    nh = len(hours)
    ONE_H = pd.Timedelta(hours=1)

    series: dict[str, np.ndarray] = {}
    methods: dict[str, Counter] = {}
    offsets: dict[str, Counter] = {}
    n_rows = n_kept = n_chunks = 0

    t0 = time.time()
    aq.log(f"{var} {year}: reading {url}")
    aq.log("    (the national file is several hundred MB; progress is "
           "reported every five chunks)")
    fh = aq.open_remote_csv(url, a.timeout, a.retries, a.pause)
    with fh:
        reader = pd.read_csv(fh, usecols=aq.USECOLS, dtype=aq.DTYPES,
                             chunksize=a.chunk_rows, low_memory=False)
        for ch in reader:
            n_rows += len(ch)
            ch = ch[ch["Parameter Code"] == pcode]
            if ch.empty:
                continue
            ch = ch[~ch["State Code"].str.zfill(2).isin(aq.NON_CONUS)]
            ch = ch[ch["Sample Measurement"].notna()]
            if a.drop_qualified:
                ch = ch[ch["Qualifier"].isna()
                        | (ch["Qualifier"].str.strip() == "")]
            if ch.empty:
                continue

            sid = (ch["State Code"].str.zfill(2) + "_"
                   + ch["County Code"].str.zfill(3) + "_"
                   + ch["Site Num"].str.zfill(4))
            ch = ch.assign(site=sid.to_numpy())
            # Only sites with a replicate. The national file is downloaded in
            # full anyway, but keeping a quarter of the sites cuts memory and
            # disk by the same factor.
            ch = ch[ch["site"].isin(sites)]
            if ch.empty:
                continue

            tg = pd.to_datetime(ch["Date GMT"] + " " + ch["Time GMT"],
                                format="%Y-%m-%d %H:%M", errors="coerce")
            tlf = pd.to_datetime(ch["Date Local"] + " " + ch["Time Local"],
                                 format="%Y-%m-%d %H:%M", errors="coerce")
            ok = (tg.notna() & tlf.notna()).to_numpy()
            if not ok.any():
                continue
            ch, tg, tlf = ch[ok], tg[ok], tlf[ok]

            hi = ((tg - origin) // ONE_H).to_numpy()
            inside = (hi >= 0) & (hi < nh)
            if not inside.any():
                continue
            ch, tg, tlf, hi = ch[inside], tg[inside], tlf[inside], hi[inside]

            vals = aq.convert(ch["Sample Measurement"].to_numpy(),
                              ch["Units of Measure"], var)
            off = ((tlf - tg) // ONE_H).to_numpy()
            mcs = ch["Method Code"].fillna("").astype(str).str.strip() \
                    .str.zfill(3).to_numpy()
            keys = (ch["site"].astype(str) + "#"
                    + ch["POC"].astype(str)).to_numpy()

            # Comparing `keys == k` once per column would scan the whole chunk
            # as many times as there are columns. Sorting once and cutting at
            # the boundaries of each key scans it a single time.
            order = np.argsort(keys, kind="stable")
            ks = keys[order]
            cuts = np.flatnonzero(
                np.concatenate(([True], ks[1:] != ks[:-1], [True])))
            for i in range(len(cuts) - 1):
                lo, hi_ = cuts[i], cuts[i + 1]
                k = ks[lo]
                idx = order[lo:hi_]
                if k not in series:
                    series[k] = np.full(nh, np.nan, dtype="float32")
                    methods[k] = Counter()
                    offsets[k] = Counter()
                series[k][hi[idx]] = vals[idx]
                u, c = np.unique(mcs[idx], return_counts=True)
                methods[k].update(dict(zip(u.tolist(), c.tolist())))
                u, c = np.unique(off[idx], return_counts=True)
                offsets[k].update(dict(zip(u.tolist(), c.tolist())))
                n_kept += len(idx)
            n_chunks += 1
            if n_chunks % 5 == 0:
                aq.log(f"    {var} {year}: {n_rows:,} rows read, "
                       f"{len(series)} columns, {time.time() - t0:.0f} s")

    if not series:
        aq.log(f"  {var} {year}: no usable rows")
        return None

    cols = sorted(series)
    H = pd.DataFrame({c: series[c] for c in cols}, index=hours)
    H.index.name = "time_utc"

    idx = []
    for c in cols:
        site, poc = c.rsplit("#", 1)
        mc = methods[c].most_common()
        mc = [(k, v) for k, v in mc if k not in ("", "000")]
        off = offsets[c].most_common(1)[0][0] if offsets[c] else np.nan
        idx.append(dict(
            variable=var, year=year, column=c, site=site, poc=int(poc),
            n_hours=int(np.isfinite(series[c]).sum()),
            method_code=mc[0][0] if mc else None,
            method_n_hours=mc[0][1] if mc else 0,
            # A change of instrument within the year breaks the assumption that
            # the column is a single instrument. It is flagged here, not dropped.
            method_changed=len(mc) > 1,
            method_all=";".join(f"{k}:{v}" for k, v in mc) if mc else None,
            gmt_offset=int(off) if np.isfinite(off) else None))
    I = pd.DataFrame(idx)

    (a.out_dir / "hourly_pocs").mkdir(parents=True, exist_ok=True)
    H.to_parquet(a.out_dir / "hourly_pocs"
                 / f"hourly_pocs_{var}_{year}.parquet")
    el = time.time() - t0
    aq.log(f"  {var} {year}: {len(cols)} site-POC columns at "
           f"{I.site.nunique()} sites, {n_kept} usable hours of {n_rows} "
           f"rows, {el:.0f} s")
    return {"index": I, "n_rows": n_rows, "n_kept": n_kept,
            "seconds": round(el, 1)}


def merge_index(out: Path, I: pd.DataFrame) -> int:
    """Merge the index of one pollutant-year into the cumulative file.

    Called **after each pollutant-year** rather than at the end, so that an
    interrupted run never leaves an hourly parquet on disk without its index
    rows (the next run would skip that combination and never write them).
    """
    if out.exists():
        old = pd.read_parquet(out)
        I = pd.concat([old, I], ignore_index=True).drop_duplicates(
            ["variable", "year", "column"], keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    I.to_parquet(out, index=False)
    return len(I)


def reindex(a, aq) -> int:
    """Rebuild poc_columns.parquet from the hourly parquet files on disk.

    Sites, POCs and observed hours come from the hourly files themselves. The
    two fields that were computed from the raw CSV are taken from other
    sources, and the rows are marked `source="reindex"`:

      method_code  from `poc_methods.csv` (AQS daily summaries), by pollutant,
                   site, POC and year.
      gmt_offset   from `stations_hourly.parquet`, since the offset is a
                   property of the site, not of the instrument.
    """
    K = pd.read_csv(a.methods, dtype={"site": str, "method_code": str})
    K["method_code"] = K.method_code.str.zfill(3)
    nmeth = K.groupby(["variable", "site", "poc", "year"]).method_code.nunique()
    K = K.sort_values("n_days", ascending=False).drop_duplicates(
        ["variable", "site", "poc", "year"])
    mcode = K.set_index(["variable", "site", "poc", "year"]).method_code
    St = pd.read_parquet(a.stations_hourly)
    off = St["gmt_offset"] if "gmt_offset" in St.columns else None

    rows = []
    files = sorted((a.out_dir / "hourly_pocs").glob("hourly_pocs_*.parquet"))
    for f in files:
        stem = f.stem.replace("hourly_pocs_", "")
        var, year = stem.rsplit("_", 1)
        year = int(year)
        H = pd.read_parquet(f)
        n = H.notna().sum()
        for c in H.columns:
            site, poc = c.rsplit("#", 1)
            poc = int(poc)
            k = (var, site, poc, year)
            rows.append(dict(
                variable=var, year=year, column=c, site=site, poc=poc,
                n_hours=int(n[c]),
                method_code=mcode.get(k),
                method_n_hours=None,
                method_changed=bool(nmeth.get(k, 1) > 1),
                method_all=None,
                gmt_offset=(int(off[site]) if off is not None
                            and site in off.index
                            and pd.notna(off[site]) else None),
                source="reindex"))
        aq.log(f"  {var} {year}: {H.shape[1]} columns")
    I = pd.DataFrame(rows)
    out = a.out_dir / "poc_columns.parquet"
    if out.exists():
        old = pd.read_parquet(out)
        # Rows from the original ingestion take precedence over rebuilt ones.
        if "source" not in old.columns:
            old["source"] = "ingest"
        I = pd.concat([I, old], ignore_index=True).drop_duplicates(
            ["variable", "year", "column"], keep="last")
    I.to_parquet(out, index=False)
    aq.log(f"\n{len(I)} columns indexed -> {out}")
    aq.log(f"  from ingestion: {int((I.source == 'ingest').sum())}, "
           f"rebuilt: {int((I.source == 'reindex').sum())}")
    aq.log(f"  without method code: {int(I.method_code.isna().sum())}")
    return 0


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(
        description="Hourly ingestion keeping every POC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--aqs-io", type=Path, default=here / "aqs_io.py",
                   help="module providing the streaming reader and conversion")
    p.add_argument("--pairs", type=Path,
                   default=root / "Results_Collocated"
                   / "pairs_inventory.parquet")
    p.add_argument("--out-dir", type=Path, default=root / "Data_Collocated")
    p.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    p.add_argument("--years", type=int, nargs="+",
                   default=[2023, 2024, 2025, 2026])
    p.add_argument("--base-url", default=None,
                   help="defaults to aqs_io.BASE")
    p.add_argument("--chunk-rows", type=int, default=500_000)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--pause", type=float, default=5.0)
    p.add_argument("--drop-qualified", action="store_true",
                   help="drop hours that carry a qualifier code")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reindex", action="store_true",
                   help="rebuild poc_columns.parquet from the parquet files "
                        "on disk, without network")
    p.add_argument("--methods", type=Path,
                   default=root / "Data_Collocated" / "poc_methods.csv",
                   help="--reindex only: method code per POC and year")
    p.add_argument("--stations-hourly", type=Path,
                   default=root / "Data_Collocated" / "inputs"
                   / "stations_hourly.parquet",
                   help="--reindex only: GMT offset per site")
    a = p.parse_args(argv)

    aq = load(a.aqs_io, "aqs_io")
    if a.reindex:
        return reindex(a, aq)
    if a.base_url is None:
        a.base_url = aq.BASE
    by_var = site_sets(a.pairs)
    aq.log("sites with a replicate, by pollutant: "
           + ", ".join(f"{v} {len(by_var.get(v, ()))}" for v in a.vars))

    index, manifest = [], []
    for var in a.vars:
        if var not in aq.VARS:
            aq.log(f"  unknown variable: {var}")
            continue
        sites = by_var.get(var, set())
        if not sites:
            aq.log(f"{var}: no site with a replicate, skipped")
            continue
        for year in a.years:
            f = a.out_dir / "hourly_pocs" / f"hourly_pocs_{var}_{year}.parquet"
            if f.exists() and not a.overwrite:
                aq.log(f"{var} {year}: already exists, skipped")
                continue
            try:
                r = ingest_pocs(aq, var, year, sites, a)
            except Exception as exc:                      # noqa: BLE001
                aq.log(f"  {var} {year}: FAILED {type(exc).__name__}: {exc}")
                manifest.append(dict(variable=var, year=year, ok=False,
                                     error=f"{type(exc).__name__}: {exc}"))
                continue
            if r is None:
                manifest.append(dict(variable=var, year=year, ok=False,
                                     error="no rows"))
                continue
            I = r.pop("index")
            I["source"] = "ingest"
            n = merge_index(a.out_dir / "poc_columns.parquet", I)
            aq.log(f"    index: {len(I)} columns added, {n} in total")
            index.append(I)
            manifest.append(dict(variable=var, year=year, ok=True, **r))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    if index:
        I = pd.concat(index, ignore_index=True)
        aq.log(f"\nthis run: {len(I)} columns in "
               f"{len(index)} pollutant-years")
        aq.log(f"  with a change of method within the year: "
               f"{int(I.method_changed.sum())}")
    # The manifest is merged rather than overwritten, so that a run that skips
    # everything does not erase the record of the run that did the work.
    mf = a.out_dir / "build_pocs_manifest.json"
    if manifest or not mf.exists():
        prev = []
        if mf.exists():
            try:
                prev = json.load(open(mf))
            except (json.JSONDecodeError, OSError):
                prev = []
        seen = {(m.get("variable"), m.get("year")) for m in manifest}
        merged = manifest + [m for m in prev
                             if (m.get("variable"), m.get("year")) not in seen]
        with open(mf, "w") as fh:
            json.dump(merged, fh, indent=2)
    bad = [m for m in manifest if not m.get("ok")]
    if bad:
        aq.log(f"\n{len(bad)} combinations failed: "
               + ", ".join(f"{m['variable']} {m['year']}" for m in bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
