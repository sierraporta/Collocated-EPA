"""End-to-end test of 01_build_pocs.py against a local HTTP server.

The script is validated without network access by writing synthetic files in
the exact AQS hourly format and serving them over local HTTP.

The checks cover what can actually break in this script: the series are
**not** collapsed to one POC per site, each column carries its own method code,
a change of instrument within the year is flagged, sites without a replicate
are dropped, the index stays in UTC with the local offset recovered, and units
are converted only where they should be.
"""
import functools
import http.server
import importlib.util
import socketserver
import sys
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent
SRV = S / "_test_pocs" / "server"
OUT = S / "_test_pocs" / "out"
YEAR = 2024
NH = 96

COLS = ["State Code", "County Code", "Site Num", "Parameter Code", "POC",
        "Latitude", "Longitude", "Datum", "Parameter Name", "Date Local",
        "Time Local", "Date GMT", "Time GMT", "Sample Measurement",
        "Units of Measure", "MDL", "Uncertainty", "Qualifier", "Method Type",
        "Method Code", "Method Name", "State Name", "County Name",
        "Date of Last Change"]

FAILS = []
N_CHECKS = 0


def check(cond, msg):
    global N_CHECKS
    N_CHECKS += 1
    print(("  ok     " if cond else "  FAILED ") + msg)
    if not cond:
        FAILS.append(msg)


def rows(site, pcode, pname, poc, units, values, hours, offset,
         method="100", state="01", county="073", lat=33.55, lon=-86.81):
    out = []
    for t, v in zip(hours, values):
        loc = t + pd.Timedelta(hours=offset)
        mc = method(t) if callable(method) else method
        out.append([state, county, site, pcode, poc, lat, lon, "WGS84", pname,
                    loc.strftime("%Y-%m-%d"), loc.strftime("%H:%M"),
                    t.strftime("%Y-%m-%d"), t.strftime("%H:%M"),
                    "" if v is None else f"{v:.3f}", units, 0.4, "", "", "FEM",
                    mc, "INSTRUMENTAL", "Alabama", "Jefferson", "2026-06-15"])
    return out


def build_fixtures():
    SRV.mkdir(parents=True, exist_ok=True)
    hours = pd.date_range(f"{YEAR}-01-01", periods=NH, freq="h")
    rng = np.random.default_rng(0)

    a = rng.normal(10, 2, NH)
    b = a + rng.normal(0, 0.5, NH)          # correlated replicate
    r = []
    # site 0023: two POCs, same method -> precision pair
    r += rows("0023", 88101, "PM2.5", 1, "Micrograms/cubic meter (LC)",
              a, hours, -6, method="170")
    r += rows("0023", 88101, "PM2.5", 2, "Micrograms/cubic meter (LC)",
              b, hours, -6, method="170")
    # site 1005: two POCs, different methods -> comparability pair,
    # and POC 2 changes instrument during the year
    r += rows("1005", 88101, "PM2.5", 1, "Micrograms/cubic meter (LC)",
              rng.normal(8, 2, NH), hours, -6, method="145",
              county="089", lat=34.8, lon=-87.0)
    r += rows("1005", 88101, "PM2.5", 2, "Micrograms/cubic meter (LC)",
              rng.normal(9, 2, NH), hours, -6, county="089", lat=34.8,
              lon=-87.0, method=lambda t: "170" if t.day < 3 else "238")
    # site 2001: a single POC, NOT in the pair list -> dropped
    r += rows("2001", 88101, "PM2.5", 1, "Micrograms/cubic meter (LC)",
              rng.normal(7, 2, NH), hours, -6, county="097", lat=30.7,
              lon=-88.1)
    write_zip("hourly_88101_2024.zip", r)

    # O3 in ppm: gases are kept in the reported unit; only temperature and
    # wind are converted.
    r = rows("0023", 44201, "Ozone", 1, "Parts per million",
             np.full(NH, 0.045), hours, -6, method="047")
    r += rows("0023", 44201, "Ozone", 2, "Parts per million",
              np.full(NH, 0.047), hours, -6, method="047")
    write_zip("hourly_44201_2024.zip", r)

    # TEMP in Fahrenheit: converted, and the conversion is checked
    r = rows("0023", 62101, "Outdoor Temperature", 1, "Degrees Fahrenheit",
             np.full(NH, 68.0), hours, -6, method="040")
    r += rows("0023", 62101, "Outdoor Temperature", 2, "Degrees Fahrenheit",
              np.full(NH, 68.0), hours, -6, method="040")
    write_zip("hourly_TEMP_2024.zip", r)
    return hours, a, b


def write_zip(name, data):
    df = pd.DataFrame(data, columns=COLS)
    with zipfile.ZipFile(SRV / name, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name.replace(".zip", ".csv"), df.to_csv(index=False))


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(SRV))
    class Q(socketserver.TCPServer):
        allow_reuse_address = True
        def handle_error(self, *_): pass
    srv = Q(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> int:
    hours, a, b = build_fixtures()
    OUT.mkdir(parents=True, exist_ok=True)
    # synthetic pair table: only the two sites with a replicate
    pairs = OUT / "pairs.parquet"
    pd.DataFrame({"site": ["01_073_0023", "01_089_1005",
                           "01_073_0023", "01_073_0023"],
                  "variable": ["pm25", "pm25", "o3", "temp"],
                  "year": [YEAR] * 4}).to_parquet(pairs, index=False)

    srv, base = serve()
    spec = importlib.util.spec_from_file_location("bp", S / "01_build_pocs.py")
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)
    rc = bp.main(["--aqs-io", str(S / "aqs_io.py"),
                  "--pairs", str(pairs), "--out-dir", str(OUT),
                  "--vars", "pm25", "o3", "temp",
                  "--years", str(YEAR),
                  "--base-url", base, "--overwrite"])
    srv.shutdown()

    print("\n--- checks ---")
    check(rc == 0, "ingestion finishes without failures")
    H = pd.read_parquet(OUT / "hourly_pocs" / f"hourly_pocs_pm25_{YEAR}.parquet")
    I = pd.read_parquet(OUT / "poc_columns.parquet")

    check(list(H.columns) == ["01_073_0023#1", "01_073_0023#2",
                              "01_089_1005#1", "01_089_1005#2"],
          f"one column per site and POC, not collapsed: {list(H.columns)}")
    check("01_097_2001#1" not in H.columns,
          "the site without a replicate is dropped")
    check(H.index.name == "time_utc" and len(H) == 8784,
          "complete hourly index for the year, in UTC")
    check(np.allclose(H["01_073_0023#1"].to_numpy()[:NH], a.astype("float32"),
                      atol=1e-3),
          "POC 1 values arrive intact")
    check(np.allclose(H["01_073_0023#2"].to_numpy()[:NH], b.astype("float32"),
                      atol=1e-3),
          "POC 2 values arrive intact and differ from POC 1")

    P = I[I.variable == "pm25"].set_index("column")
    check(P.loc["01_073_0023#1", "method_code"] == "170"
          and P.loc["01_073_0023#2", "method_code"] == "170",
          "the precision pair keeps the same method code")
    check(P.loc["01_089_1005#1", "method_code"] == "145",
          "the comparability pair keeps different methods")
    check(bool(P.loc["01_089_1005#2", "method_changed"]),
          "a change of instrument during the year is flagged")
    check(not bool(P.loc["01_073_0023#1", "method_changed"]),
          "a stable column is not flagged by mistake")
    check(P.loc["01_073_0023#1", "gmt_offset"] == -6,
          "the local time offset is recovered")
    check(int(P.loc["01_073_0023#1", "n_hours"]) == NH,
          "the count of observed hours is correct")

    O = pd.read_parquet(OUT / "hourly_pocs" / f"hourly_pocs_o3_{YEAR}.parquet")
    check(abs(float(O["01_073_0023#1"].iloc[0]) - 0.045) < 1e-4,
          "gases keep the reported unit (ozone in ppm)")
    T = pd.read_parquet(OUT / "hourly_pocs" / f"hourly_pocs_temp_{YEAR}.parquet")
    check(abs(float(T["01_073_0023#1"].iloc[0]) - 20.0) < 0.05,
          "temperature is converted from Fahrenheit to Celsius (68 F -> 20 C)")

    print(f"\n{'ALL PASSED' if not FAILS else 'SOME FAILED'}: "
          f"{len(FAILS)} of {N_CHECKS} checks fail")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
