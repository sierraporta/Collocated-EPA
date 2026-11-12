"""
aqs_io.py — Streaming reader for the EPA AQS hourly files
=========================================================

Library module used by 01_build_pocs.py. It reads the national hourly files
published at https://aqs.epa.gov/aqsweb/airdata/ **without writing them to
disk**: a single GET request is opened, the zip member is decompressed as the
bytes arrive, and the CSV text is handed to pandas in chunks.

    GET -> compressed bytes -> incremental zlib -> text -> pandas chunks

A zip member can be decompressed sequentially from its local header without
reading the central directory at the end of the file, so no range requests are
needed and any server that delivers the file in one piece will do.

Contents
--------
BASE, VARS, USECOLS, DTYPES, NON_CONUS   constants describing the AQS files
log()                                    timestamped console output
open_remote_csv()                        text stream of the CSV inside a remote zip
convert()                                unit conversion to canonical units
"""

from __future__ import annotations

import http.client
import io
import random
import socket
import time
import urllib.error
import urllib.request
import zlib

import numpy as np
import pandas as pd

BASE = "https://aqs.epa.gov/aqsweb/airdata"

# name -> (file stem, parameter code, canonical unit)
# The WIND and RH_DP files contain two parameters each and are split by
# Parameter Code.
VARS = {
    "o3":    ("44201", 44201, "Parts per million"),
    "so2":   ("42401", 42401, "Parts per billion"),
    "co":    ("42101", 42101, "Parts per million"),
    "no2":   ("42602", 42602, "Parts per billion"),
    "pm25":  ("88101", 88101, "Micrograms/cubic meter (LC)"),
    "pm25n": ("88502", 88502, "Micrograms/cubic meter (LC)"),
    "pm10":  ("81102", 81102, "Micrograms/cubic meter (25 C)"),
    "pmc":   ("86101", 86101, "Micrograms/cubic meter (25 C)"),
    "wspd":  ("WIND",  61103, "Knots"),
    "wdir":  ("WIND",  61104, "Degrees Compass"),
    "temp":  ("TEMP",  62101, "Degrees Fahrenheit"),
    "rh":    ("RH_DP", 62201, "Percent relative humidity"),
    "dewp":  ("RH_DP", 62103, "Degrees Fahrenheit"),
}

USECOLS = ["State Code", "County Code", "Site Num", "Parameter Code", "POC",
           "Latitude", "Longitude", "Datum", "Date Local", "Time Local",
           "Date GMT", "Time GMT", "Sample Measurement", "Units of Measure",
           "Qualifier", "Method Code", "State Name", "County Name"]

DTYPES = {"State Code": "string", "County Code": "string", "Site Num": "string",
          "Parameter Code": "int32", "POC": "int16", "Latitude": "float64",
          "Longitude": "float64", "Datum": "string", "Date Local": "string",
          "Time Local": "string", "Date GMT": "string", "Time GMT": "string",
          "Sample Measurement": "float32", "Units of Measure": "string",
          "Qualifier": "string", "Method Code": "string",
          "State Name": "string", "County Name": "string"}

# States and territories outside the contiguous United States (80 is Mexico,
# which appears in the border network).
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78", "80", "CC"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Streaming read of a remote zip
# ---------------------------------------------------------------------------

# The EPA server rate-limits requests: one request immediately after another
# can return 503 even if the previous one succeeded. Over a run of several
# files this is the normal case, so retrying with increasing waits is part of
# the procedure.
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# Interruptions in mid-transfer. A deflate stream has no re-entry points, so
# a broken transfer is repeated from the start.
TRANSIENT = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
             TimeoutError, socket.timeout, http.client.IncompleteRead,
             http.client.RemoteDisconnected, urllib.error.URLError, EOFError)


def open_url(url: str, timeout: int, retries: int, pause: float):
    """Open the URL, retrying on transient server responses."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "aqs-collocated/1.0 (academic research)",
        "Accept": "application/zip, */*"})
    last = None
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in RETRY_STATUS or attempt == retries:
                raise
            wait = e.headers.get("Retry-After") if e.headers else None
            delay = float(wait) if (wait or "").isdigit() else pause * (2 ** attempt)
        except urllib.error.URLError as e:
            last = e
            if attempt == retries:
                raise
            delay = pause * (2 ** attempt)
        delay += random.uniform(0, pause)
        log(f"    {type(last).__name__} {getattr(last, 'code', '')}: "
            f"retry {attempt + 1} of {retries} in {delay:.0f} s")
        time.sleep(delay)
    raise last


class _ZipStream(io.RawIOBase):
    """Decompress the first member of a remote zip as it streams in.

    Reads the local header (30 bytes plus name and extra field), checks that
    the method is deflate or stored, and decompresses as bytes arrive.
    """

    def __init__(self, url: str, timeout: int = 120, chunk: int = 1 << 20,
                 retries: int = 5, pause: float = 5.0):
        # assigned before the request so that close() does not fail if urlopen
        # raises; otherwise the destructor would mask the real error
        self.resp = None
        self.chunk = chunk
        self.url = url
        self.resp = open_url(url, timeout, retries, pause)
        # A transfer that ends cleanly but early raises no exception: read()
        # returns empty and the CSV is silently truncated. Completeness is
        # therefore checked in two independent ways: the announced byte count
        # and the end of the deflate stream.
        self.n_read = 0
        cl = self.resp.headers.get("Content-Length") if self.resp.headers else None
        self.expected = int(cl) if (cl or "").isdigit() else None
        self.buf = b""
        self.eof = False

        head = self._fill(30)
        if head[:4] != b"PK\x03\x04":
            raise ValueError(f"{url}: does not start with a zip local header")
        method = int.from_bytes(head[8:10], "little")
        nlen = int.from_bytes(head[26:28], "little")
        elen = int.from_bytes(head[28:30], "little")
        self._drop(30)
        meta = self._fill(nlen + elen)
        self.member = meta[:nlen].decode("utf-8", "replace")
        self._drop(nlen + elen)
        if method == 8:
            self.dec = zlib.decompressobj(-zlib.MAX_WBITS)
        elif method == 0:
            self.dec = None          # stored without compression
        else:
            raise ValueError(f"{url}: unsupported compression method {method}")
        self.out = b""

    def _pull(self) -> bool:
        if self.eof:
            return False
        block = self.resp.read(self.chunk)
        if not block:
            self.eof = True
            self._check_complete()
            return False
        self.n_read += len(block)
        self.buf += block
        return True

    def _check_complete(self) -> None:
        """Verify that the whole file arrived when the HTTP body ends."""
        if self.expected is not None and self.n_read < self.expected:
            raise http.client.IncompleteRead(
                b"", self.expected - self.n_read)
        if self.dec is not None and not self.dec.eof:
            raise http.client.IncompleteRead(
                b"", 0)

    def _fill(self, n: int) -> bytes:
        while len(self.buf) < n and self._pull():
            pass
        if len(self.buf) < n:
            raise EOFError("remote file ended before the zip header")
        return self.buf[:n]

    def _drop(self, n: int) -> None:
        self.buf = self.buf[n:]

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self.out:
            if not self.buf and not self._pull():
                if self.dec is not None:
                    self.out = self.dec.flush()
                if not self.out:
                    return 0
                break
            take, self.buf = self.buf, b""
            self.out = self.dec.decompress(take) if self.dec is not None else take
        n = min(len(b), len(self.out))
        b[:n] = self.out[:n]
        self.out = self.out[n:]
        return n

    def close(self) -> None:
        try:
            if getattr(self, "resp", None) is not None:
                self.resp.close()
                self.resp = None
        finally:
            super().close()


def open_remote_csv(url: str, timeout: int, retries: int, pause: float):
    """Return a text stream with the CSV contained in the remote zip."""
    z = _ZipStream(url, timeout, retries=retries, pause=pause)
    return io.TextIOWrapper(io.BufferedReader(z, 1 << 20),
                            encoding="utf-8", errors="replace", newline="")


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def convert(values: np.ndarray, units: pd.Series, var: str) -> np.ndarray:
    """Convert to canonical units: Celsius for temperatures, m/s for wind.

    Pollutant concentrations are kept in the reported unit (ozone in ppm).
    """
    u = units.fillna("").str.lower()
    v = values.astype("float32", copy=True)
    if var in ("temp", "dewp"):
        m = u.str.contains("fahrenheit").to_numpy()
        v[m] = (v[m] - 32.0) * 5.0 / 9.0
    elif var == "wspd":
        m = u.str.contains("knot").to_numpy()
        v[m] = v[m] * 0.514444
        m = u.str.contains("miles per hour").to_numpy()
        v[m] = v[m] * 0.44704
    return v
