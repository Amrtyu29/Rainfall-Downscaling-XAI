"""
Week 1 of the CMIP6 extension (proposal B.5, Step 1).

Downloads monthly CMIP6 predictors from the public Google Cloud archive
(no account needed) for the India box, converts them to the exact variable
names/units of the training cube (model_input.nc), and saves one NetCDF
per (GCM, experiment).

Design: every variable is fetched in its OWN subprocess with a hard timeout.
A hung connection kills only that subprocess; the orchestrator retries with a
completely fresh process. Finished variables are cached to data_cmip6/cache/,
so the download is fully resumable.

Usage:
  python cmip6_download.py                 # orchestrate everything
  python cmip6_download.py --one M EXP V   # (internal) fetch one variable
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config as C

OUT = C.ROOT / "data_cmip6"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

CATALOG = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
CATALOG_LOCAL = OUT / "catalog.csv"
MEMBER = "r1i1p1f1"
GCMS = ["MPI-ESM1-2-HR", "EC-Earth3"]
RUNS = {
    "historical": ("1980-01-01", "2014-12-31"),
    "ssp245": ("2040-01-01", "2100-12-31"),
    "ssp585": ("2040-01-01", "2100-12-31"),
}
BOX = dict(lat=slice(4.0, 40.0), lon=slice(64.0, 102.0))   # padded India box

VAR_TIMEOUT = 900   # seconds per subprocess attempt (bandwidth-limited link)
RETRIES = 8         # retries are cheap now: each resumes from cached pieces

# (cmip6 var, pressure level Pa or None)  ->  (our name, unit conversion)
VARMAP = {
    ("ua", 85000): ("u850", lambda x: x),
    ("va", 85000): ("v850", lambda x: x),
    ("hus", 85000): ("q850", lambda x: x * 1000.0),   # kg/kg -> g/kg
    ("hur", 85000): ("rh850", lambda x: x),           # %
    ("zg", 50000): ("z500", lambda x: x),             # m
    ("psl", None): ("mslp", lambda x: x / 100.0),     # Pa -> hPa
    ("tas", None): ("t2m", lambda x: x),              # K
}
BYNAME = {name: (var, plev, conv) for (var, plev), (name, conv) in VARMAP.items()}


def catalog():
    if not CATALOG_LOCAL.exists():
        pd.read_csv(CATALOG).to_csv(CATALOG_LOCAL, index=False)
    return pd.read_csv(CATALOG_LOCAL)


def fetch_one(model, exp, name):
    """Runs inside a dedicated subprocess: fetch one variable, write cache file."""
    import gcsfs
    var, plev, conv = BYNAME[name]
    t0, t1 = RUNS[exp]
    cat = catalog()
    row = cat[(cat.source_id == model) & (cat.experiment_id == exp) &
              (cat.table_id == "Amon") & (cat.variable_id == var) &
              (cat.member_id == MEMBER)]
    if len(row) == 0:
        raise RuntimeError(f"not in catalog: {model} {exp} {var} {MEMBER}")
    gcs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(gcs.get_mapper(row.zstore.values[0]), consolidated=True)
    da = ds[var]
    # some CMIP6 stores have out-of-order time axes -> sort before slicing
    if not da.indexes["time"].is_monotonic_increasing:
        da = da.sortby("time")
    da = da.sel(time=slice(t0, t1))
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    da = da.sel(**BOX)
    if plev is not None:
        da = da.sel(plev=plev, method="nearest")
        da = da.reset_coords("plev", drop=True)
    # The archive's chunks span all pressure levels globally (~90-170 MB each,
    # ~4 GB per future variable). Load in small time batches, caching each
    # piece: a timeout/retry resumes mid-variable instead of restarting from 0.
    BATCH = 30  # months per piece — aligned to the archive's 30-month chunks
    nt = da.sizes["time"]
    parts = []
    for k in range(0, nt, BATCH):
        part_file = CACHE / f".part_{model}_{exp}_{name}_{k:04d}.nc"
        if part_file.exists():
            parts.append(xr.open_dataarray(part_file))
            continue
        piece = conv(da.isel(time=slice(k, k + BATCH))).astype("float32").load()
        piece.name = name
        piece.to_netcdf(part_file)
        parts.append(piece)
    full = xr.concat(parts, dim="time")
    full.name = name
    tmp = CACHE / f".tmp_{model}_{exp}_{name}.nc"
    full.to_netcdf(tmp)
    tmp.rename(CACHE / f"{model}_{exp}_{name}.nc")
    for k in range(0, nt, BATCH):   # clean up pieces
        (CACHE / f".part_{model}_{exp}_{name}_{k:04d}.nc").unlink(missing_ok=True)


def _download_task(args):
    m, e, name = args
    for attempt in range(1, RETRIES + 1):
        try:
            subprocess.run(
                [sys.executable, "-u", __file__, "--one", m, e, name],
                timeout=VAR_TIMEOUT, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  [ok] {m} {e} {name} (attempt {attempt})", flush=True)
            return True
        except subprocess.TimeoutExpired:
            print(f"  [timeout] {m} {e} {name} attempt {attempt}", flush=True)
        except subprocess.CalledProcessError:
            print(f"  [error]   {m} {e} {name} attempt {attempt}", flush=True)
    print(f"  [FAILED]  {m} {e} {name} — continuing with the rest", flush=True)
    return False


def orchestrate():
    from concurrent.futures import ThreadPoolExecutor

    todo = [(m, e, name) for m in GCMS for e in RUNS
            for name in BYNAME
            if not (CACHE / f"{m}_{e}_{name}.nc").exists()]
    print(f"{len(todo)} variables to download "
          f"({len(GCMS) * len(RUNS) * len(BYNAME) - len(todo)} already cached)",
          flush=True)

    # 2 isolated downloads in parallel: enough to hide slow stores without
    # saturating a bandwidth-limited connection (4 workers made every
    # transfer slower than the timeout)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_download_task, todo))

    # assemble per-(model, experiment) datasets from cache
    for m in GCMS:
        for e in RUNS:
            path = OUT / f"{m}_{e}.nc"
            files = [CACHE / f"{m}_{e}_{n}.nc" for n in BYNAME]
            if path.exists():
                print(f"[skip] {path.name} exists", flush=True)
                continue
            if not all(f.exists() for f in files):
                missing = [n for n in BYNAME if not (CACHE / f"{m}_{e}_{n}.nc").exists()]
                print(f"[incomplete] {m} {e}: missing {missing}", flush=True)
                continue
            ds = xr.merge([xr.open_dataarray(f) for f in files])
            ds["time"] = pd.to_datetime(
                [f"{t.year}-{t.month:02d}-01" for t in ds.indexes["time"]])
            ds.to_netcdf(path)
            print(f"[saved] {path.name} ({ds.sizes['time']} months, "
                  f"{ds.sizes['lat']}x{ds.sizes['lon']} grid)", flush=True)
    print("Orchestration finished.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--one":
        fetch_one(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        orchestrate()
