"""
Extreme-rainfall extension, step 1: daily large-scale predictors.

Downloads NCEP/NCAR Reanalysis DAILY means (public NOAA PSL OPeNDAP, no
account) for the India box, monsoon season (JJAS) 2000-2023. The server
subsets before sending, so each (variable, year) fetch is only a few hundred
kilobytes. Fetches are cached per (variable, year) -> fully resumable.

Output: data_daily/ncep_daily_jjas_2000_2023.nc
        u850, v850, q850, rh850, z500, mslp, t2m on the NCEP 2.5-degree grid
"""
import numpy as np
import pandas as pd
import xarray as xr

import config as C

OUT = C.ROOT / "data_daily"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

BASE = "http://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs"
YEARS = range(2000, 2024)
JJAS = [6, 7, 8, 9]
BOX = dict(lat=slice(40, 4), lon=slice(64, 102))   # NCEP lat is descending

# our name -> (path template, ncep var, level or None)
VARS = {
    "u850": ("pressure/uwnd.{y}.nc", "uwnd", 850),
    "v850": ("pressure/vwnd.{y}.nc", "vwnd", 850),
    "q850": ("pressure/shum.{y}.nc", "shum", 850),
    "rh850": ("pressure/rhum.{y}.nc", "rhum", 850),
    "z500": ("pressure/hgt.{y}.nc", "hgt", 500),
    "mslp": ("surface/slp.{y}.nc", "slp", None),
    "t2m": ("surface/air.sig995.{y}.nc", "air", None),
}
RETRIES = 4


def fetch_var_year(name, year):
    path, ncvar, level = VARS[name]
    cache = CACHE / f"{name}_{year}.nc"
    if cache.exists():
        return xr.open_dataarray(cache)
    url = f"{BASE}/{path.format(y=year)}"
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            ds = xr.open_dataset(url)
            da = ds[ncvar]
            if level is not None:
                da = da.sel(level=level)
            da = da.sel(time=da.time.dt.month.isin(JJAS), **BOX).load()
            da.name = name
            da.to_netcdf(cache)
            return da
        except Exception as e:            # transient OPeNDAP hiccups
            last = e
            print(f"    retry {attempt} {name} {year}: {type(e).__name__}", flush=True)
    raise RuntimeError(f"{name} {year}: {last}")


def main():
    merged = {}
    for name in VARS:
        print(f"[{name}]", flush=True)
        years = []
        for y in YEARS:
            da = fetch_var_year(name, y)
            if "level" in da.coords:          # drop scalar level tag (850 vs 500 conflicts)
                da = da.reset_coords("level", drop=True)
            years.append(da)
            print(f"    {y} ok", flush=True)
        merged[name] = xr.concat(years, dim="time")

    ds = xr.Dataset(merged).sortby("lat")

    # ---- unit normalisation to match the project convention ----
    # q850: NCEP shum is g/kg already if magnitude ~5-20; kg/kg if ~0.005-0.02
    if float(ds.q850.mean()) < 1.0:
        ds["q850"] = ds["q850"] * 1000.0
    # mslp: Pa -> hPa if needed
    if float(ds.mslp.mean()) > 10000:
        ds["mslp"] = ds["mslp"] / 100.0
    print("\nSanity means:", {v: round(float(ds[v].mean()), 2) for v in ds.data_vars},
          flush=True)

    # strip NCEP's conflicting _FillValue / missing_value metadata
    for v in ds.data_vars:
        ds[v].encoding.clear()
        ds[v].attrs.pop("missing_value", None)

    out = OUT / "ncep_daily_jjas_2000_2023.nc"
    ds.to_netcdf(out)
    print(f"saved {out} — {ds.sizes['time']} days, "
          f"{ds.sizes['lat']}x{ds.sizes['lon']} grid", flush=True)


if __name__ == "__main__":
    main()
