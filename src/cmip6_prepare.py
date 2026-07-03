"""
Week 2 of the CMIP6 extension (proposal B.5, Step 2).

1. Regrid raw GCM predictors (data_cmip6/<model>_<exp>.nc) to the 0.25-degree
   analysis grid of model_input.nc (bilinear).
2. Bias-correct each predictor against ERA5 with per-calendar-month,
   per-grid-cell empirical quantile mapping fitted on the 1980-2014 overlap.
   Values outside the fitted range are shifted by the tail offset
   (standard practice for QM extrapolation).

Output: data_cmip6/prepared_<model>_<exp>.nc   (bias-corrected, 0.25 deg)
        + a verification printout comparing corrected historical stats to ERA5.
"""
import numpy as np
import xarray as xr

import config as C

RAW = C.ROOT / "data_cmip6"
GCMS = ["MPI-ESM1-2-HR", "EC-Earth3"]
EXPS = ["historical", "ssp245", "ssp585"]
OVERLAP = slice("1980-01-01", "2014-12-31")


def load_target_grid():
    ds = xr.open_dataset(C.MODEL_INPUT)
    return ds.lat, ds.lon, ds.sel(time=OVERLAP)


def regrid(ds, lat, lon):
    return ds.interp(lat=lat, lon=lon, method="linear")


def fit_apply_qm(gcm_hist, era5, gcm_target):
    """
    Empirical quantile mapping per calendar month per cell.
      gcm_hist, era5 : (time, lat, lon) overlap-period arrays (same length)
      gcm_target     : (time, lat, lon) data to correct (any period)
    """
    out = np.empty_like(gcm_target.values)
    months_h = gcm_hist["time"].dt.month.values
    months_t = gcm_target["time"].dt.month.values
    H, E, T = gcm_hist.values, era5.values, gcm_target.values
    nlat, nlon = H.shape[1], H.shape[2]

    for m in range(1, 13):
        hi = months_h == m
        ti = months_t == m
        Hm = np.sort(H[hi], axis=0)          # (n, lat, lon) sorted climatology
        Em = np.sort(E[hi], axis=0)
        Tm = T[ti]                            # values to correct
        corrected = np.empty_like(Tm)
        for i in range(nlat):
            for j in range(nlon):
                h, e = Hm[:, i, j], Em[:, i, j]
                if np.isnan(h).any() or np.isnan(e).any():
                    corrected[:, i, j] = np.nan
                    continue
                x = Tm[:, i, j]
                y = np.interp(x, h, e)
                # tail extrapolation: shift by end-point offset
                y = np.where(x < h[0], x + (e[0] - h[0]), y)
                y = np.where(x > h[-1], x + (e[-1] - h[-1]), y)
                corrected[:, i, j] = y
        out[ti] = corrected
    return xr.DataArray(out, coords=gcm_target.coords, dims=gcm_target.dims)


def prepare(model):
    lat, lon, era5 = load_target_grid()
    hist_raw = regrid(xr.open_dataset(RAW / f"{model}_historical.nc"), lat, lon)

    for exp in EXPS:
        out_path = RAW / f"prepared_{model}_{exp}.nc"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        if exp != "historical" and not (RAW / f"{model}_{exp}.nc").exists():
            print(f"[skip] {model} {exp}: raw file not downloaded yet")
            continue
        tgt_raw = hist_raw if exp == "historical" else \
            regrid(xr.open_dataset(RAW / f"{model}_{exp}.nc"), lat, lon)
        print(f"\n=== {model} | {exp}: quantile-mapping {len(C.PREDICTORS)} vars ===")
        corrected = {}
        for v in C.PREDICTORS:
            corrected[v] = fit_apply_qm(
                hist_raw[v].sel(time=OVERLAP), era5[v], tgt_raw[v])
            print(f"  {v} done")
        xr.Dataset(corrected).to_netcdf(out_path)
        print(f"  -> saved {out_path.name}")


def verify(model):
    """Corrected historical must match ERA5 climatology closely."""
    lat, lon, era5 = load_target_grid()
    corr = xr.open_dataset(RAW / f"prepared_{model}_historical.nc").sel(time=OVERLAP)
    print(f"\n=== Verification vs ERA5 (1980-2014 means) — {model} ===")
    print(f"{'var':7s} {'ERA5 mean':>10s} {'GCM corrected':>14s} {'diff':>8s}")
    for v in C.PREDICTORS:
        e = float(era5[v].mean())
        g = float(corr[v].mean())
        print(f"{v:7s} {e:10.2f} {g:14.2f} {g - e:8.3f}")


def main():
    for model in GCMS:
        prepare(model)
        verify(model)


if __name__ == "__main__":
    main()
