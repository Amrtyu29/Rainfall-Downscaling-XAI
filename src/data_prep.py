"""
Build the tabular training set from the gridded NetCDF cube.

Each (time, lat, lon) cell becomes one row:
    [predictors + lat + lon + cyclical month]  ->  rainfall
Ocean / outside-India cells (NaN rainfall) are dropped.
Splitting is temporal (train / val / test) per config.
"""
import numpy as np
import pandas as pd
import xarray as xr

import config as C


def load_cube() -> xr.Dataset:
    ds = xr.open_dataset(C.MODEL_INPUT)
    return ds


def cube_to_table(ds: xr.Dataset) -> pd.DataFrame:
    """Flatten the cube to a tidy DataFrame, dropping cells with no rainfall obs."""
    df = ds.to_dataframe().reset_index()
    # keep only valid land cells (rainfall observed)
    df = df.dropna(subset=[C.TARGET] + C.PREDICTORS).copy()
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    # cyclical encoding of month so Dec(12) is adjacent to Jan(1)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["season"] = df["month"].map(_month_to_season)
    return df


def _month_to_season(m: int) -> str:
    for name, months in C.SEASONS.items():
        if m in months:
            return name
    return "NA"


def temporal_split(df: pd.DataFrame):
    def _slice(s):
        lo, hi = pd.Timestamp(s[0]), pd.Timestamp(s[1])
        return df[(df["time"] >= lo) & (df["time"] <= hi)]

    return _slice(C.TRAIN_SLICE), _slice(C.VAL_SLICE), _slice(C.TEST_SLICE)


def add_subregion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["subregion"] = "Other"
    for name, (la0, la1, lo0, lo1) in C.SUBREGIONS.items():
        mask = (df.lat >= la0) & (df.lat <= la1) & (df.lon >= lo0) & (df.lon <= lo1)
        df.loc[mask, "subregion"] = name
    return df


if __name__ == "__main__":
    ds = load_cube()
    df = cube_to_table(ds)
    df = add_subregion(df)
    tr, va, te = temporal_split(df)
    print(f"Total rows         : {len(df):,}")
    print(f"  train (<=2010)   : {len(tr):,}")
    print(f"  val   (2011-15)  : {len(va):,}")
    print(f"  test  (2016-23)  : {len(te):,}")
    print(f"Rainfall (target)  : min={df[C.TARGET].min():.2f} "
          f"max={df[C.TARGET].max():.2f} mean={df[C.TARGET].mean():.2f}")
    print("\nRows per season:")
    print(df["season"].value_counts())
    print("\nRows per subregion:")
    print(df["subregion"].value_counts())
    print("\nFeature columns:", C.FEATURES)
