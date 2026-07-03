"""
Validation & comparison (publication-quality figures):
  - per-season and per-subregion skill tables (best model)
  - Taylor diagram comparing all models against observations
  - India maps with state boundaries from the project shapefile:
      observed vs predicted rainfall (sample monsoon month),
      per-grid-cell RMSE / Bias over the test period.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data_prep as D
import mapstyle as M

BEST = "XGBoost"          # column suffix in test_predictions.parquet
SAMPLE_MONTH = "2020-07-01"


def _metrics(y, p):
    rmse = np.sqrt(np.mean((p - y) ** 2))
    return dict(RMSE=rmse, MAE=np.mean(np.abs(p - y)),
                Bias=np.mean(p - y),
                R2=1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2),
                PCC=np.corrcoef(y, p)[0, 1], n=len(y))


def skill_tables(pred: pd.DataFrame):
    col = f"pred_{BEST}"
    for by in ["season", "subregion"]:
        rows = []
        for key, g in pred.groupby(by):
            m = _metrics(g[C.TARGET].values, g[col].values); m[by] = key
            rows.append(m)
        t = pd.DataFrame(rows).set_index(by)[["RMSE", "MAE", "Bias", "R2", "PCC", "n"]]
        t.to_csv(C.OUT_PRED / f"skill_by_{by}.csv")
        print(f"\n=== {BEST} skill by {by} ===")
        print(t.round(3).to_string())


def taylor_diagram(pred: pd.DataFrame):
    """Quarter-circle Taylor diagram: radius = normalised std, angle = arccos(corr)."""
    obs = pred[C.TARGET].values
    ref_std = obs.std()
    models = [c[5:] for c in pred.columns if c.startswith("pred_")]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    rmax = 1.6
    ax.set_rmax(rmax); ax.set_rticks([0.5, 1.0, 1.5])
    ax.set_rlabel_position(90)
    ax.set_thetagrids([])          # hide the 10..90 degree labels (only corr matters)

    # correlation rays + labels along the arc
    for corr in [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        th = np.arccos(corr)
        ax.plot([th, th], [0, rmax], color="0.75", ls=":", lw=0.7, zorder=1)
        ax.text(th, rmax * 1.04, f"{corr}", fontsize=9, color="0.35",
                ha="center", va="center", rotation=np.degrees(th) - 90)
    ax.text(np.deg2rad(45), rmax * 1.17, "Correlation", fontsize=11,
            color="0.35", ha="center", rotation=-45)

    # std arcs are the default r-grid; add the reference arc through obs
    th_arc = np.linspace(0, np.pi / 2, 100)
    ax.plot(th_arc, np.ones_like(th_arc), color="k", lw=0.8, ls="--", zorder=1)

    # reference point (observations)
    ax.plot(0, 1.0, "k*", ms=18, zorder=5, label="Observed (reference)")

    for i, mdl in enumerate(models):
        p = pred[f"pred_{mdl}"].values
        corr = np.corrcoef(obs, p)[0, 1]
        ax.plot(np.arccos(corr), p.std() / ref_std, "o", ms=11,
                color=colors[i], zorder=5,
                label=f"{mdl}  (r={corr:.2f})")

    ax.legend(loc="lower left", bbox_to_anchor=(-0.08, -0.18), ncol=2,
              frameon=True, fontsize=10)
    ax.set_title("Taylor diagram — monthly rainfall, test set (2016–2023)",
                 pad=28, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "Radial axis: normalised standard deviation",
             ha="center", fontsize=9, color="0.35")
    M.save(fig, C.OUT_FIG / "taylor_diagram.png")


def spatial_maps(pred: pd.DataFrame):
    """Per-grid-cell RMSE & Bias over the test period, with state boundaries."""
    col = f"pred_{BEST}"
    g = pred.groupby(["lat", "lon"])
    cell = g.apply(lambda d: pd.Series({
        "RMSE": np.sqrt(np.mean((d[col] - d[C.TARGET]) ** 2)),
        "Bias": np.mean(d[col] - d[C.TARGET]),
    })).reset_index()

    fig = plt.figure(figsize=(15, 7))
    for i, (var, cmap) in enumerate([("RMSE", "viridis"), ("Bias", "RdBu_r")], start=1):
        ax = M.india_axes(fig, 120 + i)
        piv = cell.pivot(index="lat", columns="lon", values=var)
        vlim = np.nanpercentile(np.abs(piv.values), 98)
        kw = dict(vmin=-vlim, vmax=vlim) if var == "Bias" else dict(vmin=0, vmax=vlim)
        mesh = M.field(ax, piv, cmap=cmap, **kw)
        M.finish_map(ax, mesh, f"{BEST} {var} — test period (2016–2023)",
                     f"{var} (mm/month)")
    M.save(fig, C.OUT_FIG / "spatial_skill_maps.png")


def sample_month_map():
    """Observed vs predicted rainfall for one monsoon month, with boundaries."""
    with open(C.MODELS_DIR / f"{BEST.lower()}.pkl", "rb") as f:
        model = pickle.load(f)
    df = D.cube_to_table(D.load_cube())
    m = df[df["time"] == pd.Timestamp(SAMPLE_MONTH)].copy()
    m["pred"] = model.predict(m[C.FEATURES].values)

    fig = plt.figure(figsize=(15, 7))
    vmax = np.nanpercentile(m[C.TARGET], 99)
    for i, (var, title) in enumerate(
            [(C.TARGET, "Observed (IMD)"), ("pred", f"Downscaled ({BEST})")], start=1):
        ax = M.india_axes(fig, 120 + i)
        piv = m.pivot(index="lat", columns="lon", values=var)
        mesh = M.field(ax, piv, cmap="YlGnBu", vmin=0, vmax=vmax)
        M.finish_map(ax, mesh, f"{title} — {SAMPLE_MONTH[:7]}", "rainfall (mm/month)")
    M.save(fig, C.OUT_FIG / "observed_vs_predicted_map.png")


def main():
    pred = pd.read_parquet(C.OUT_PRED / "test_predictions.parquet")
    skill_tables(pred)
    taylor_diagram(pred)
    spatial_maps(pred)
    sample_month_map()


if __name__ == "__main__":
    main()
