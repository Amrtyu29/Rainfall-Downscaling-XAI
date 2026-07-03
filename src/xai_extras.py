"""
Synopsis completeness: the two remaining XAI methods.

  LIME — local explanation of individual predictions (why THIS month at THIS
         place was predicted wet/dry), complementing SHAP's global view.
  PDP  — partial dependence: the marginal effect of each predictor on
         predicted rainfall.

Outputs in outputs/xai_extras/.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data_prep as D

OUT = C.ROOT / "outputs" / "xai_extras"
OUT.mkdir(parents=True, exist_ok=True)

N_BG = 100_000          # background sample for LIME / PDP averaging
PDP_VARS = ["q850", "rh850", "mslp", "t2m", "u850", "z500"]


def load():
    with open(C.MODELS_DIR / "xgboost.pkl", "rb") as f:
        model = pickle.load(f)
    df = D.add_subregion(D.cube_to_table(D.load_cube()))
    tr, _, te = D.temporal_split(df)
    bg = tr.sample(n=min(N_BG, len(tr)), random_state=C.RANDOM_STATE)
    return model, bg, te


def pdp(model, bg):
    """Manual partial dependence on a background sample (model-agnostic)."""
    X = bg[C.FEATURES].reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, var in zip(axes.ravel(), PDP_VARS):
        grid = np.linspace(X[var].quantile(0.02), X[var].quantile(0.98), 25)
        sub = X.sample(n=4000, random_state=C.RANDOM_STATE).copy()
        means = []
        for g in grid:
            sub[var] = g
            means.append(model.predict(sub[C.FEATURES].values).mean())
        ax.plot(grid, means, lw=2.2, color="#2166ac")
        ax.set_xlabel(var)
        ax.set_ylabel("predicted rainfall (mm/month)")
        ax.grid(alpha=0.3)
    fig.suptitle("Partial Dependence — marginal effect of each predictor (XGBoost)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "pdp_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved pdp_plots.png", flush=True)


def lime_cases(model, bg, te):
    """LIME local explanations for three contrasting real cases."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        bg[C.FEATURES].values, feature_names=C.FEATURES,
        mode="regression", discretize_continuous=True, random_state=C.RANDOM_STATE)

    cases = {
        "Wet: Western Ghats, July 2019":
            te[(te.subregion == "Western Ghats") & (te.month == 7) &
               (te.year == 2019)].nlargest(1, C.TARGET),
        "Dry: Arid Rajasthan, January 2019":
            te[(te.subregion == "Arid Rajasthan") & (te.month == 1) &
               (te.year == 2019)].nsmallest(1, C.TARGET),
        "Moderate: Indo-Gangetic, Sept 2020":
            te[(te.subregion == "Indo-Gangetic") & (te.month == 9) &
               (te.year == 2020)].sample(1, random_state=C.RANDOM_STATE),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, row) in zip(axes, cases.items()):
        row = row.iloc[0]
        exp = explainer.explain_instance(
            row[C.FEATURES].values.astype(float),
            lambda x: model.predict(x), num_features=7)
        pairs = exp.as_list()
        labels = [p[0] for p in pairs][::-1]
        vals = [p[1] for p in pairs][::-1]
        ax.barh(labels, vals,
                color=["#b2182b" if v > 0 else "#2166ac" for v in vals])
        ax.axvline(0, color="k", lw=0.8)
        pred = model.predict(row[C.FEATURES].values.reshape(1, -1).astype(float))[0]
        ax.set_title(f"{title}\nobs {row[C.TARGET]:.0f} | pred {pred:.0f} mm/mo",
                     fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
    fig.suptitle("LIME — local explanations of individual predictions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "lime_cases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved lime_cases.png", flush=True)


if __name__ == "__main__":
    model, bg, te = load()
    pdp(model, bg)
    lime_cases(model, bg, te)
    print("XAI extras complete", flush=True)
