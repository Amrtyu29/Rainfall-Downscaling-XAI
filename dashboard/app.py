"""
India Rainfall XAI — interactive dashboard (enterprise redesign).

Run:
    cd Rainfall_Downscaling_XAI
    source .venv/bin/activate
    streamlit run dashboard/app.py

Design system: Data-Dense Dashboard · #1E40AF primary / #F8FAFC surface ·
Fira Sans + Fira Code · WCAG-AA contrast · India state boundaries drawn as
vector overlays on every interactive map.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

# ----------------------------------------------------------------- page ----
st.set_page_config(page_title="India Rainfall XAI", page_icon=":material/water_drop:",
                   layout="wide", initial_sidebar_state="expanded")

PRIMARY = "#1E40AF"
INK = "#0F172A"
MUTED = "#475569"
BORDER = "#E2E8F0"
SURFACE = "#FFFFFF"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&family=Fira+Code:wght@500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Fira Sans', -apple-system, sans-serif; }}

/* layout width + top padding (clear Streamlit's fixed header) */
.block-container {{ max-width: 1320px; padding-top: 3.6rem; padding-bottom: 2rem; }}

/* hide default chrome; make the fixed header transparent so it never
   masks the page title underneath */
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---------- hero ---------- */
.hero-title {{ font-size: 1.85rem; font-weight: 700; color: {INK};
  letter-spacing: -0.015em; margin: 0; }}
.hero-sub {{ font-size: 0.98rem; color: {MUTED}; margin: 0.2rem 0 0 0; }}

/* ---------- KPI cards ---------- */
[data-testid="stMetric"] {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,23,42,0.05);
  transition: box-shadow 0.2s ease;
}}
[data-testid="stMetric"]:hover {{ box-shadow: 0 4px 12px rgba(15,23,42,0.10); }}
[data-testid="stMetricValue"] {{ font-family: 'Fira Code', monospace;
  font-size: 1.55rem; font-weight: 600; color: {PRIMARY}; }}
[data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 500; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
  font-size: 15px; font-weight: 600; color: {MUTED};
  padding: 10px 18px; border-radius: 8px 8px 0 0;
}}
.stTabs [aria-selected="true"] {{ color: {PRIMARY}; background: #EFF4FF; }}

/* ---------- cards & images ---------- */
[data-testid="stImage"] img {{ border-radius: 12px; border: 1px solid {BORDER};
  background: {SURFACE}; }}
div[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 12px;
  background: {SURFACE}; }}

/* ---------- status pill ---------- */
.pill {{ display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-size: 0.82rem; font-weight: 600; }}
.pill-test {{ background: #DCFCE7; color: #14532D; }}
.pill-val  {{ background: #E0F2FE; color: #0C4A6E; }}
.pill-train{{ background: #FEF9C3; color: #713F12; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .stMarkdown p {{ font-size: 0.9rem; color: {MUTED}; }}

.takeaway {{ background: #EFF4FF; border: 1px solid #C7D7FE; border-radius: 12px;
  padding: 12px 16px; color: {INK}; font-size: 0.95rem; }}

/* detail panel */
.detail-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 6px 16px 12px; }}
.detail-row {{ display:flex; justify-content:space-between; padding:8px 0;
  border-bottom:1px dashed {BORDER}; font-size:0.9rem; }}
.detail-row:last-child {{ border-bottom:none; }}
.detail-k {{ color:{MUTED}; }}
.detail-v {{ color:{INK}; font-weight:600; font-family:'Fira Code',monospace; }}

/* subtle, tasteful entrance — disabled for reduced-motion users */
@media (prefers-reduced-motion: no-preference) {{
  [data-testid="stPlotlyChart"], [data-testid="stImage"] {{
    animation: fadeUp 0.28s ease-out both;
  }}
  @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(5px); }}
                       to {{ opacity:1; transform:translateY(0); }} }}
}}
</style>
""", unsafe_allow_html=True)

MONTHS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
          6: "June", 7: "July", 8: "August", 9: "September", 10: "October",
          11: "November", 12: "December"}


# ----------------------------------------------------------------- data ----
@st.cache_resource
def load_model():
    with open(C.MODELS_DIR / "xgboost.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_cube():
    return xr.open_dataset(C.MODEL_INPUT).load()


@st.cache_resource
def load_projection(gcm):
    return xr.open_dataset(ROOT / f"outputs/cmip6/{gcm}_rainfall_projection.nc").load()


@st.cache_data
def regional_shap():
    return pd.read_csv(C.OUT_SHAP / "shap_by_region.csv", index_col=0)


@st.cache_data
def seasonal_shap():
    return pd.read_csv(C.OUT_SHAP / "shap_by_season.csv", index_col=0)


@st.cache_data
def change_summary(gcm):
    return pd.read_csv(ROOT / f"outputs/cmip6/change_summary_{gcm}.csv")


# ------------------------------------------------------------- map maker ---
# Maps are rendered with the SAME publication engine used for the report
# figures (src/mapstyle.py): cartopy + state boundaries + clipped to India's
# outline + smooth shading. Each render is cached as PNG bytes, so repeat
# views are instant.
import io                                    # noqa: E402
import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import mapstyle as M                         # noqa: E402


@st.cache_data(show_spinner="Preparing month …")
def month_table(year: int, month: int) -> pd.DataFrame:
    """Predictors + observed + model prediction for one month (tabular)."""
    ds = load_cube()
    sel = ds.sel(time=f"{year}-{month:02d}-01")
    df = sel.to_dataframe().reset_index().dropna(subset=[C.TARGET] + C.PREDICTORS)
    df["month"] = month
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["pred"] = load_model().predict(df[C.FEATURES].values).clip(min=0)
    df["diff"] = df["pred"] - df[C.TARGET]
    return df


def _fig_png(fig, dpi=180) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# Interactive frame around a publication PNG: full Plotly toolbar
# (download, box-zoom, pan, +/-, reset) on the exact cartopy-rendered map.
MAP_TOOLBAR = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "toImageButtonOptions": {"format": "png", "filename": "india_rainfall_map",
                             "scale": 2},
}


def show_map(png_bytes: bytes, height: int = 500, key: str = "map"):
    """Display a rendered map PNG inside a zoomable/pannable Plotly canvas."""
    import base64
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes))
    w, h = im.size
    b64 = base64.b64encode(png_bytes).decode()
    fig = go.Figure()
    fig.add_layout_image(dict(
        source=f"data:image/png;base64,{b64}",
        xref="x", yref="y", x=0, y=h, sizex=w, sizey=h,
        sizing="stretch", layer="below"))
    fig.update_xaxes(visible=False, range=[0, w], constrain="domain")
    fig.update_yaxes(visible=False, range=[0, h], scaleanchor="x")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", dragmode="pan",
                      modebar=dict(bgcolor="rgba(255,255,255,0.9)",
                                   color=MUTED, activecolor=PRIMARY))
    st.plotly_chart(fig, use_container_width=True, config=MAP_TOOLBAR, key=key)


@st.cache_data(show_spinner="Rendering map …")
def render_single_png(year: int, month: int, kind: str) -> bytes:
    """One map (kind: 'obs' or 'pred') — identical style to the report figure.
    Both kinds share the same color scale (99th pct of observed)."""
    df = month_table(year, month)
    vmax = float(np.nanpercentile(df[C.TARGET], 99)) or 10.0
    col, title = ((C.TARGET, "Observed (IMD)") if kind == "obs"
                  else ("pred", "Downscaled (XGBoost)"))
    fig = plt.figure(figsize=(9.5, 8))
    ax = M.india_axes(fig)
    piv = df.pivot(index="lat", columns="lon", values=col)
    mesh = M.field(ax, piv, cmap="YlGnBu", vmin=0, vmax=vmax)
    M.finish_map(ax, mesh, f"{title} — {MONTHS[month]} {year}",
                 "rainfall (mm/month)")
    return _fig_png(fig)


@st.cache_data(show_spinner="Rendering map …")
def render_diff_png(year: int, month: int) -> bytes:
    """Model − Observed difference map, publication style."""
    df = month_table(year, month)
    dlim = float(np.nanpercentile(np.abs(df["diff"]), 98)) or 10.0
    fig = plt.figure(figsize=(9.5, 8))
    ax = M.india_axes(fig)
    piv = df.pivot(index="lat", columns="lon", values="diff")
    mesh = M.field(ax, piv, cmap="RdBu", vmin=-dlim, vmax=dlim)
    M.finish_map(ax, mesh, f"Model − Observed — {MONTHS[month]} {year}",
                 "difference (mm/month)")
    return _fig_png(fig)


@st.cache_data(show_spinner="Rendering map …")
def render_change_png(gcm: str, ssp: str, hz: str) -> bytes:
    """Projected JJAS change map, publication style."""
    proj = load_projection(gcm)
    piv = proj[f"{ssp}_{hz}_change"].to_pandas()
    fig = plt.figure(figsize=(9.5, 8))
    ax = M.india_axes(fig)
    mesh = M.field(ax, piv, cmap="BrBG", vmin=-40, vmax=40)
    M.finish_map(ax, mesh, f"{ssp.upper()} {hz} vs 1985–2014 — {gcm}",
                 "JJAS rainfall change (%)")
    return _fig_png(fig)


# ---- ETCCDI extreme-rainfall indices (standard WMO/RClimDex; guide-requested) ----
OUT_IDX = ROOT / "outputs" / "extremes" / "indices"

INDEX_META = {
    "PRCPTOT": ("Annual wet-day total (PRCPTOT)", "mm", "YlGnBu",
                "Total rain on wet days (≥1 mm) each year — overall wetness."),
    "RX1day": ("Max 1-day rain (RX1day)", "mm", "YlGnBu",
               "Wettest single day of the year — flash-flood intensity."),
    "Rx5day": ("Max 5-day rain (Rx5day)", "mm", "YlGnBu",
               "Wettest 5 consecutive days — river-flood intensity."),
    "R95p": ("Very wet-day total (R95p)", "mm", "YlGnBu",
             "Rain falling on days above the local 95th percentile."),
    "R99p": ("Extremely wet-day total (R99p)", "mm", "YlGnBu",
             "Rain falling on days above the local 99th percentile."),
    "R20": ("Very heavy days ≥20 mm (R20)", "days", "viridis",
            "Number of very heavy rain days per year."),
    "R10": ("Heavy days ≥10 mm (R10)", "days", "viridis",
            "Number of heavy rain days per year."),
    "CDD": ("Consecutive dry days (CDD)", "days", "YlOrRd",
            "Longest dry spell (rain <1 mm) — drought indicator."),
    "CWD": ("Consecutive wet days (CWD)", "days", "GnBu",
            "Longest wet spell (rain ≥1 mm)."),
    "SDII": ("Daily intensity (SDII)", "mm/day", "YlGnBu",
             "Average rain per wet day — rainfall ‘punchiness’."),
}


@st.cache_resource
def load_indices():
    # decode_timedelta=False: CDD/CWD/R10/R20 have units="days", which xarray's
    # CF decoding would otherwise turn into timedelta64 and break rendering.
    return xr.open_dataset(OUT_IDX / "etccdi_indices_annual.nc",
                           decode_timedelta=False).load()


@st.cache_data
def index_series():
    return pd.read_csv(OUT_IDX / "etccdi_annual_series.csv", index_col="year")


@st.cache_data
def index_trends():
    return pd.read_csv(OUT_IDX / "etccdi_trends.csv", index_col="index")


@st.cache_data
def index_trend_field(index):
    """Per-cell OLS slope per decade (vectorised)."""
    ds = load_indices()
    yr = ds["year"].values.astype(float)
    yrc = yr - yr.mean()
    x = ds[index].values
    denom = float((yrc ** 2).sum())
    slope = np.nansum(yrc[:, None, None] * (x - np.nanmean(x, axis=0)[None]),
                      axis=0) / denom
    slope[np.all(~np.isfinite(x), axis=0)] = np.nan
    return slope * 10.0


@st.cache_data(show_spinner="Rendering index map …")
def render_index_png(index, view, year):
    """Publication-style ETCCDI index map (climatology / single year / trend)."""
    ds = load_indices()
    ln, un, cmap, _ = INDEX_META[index]
    short = ln.split("(")[0].strip()
    yr0, yr1 = int(ds.year.min()), int(ds.year.max())
    if view == "Trend (per decade)":
        arr = index_trend_field(index)
        lim = float(np.nanpercentile(np.abs(arr), 96)) or 1.0
        vmin, vmax, cmap = -lim, lim, "RdBu"
        title, cbar = f"{index} trend — {yr0}–{yr1}", f"{un} / decade"
    else:
        if view == "Single year":
            arr = ds[index].sel(year=int(year)).values
            title = f"{index} — {short} · {int(year)}"
        else:
            arr = ds[index].mean("year").values
            title = f"{index} — {short} · {yr0}–{yr1} mean"
        vmin, vmax = np.nanpercentile(arr, [2, 98])
        cbar = un
    piv = xr.DataArray(arr, coords={"lat": ds.lat, "lon": ds.lon},
                       dims=["lat", "lon"]).to_pandas()
    fig = plt.figure(figsize=(9.5, 8))
    ax = M.india_axes(fig)
    mesh = M.field(ax, piv, cmap=cmap, vmin=vmin, vmax=vmax)
    M.finish_map(ax, mesh, title, cbar)
    return _fig_png(fig)


def card_caption(text):
    st.markdown(f"<p style='color:{MUTED}; font-size:0.86rem; margin-top:-6px'>{text}</p>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.markdown(f"<p style='font-weight:700; font-size:1.05rem; color:{INK}; "
                "margin-bottom:0'>India Rainfall XAI</p>", unsafe_allow_html=True)
    st.markdown("Machine-learning downscaling of Indian rainfall, explained with "
                "SHAP and applied to CMIP6 futures and daily extremes.")
    st.divider()
    st.markdown("**Data**")
    st.markdown("ERA5 predictors · IMD 0.25° rainfall · 1980–2023\n\n"
                "CMIP6: MPI-ESM1-2-HR & EC-Earth3\n\n"
                "NCEP/NCAR daily (extremes)")
    st.markdown("**Evaluation**")
    st.markdown("Train 1980–2010 · Val 2011–2015 · **Test 2016–2023** — "
                "all headline numbers are from held-out years.")
    st.divider()
    st.markdown(f"<p style='font-size:0.8rem; color:{MUTED}'>Academic Year 2025–26 · "
                "reproducible pipeline in <code>src/</code></p>", unsafe_allow_html=True)

# ------------------------------------------------------------------ hero ---
st.markdown("<p class='hero-title'>Downscaling Rainfall over India with Explainable AI</p>"
            "<p class='hero-sub'>Accurate · physically verified · applied to India's "
            "climate future</p>", unsafe_allow_html=True)
st.write("")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Downscaling skill (R²)", "0.73", "3.5× vs linear",
          help="LightGBM on held-out 2016–2023; linear baseline R² = 0.21")
k2.metric("Projection-chain validation", "r = 0.965",
          help="GCM-driven downscaled climatology vs observed IMD, 1985–2014")
k3.metric("Monsoon change by 2040–70", "+17…+49%",
          help="All-India JJAS, scenario & model range (SSP2-4.5 / SSP5-8.5)")
k4.metric("Flood-day detection (AUC)", "0.862", "10× vs chance",
          help="Heavy-rain days ≥ 64.5 mm/day, unseen years")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Explore & Predict", "Drivers (XAI)", "Future Climate", "Extremes"])

# ------------------------------------------------------------------- tab 1 --
def region_detail_html(g):
    gb = float(g["diff"].mean()) if len(g) else float("nan")
    gp = float(np.corrcoef(g.pred, g[C.TARGET])[0, 1]) if len(g) > 2 else float("nan")
    rows = [
        ("Grid cells", f"{len(g):,}"),
        ("Observed mean", f"{g[C.TARGET].mean():.0f} mm/mo"),
        ("Model mean", f"{g.pred.mean():.0f} mm/mo"),
        ("Wettest cell (obs)", f"{g[C.TARGET].max():.0f} mm/mo"),
        ("Bias", f"{gb:+.1f} mm/mo"),
        ("Correlation", f"{gp:.3f}"),
    ]
    body = "".join(f"<div class='detail-row'><span class='detail-k'>{k}</span>"
                   f"<span class='detail-v'>{v}</span></div>" for k, v in rows)
    return f"<div class='detail-card'>{body}</div>"


def region_subset(df, region):
    if region == "All India":
        return df
    la0, la1, lo0, lo1 = C.SUBREGIONS[region]
    return df[(df.lat.between(la0, la1)) & (df.lon.between(lo0, lo1))]


with tab1:
    # ---- toolbar ----
    tb = st.columns([2.2, 1.4, 1.9], vertical_alignment="bottom")
    year = tb[0].slider("Year", 1980, 2023, 2020)
    month = tb[1].selectbox("Month", list(MONTHS), index=6,
                            format_func=lambda m: MONTHS[m])
    view = tb[2].radio("View", ["Observed vs Model", "Difference"], horizontal=True)

    df = month_table(year, month)
    rmse = float(np.sqrt(np.mean(df["diff"] ** 2)))
    pcc = float(np.corrcoef(df.pred, df[C.TARGET])[0, 1])
    bias = float(df["diff"].mean())
    period = ("Test period — unseen by the model", "pill-test") if year >= 2016 else \
             ("Validation period", "pill-val") if year >= 2011 else \
             ("Training period", "pill-train")

    # ---- summary cards ----
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Observed mean", f"{df[C.TARGET].mean():.0f} mm/mo",
              help=f"{MONTHS[month]} {year} · all India")
    s2.metric("Spatial correlation", f"{pcc:.3f}")
    s3.metric("RMSE", f"{rmse:.1f} mm/mo")
    s4.metric("Mean bias", f"{bias:+.1f} mm/mo")

    # ---- the maps: publication renderer, exactly like the report figures ----
    st.markdown(f"<span class='pill {period[1]}'>{period[0]}</span>",
                unsafe_allow_html=True)
    if view == "Observed vs Model":
        l, r = st.columns(2, gap="medium")
        with l:
            show_map(render_single_png(year, month, "obs"), height=520, key="map_obs")
        with r:
            show_map(render_single_png(year, month, "pred"), height=520, key="map_pred")
        card_caption("Publication-style rendering — clipped to India, state boundaries, "
                     "shared color scale · each map zooms independently · toolbar "
                     "(top-right): download PNG, box-zoom, pan, +/−, reset · scroll to zoom.")
    else:
        mc = st.columns([1, 2.2, 1])[1]
        with mc:
            show_map(render_diff_png(year, month), height=560, key="map_diff")
        card_caption("Blue = model too wet · red = model too dry · toolbar: zoom, pan, "
                     "reset, download.")

    with st.expander("Region statistics"):
        rc1, rc2 = st.columns([1, 2])
        region = rc1.selectbox("Region", ["All India"] + list(C.SUBREGIONS))
        rc2.markdown(region_detail_html(region_subset(df, region)),
                     unsafe_allow_html=True)

# ------------------------------------------------------------------- tab 2 --
with tab2:
    l, r = st.columns([1.25, 1], gap="large")
    with l:
        st.markdown("**Which variable controls rainfall — everywhere in India**")
        st.image(str(C.OUT_SHAP / "driver_map_india.png"), use_container_width=True)
        card_caption("Dominant driver per grid cell (mean |SHAP|): annual · monsoon · winter. "
                     "Humidity governs rainfall across India; moisture supply rules the "
                     "monsoon, saturation takes the peninsula in winter.")
        with st.expander("Secondary drivers — what matters after humidity"):
            st.image(str(C.OUT_SHAP / "secondary_driver_map.png"),
                     use_container_width=True)
            card_caption("Temperature in the Himalaya · meridional wind in the far "
                         "south · sea-level pressure at the cyclone-prone Odisha coast.")
    with r:
        with st.container(border=True):
            region = st.selectbox("Sub-region", list(C.SUBREGIONS))
            rs = regional_shap()
            if region in rs.columns:
                series = rs[region].sort_values()
                fig = go.Figure(go.Bar(x=series.values, y=series.index,
                                       orientation="h", marker_color=PRIMARY))
                fig.update_layout(height=290, margin=dict(l=0, r=10, t=8, b=0),
                                  xaxis_title="mean |SHAP| (mm/month)",
                                  font=dict(family="Fira Sans", color=INK),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#FFFFFF")
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
        with st.container(border=True):
            st.markdown("**Driver importance by season**")
            ss = seasonal_shap()
            fig = go.Figure()
            palette = ["#1E40AF", "#3B82F6", "#93C5FD", "#F59E0B"]
            for i, season in enumerate(ss.columns):
                fig.add_trace(go.Bar(name=season, x=ss.index, y=ss[season],
                                     marker_color=palette[i % 4]))
            fig.update_layout(barmode="group", height=290,
                              margin=dict(l=0, r=0, t=8, b=0),
                              yaxis_title="mean |SHAP| (mm/month)",
                              font=dict(family="Fira Sans", color=INK),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="#FFFFFF",
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
    st.markdown("<div class='takeaway'><b>Reading:</b> humidity (q850 / rh850) dominates "
                "everywhere and its grip is strongest during the monsoon. Only physical "
                "drivers are shown — location and season encodings are excluded.</div>",
                unsafe_allow_html=True)

# ------------------------------------------------------------------- tab 3 --
with tab3:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.3, 1.5, 1.2, 1.6], vertical_alignment="center")
        gcm = c1.selectbox("Climate model", ["MPI-ESM1-2-HR", "EC-Earth3"])
        ssp = c2.selectbox("Scenario", ["ssp245", "ssp585"],
                           format_func=lambda s: {"ssp245": "SSP2-4.5 · middle path",
                                                  "ssp585": "SSP5-8.5 · high emissions"}[s])
        hz = c3.selectbox("Horizon", ["2040-2070", "2070-2100"])
        c4.markdown("<span class='pill pill-val'>Chain validated: r = 0.965, "
                    "bias +3.6%</span>", unsafe_allow_html=True)

    l, r = st.columns([1.35, 1], gap="medium")
    with l:
        st.markdown(f"**Monsoon (JJAS) rainfall change — {hz} vs 1985–2014**")
        show_map(render_change_png(gcm, ssp, hz), height=540, key="map_change")
        card_caption("Green = wetter · brown = drier · capped at ±40% for readability · "
                     "toolbar: zoom, pan, reset, download.")
    with r:
        cs = change_summary(gcm)
        sub = cs[(cs.scenario == ssp) & (cs.horizon == hz)][["subregion", "JJAS_change_pct"]]
        sub = sub.set_index("subregion").sort_values("JJAS_change_pct")
        fig = go.Figure(go.Bar(
            x=sub.JJAS_change_pct, y=sub.index, orientation="h",
            marker_color=np.where(sub.JJAS_change_pct > 0, "#1E40AF", "#B45309")))
        fig.update_layout(height=430, margin=dict(l=0, r=10, t=8, b=0),
                          xaxis_title="JJAS change (%)",
                          font=dict(family="Fira Sans", color=INK),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF")
        st.markdown("**Regional change**")
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
    with st.expander("Why does the monsoon get wetter? — SHAP attribution"):
        st.image(str(ROOT / f"outputs/cmip6/shap_change_bar_{gcm}.png"),
                 use_container_width=True)
        card_caption("Rising low-level humidity drives the wetter monsoon; temperature "
                     "and pressure act weakly opposite — thermodynamic moistening beats "
                     "dynamic weakening. Far-future SSP5-8.5 magnitudes are upper-end "
                     "(stationarity caveat).")

# ------------------------------------------------------------------- tab 4 --
with tab4:
    sub = st.radio("Extremes view", ["Flood-day drivers", "ETCCDI extreme indices"],
                   horizontal=True, label_visibility="collapsed")

    if sub == "Flood-day drivers":
        l, r = st.columns([1, 1.25], gap="large")
        with l:
            e1, e2 = st.columns(2)
            e1.metric("ROC-AUC — unseen years", "0.862")
            e2.metric("Vs chance", "10×", help="PR-AUC 0.161 against a 1.54% base rate")
            st.markdown("""
**The mechanism switch**

| Timescale | Dominant physics |
|---|---|
| Monthly totals | **Humidity** — thermodynamics |
| Flood-level days | **500 hPa circulation** — monsoon depressions |
""")
            st.markdown("<div class='takeaway'><i>Thermodynamics sets the stage; "
                        "dynamics delivers the flood.</i></div>", unsafe_allow_html=True)
        with r:
            st.markdown("**Dominant driver of heavy-rain days (≥ 64.5 mm/day)**")
            st.image(str(ROOT / "outputs/extremes/extreme_driver_map.png"),
                     use_container_width=True)
            card_caption("JJAS 2016–2023, SHAP on the daily classifier — circulation "
                         "dominates the storm corridor across central and eastern India.")

    else:  # ---- ETCCDI extreme-index explorer ----
        iyrs = load_indices()["year"].values
        iy0, iy1 = int(iyrs.min()), int(iyrs.max())
        with st.container(border=True):
            cc = st.columns([1.9, 2.0, 1.3], vertical_alignment="bottom")
            idx = cc[0].selectbox("Extreme index", list(INDEX_META),
                                  format_func=lambda k: INDEX_META[k][0])
            vw = cc[1].radio("Map", [f"Climatology ({iy0}–{iy1} mean)", "Single year",
                                     "Trend (per decade)"], horizontal=True)
            yr_sel = cc[2].slider("Year", iy0, iy1, iy1, key="idx_year",
                                  disabled=(vw != "Single year"))
        view = ("Trend (per decade)" if vw.startswith("Trend")
                else "Single year" if vw == "Single year" else "Climatology")
        yr_arg = int(yr_sel) if view == "Single year" else 0

        l, r = st.columns([1.4, 1], gap="medium")
        with l:
            show_map(render_index_png(idx, view, yr_arg), height=560, key="map_index")
            note = ("blue = rising · red = falling per decade"
                    if view == "Trend (per decade)"
                    else "publication style · clipped to India · zoom / pan / download via toolbar")
            card_caption(f"{INDEX_META[idx][3]}  ·  {note}")
        with r:
            tr = index_trends().loc[idx]
            un = INDEX_META[idx][1]
            m1, m2 = st.columns(2)
            m1.metric("All-India mean", f"{tr['mean']:.0f} {un}")
            sig = "significant (p<0.05)" if tr["p_value"] < 0.05 else "not significant"
            m2.metric("Trend / decade", f"{tr['slope_per_decade']:+.2f}",
                      help=f"OLS slope · p = {tr['p_value']:.3f} · {sig}")
            st.markdown("**All-India annual trend**")
            s = index_series()[idx]
            yrs = s.index.values.astype(float)
            b = np.polyfit(yrs, s.values, 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yrs, y=s.values, mode="lines+markers",
                                     line=dict(color=PRIMARY, width=2),
                                     marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=yrs, y=b[0] * yrs + b[1], mode="lines",
                                     line=dict(color="#B45309", dash="dash")))
            fig.update_layout(height=300, margin=dict(l=0, r=6, t=6, b=0),
                              yaxis_title=un, showlegend=False,
                              font=dict(family="Fira Sans", color=INK),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False}, key="idx_trend_line")
        st.markdown("<div class='takeaway'><b>Reading:</b> wet-extreme indices (R95p, R99p) "
                    "rise while wet spells (CWD) shorten — rainfall is concentrating into fewer, "
                    "heavier bursts. All-India averages hide stronger regional signals: switch the "
                    "map to <b>Trend (per decade)</b> to see where extremes are increasing.</div>",
                    unsafe_allow_html=True)

st.divider()
st.markdown(f"<p style='font-size:0.8rem;color:{MUTED}'>All statistics computed on "
            "held-out test years unless noted · Downscaling Rainfall over India using "
            "XAI · Academic Year 2025–26</p>", unsafe_allow_html=True)
