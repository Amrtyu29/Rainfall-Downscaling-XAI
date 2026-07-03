"""
Publication-quality map styling for all India maps.

Every map in the project goes through india_axes() / finish_map() so that:
  - the IMD 0.25-degree field is drawn with smooth (Gouraud) shading
  - India state boundaries come from data/shapefiles/india_st.shp
  - coastline/extent, gridlines and fonts are consistent
  - files are saved at 300 dpi
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib import ticker

import config as C

SHP_STATES = C.SHAPEFILES / "india_st.shp"
EXTENT = [66.5, 100.0, 6.0, 38.5]          # lon0, lon1, lat0, lat1
DPI = 300

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

_state_geoms = None


def state_geometries():
    global _state_geoms
    if _state_geoms is None:
        _state_geoms = list(shpreader.Reader(str(SHP_STATES)).geometries())
    return _state_geoms


def india_axes(fig, rect=111):
    """Create a PlateCarree axes cropped to India."""
    ax = fig.add_subplot(rect, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    return ax


def draw_boundaries(ax, lw=0.6):
    """Overlay India state borders from the project shapefile."""
    ax.add_geometries(state_geometries(), ccrs.PlateCarree(),
                      facecolor="none", edgecolor="black", linewidth=lw, zorder=5)


def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.4, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlocator = ticker.FixedLocator(np.arange(70, 101, 10))
    gl.ylocator = ticker.FixedLocator(np.arange(10, 39, 10))


def field(ax, piv, cmap="Blues", vmin=None, vmax=None, shading="gouraud", clip=True):
    """Draw a lat/lon pivot table as a smooth field clipped to India's outline."""
    lon = piv.columns.values.astype(float)
    lat = piv.index.values.astype(float)
    mesh = ax.pcolormesh(lon, lat, piv.values, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading=shading, transform=ccrs.PlateCarree())
    if clip:
        clip_to_india(ax, mesh)
    return mesh


def _india_path():
    """Matplotlib Path of the union of all state polygons (incl. holes-free exteriors)."""
    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon
    from matplotlib.path import Path

    geom = unary_union(state_geometries())
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    vertices, codes = [], []
    for poly in polys:
        xy = np.asarray(poly.exterior.coords)
        vertices.extend(xy)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
    return Path(vertices, codes)


_cached_path = None


def clip_to_india(ax, artist):
    """Clip a mesh/image so nothing is drawn outside India's boundary."""
    global _cached_path
    if _cached_path is None:
        _cached_path = _india_path()
    from matplotlib.patches import PathPatch
    patch = PathPatch(_cached_path, transform=ax.transData,
                      facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    artist.set_clip_path(patch)


def finish_map(ax, mesh, title, cbar_label, cbar=True):
    draw_boundaries(ax)
    gridlines(ax)
    ax.set_title(title, pad=10)
    if cbar and mesh is not None:
        cb = plt.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03)
        cb.set_label(cbar_label)


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
