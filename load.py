"""Loaders for each raw source. Each returns a tidy pandas/geopandas frame."""
from pathlib import Path

import geopandas as gpd
import pandas as pd

from config import CA_CRS


def load_zhvi(path: Path) -> pd.DataFrame:
    """Zillow wide CSV -> long panel of California ZIPs: zip, county, month, zhvi."""
    wide = pd.read_csv(path, dtype={"RegionName": str})
    wide = wide[wide["State"] == "CA"].copy()
    wide["zip"] = wide["RegionName"].str.zfill(5)
    date_cols = [c for c in wide.columns if c[:2] in ("19", "20") and "-" in c]
    long = wide.melt(
        id_vars=["zip", "CountyName", "Metro"], value_vars=date_cols,
        var_name="date", value_name="zhvi",
    ).dropna(subset=["zhvi"])
    long["month"] = pd.to_datetime(long["date"]).dt.to_period("M")
    return long.rename(columns={"CountyName": "county", "Metro": "metro"}).drop(columns="date")


def _hazard_column(gdf: gpd.GeoDataFrame) -> str:
    """CAL FIRE files name the class column differently across releases."""
    for col in gdf.columns:
        if col == "geometry":
            continue
        vals = gdf[col].astype(str).str.lower()
        if vals.str.contains("very high|high|moderate").any():
            return col
    raise ValueError(f"No hazard-class column found; columns are {list(gdf.columns)}")


def _normalize_class(v: str) -> str:
    v = str(v).lower()
    if "very high" in v:
        return "Very High"
    if "high" in v:
        return "High"
    if "moderate" in v:
        return "Moderate"
    return "Other"


def load_fhsz(paths) -> gpd.GeoDataFrame:
    """One or more FHSZ layers -> polygons with a normalized hazard_class."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in paths:
        if not Path(p).exists():
            continue
        g = gpd.read_file(p).to_crs(CA_CRS)
        col = _hazard_column(g)
        g["hazard_class"] = g[col].map(_normalize_class)
        frames.append(g[["hazard_class", "geometry"]])
    if not frames:
        raise FileNotFoundError(f"None of the FHSZ files exist: {paths}")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["hazard_class"] != "Other"]
    # Dissolve per class so overlapping SRA/LRA polygons are not double counted
    return gpd.GeoDataFrame(out, crs=CA_CRS).dissolve(by="hazard_class").reset_index()


def load_zcta(path: Path, zips=None) -> gpd.GeoDataFrame:
    """Census ZCTA polygons, optionally limited to a set of ZIPs."""
    g = gpd.read_file(path)
    zcol = next(c for c in g.columns if c.upper().startswith("ZCTA5CE"))
    g["zip"] = g[zcol].astype(str).str.zfill(5)
    if zips is not None:
        g = g[g["zip"].isin(set(zips))]
    return g[["zip", "geometry"]].to_crs(CA_CRS)


def load_fair_plan(path: Path, zip_col=None, count_col=None) -> pd.DataFrame:
    """CDI FAIR Plan policies by ZIP. Column names vary by year, so they can
    be passed in; otherwise the first ZIP-like and first numeric columns are used."""
    df = pd.read_excel(path) if str(path).endswith(("xlsx", "xls")) else pd.read_csv(path)
    zip_col = zip_col or next(c for c in df.columns if "zip" in str(c).lower())
    num = df.select_dtypes("number").columns
    count_col = count_col or next(c for c in num if c != zip_col)
    out = df[[zip_col, count_col]].rename(columns={zip_col: "zip", count_col: "fair_policies"})
    out["zip"] = out["zip"].astype(str).str.extract(r"(\d{5})")[0]
    return out.dropna(subset=["zip"]).groupby("zip", as_index=False)["fair_policies"].sum()
