"""ZIP-level wildfire exposure: share of each ZIP's land area in each hazard class."""
import geopandas as gpd
import pandas as pd

from config import HAZARD_WEIGHTS


def zip_exposure(zcta: gpd.GeoDataFrame, fhsz: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return zip, share_<class> columns, a weighted exposure score, and terciles."""
    zcta = zcta.copy()
    zcta["zip_area"] = zcta.geometry.area
    pieces = gpd.overlay(zcta[["zip", "zip_area", "geometry"]], fhsz, how="intersection",
                         keep_geom_type=True)
    pieces["area"] = pieces.geometry.area
    shares = (
        pieces.groupby(["zip", "hazard_class"])["area"].sum()
        .div(zcta.set_index("zip")["zip_area"], level="zip")
        .unstack(fill_value=0.0)
        .clip(upper=1.0)
    )
    out = zcta[["zip"]].merge(shares, left_on="zip", right_index=True, how="left").fillna(0.0)
    out["exposure"] = sum(out.get(k, 0.0) * w for k, w in HAZARD_WEIGHTS.items())
    out = out.rename(columns={k: f"share_{k.lower().replace(' ', '_')}" for k in HAZARD_WEIGHTS})

    # Groups: zero exposure is the control; exposed ZIPs split at the median
    exposed = out["exposure"] > 0
    cut = out.loc[exposed, "exposure"].median()
    out["group"] = "none"
    out.loc[exposed & (out["exposure"] <= cut), "group"] = "low"
    out.loc[exposed & (out["exposure"] > cut), "group"] = "high"
    return out


def add_fair_plan(exp: pd.DataFrame, fair: pd.DataFrame) -> pd.DataFrame:
    """Attach FAIR Plan policy counts as a second, insurance-based exposure measure."""
    out = exp.merge(fair, on="zip", how="left")
    out["fair_policies"] = out["fair_policies"].fillna(0)
    return out
