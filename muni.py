"""Muni-spread version of the same test, for when the WRDS MSRB pull arrives.

Inputs
- msrb_ca_trades.csv from WRDS (MSRB transactions). Needs columns: cusip,
  trade_date, yield, par_traded, trade_type_indicator, maturity_date.
  (Rename columns in RENAME below if your export differs.)
- issuer_geography.csv: one row per issuer with cusip6 and zip (or a list of
  ZIPs the issuer's tax base covers, one row each). School districts and
  cities can be mapped from their names in official statements or EMMA.

Spread: without a licensed AAA curve (MMD/BVAL), each trade's yield is compared
with the median yield of all CA trades on the same day in the same maturity
bucket. That removes rate moves and the curve, leaving the issuer-specific part.
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

RENAME = {}  # e.g. {"yld": "yield", "par_traded_amt": "par_traded"}
MATURITY_BUCKETS = [0, 2, 5, 10, 15, 20, 30, 100]


def relative_spreads(trades: pd.DataFrame, min_par=100_000) -> pd.DataFrame:
    t = trades.rename(columns=RENAME).copy()
    t["trade_date"] = pd.to_datetime(t["trade_date"])
    t["maturity_date"] = pd.to_datetime(t["maturity_date"])
    # Institutional-size customer buys: the cleanest price signal
    t = t[(t["par_traded"] >= min_par) & (t["trade_type_indicator"].isin(["P", "S"]))]
    t = t.dropna(subset=["yield"])
    t["ttm"] = (t["maturity_date"] - t["trade_date"]).dt.days / 365.25
    t["bucket"] = pd.cut(t["ttm"], MATURITY_BUCKETS)
    bench = t.groupby(["trade_date", "bucket"], observed=True)["yield"].transform("median")
    t["rel_spread_bp"] = 100 * (t["yield"] - bench)
    t["cusip6"] = t["cusip"].str[:6]
    t["month"] = t["trade_date"].dt.to_period("M")
    return t


def issuer_month_panel(t: pd.DataFrame, issuer_map: pd.DataFrame, exposure: pd.DataFrame):
    """Par-weighted issuer-month spreads, with each issuer's exposure averaged over its ZIPs."""
    w = t.assign(wy=t["rel_spread_bp"] * t["par_traded"])
    g = w.groupby(["cusip6", "month"], as_index=False)[["wy", "par_traded"]].sum()
    g["spread_bp"] = g["wy"] / g["par_traded"]
    exp = (issuer_map.merge(exposure[["zip", "exposure"]], on="zip")
           .groupby("cusip6", as_index=False)["exposure"].mean())
    cut = exp.loc[exp["exposure"] > 0, "exposure"].median()
    exp["group"] = np.where(exp["exposure"] == 0, "none",
                            np.where(exp["exposure"] > cut, "high", "low"))
    p = g.merge(exp, on="cusip6")
    p["t"] = p["month"].apply(lambda x: x.ordinal)
    return p


def muni_did(p: pd.DataFrame, event: str, window=12):
    """Spread change (bp) for high- vs. no-exposure issuers after the event,
    with issuer and month fixed effects. Positive = market now charges more."""
    e = pd.Period(event, "M").ordinal
    d = p[p["group"].isin(["high", "none"]) & p["t"].between(e - window, e + window)].copy()
    d["treat_post"] = (d["group"] == "high") * (d["t"] >= e) * 1.0
    d = d.set_index(["cusip6", "t"])
    res = PanelOLS(d["spread_bp"], d[["treat_post"]], entity_effects=True, time_effects=True,
                   check_rank=False).fit(cov_type="clustered", cluster_entity=True)
    return {"event": event, "spread_change_bp": res.params["treat_post"],
            "se": res.std_errors["treat_post"], "p_value": res.pvalues["treat_post"],
            "n_issuers": d.index.get_level_values(0).nunique()}
