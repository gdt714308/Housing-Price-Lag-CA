"""Difference-in-differences and event-study estimates of home values on wildfire exposure.

Specification (ZIP i, county c, month t, event month E):

    log(ZHVI_it) = a_i + g_ct + sum_k b_k * Treat_i * 1[t - E = k] + e_it

a_i: ZIP fixed effects. g_ct: county-by-month fixed effects, so high- and
no-exposure ZIPs are compared only within the same county and month, which
absorbs local economic shocks. k = -1 is the omitted month. Standard errors are
clustered by ZIP. If markets were already pricing wildfire insurance risk, the
b_k after an information shock should be near zero; a drop means the shock
revealed risk that prices had not reflected.
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from config import LAGS, LEADS


def build_panel(prices: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    df = prices.merge(exposure[["zip", "exposure", "group"]], on="zip", how="inner")
    df["log_zhvi"] = np.log(df["zhvi"])
    df["county_month"] = df["county"].astype(str) + "|" + df["month"].astype(str)
    df["t"] = df["month"].apply(lambda p: p.ordinal)
    return df


def _sample(df, event, treated="high", control="none"):
    e = pd.Period(event, "M").ordinal
    d = df[df["group"].isin([treated, control])].copy()
    d["treat"] = (d["group"] == treated).astype(float)
    d["rel"] = (d["t"] - e).clip(-LEADS, LAGS)
    d = d[(d["t"] >= e - LEADS - 12) & (d["t"] <= e + LAGS)]
    return d, e


def _fit(d, regressors):
    p = d.set_index(["zip", "t"])
    mod = PanelOLS(p["log_zhvi"], p[regressors], entity_effects=True,
                   other_effects=p[["county_month"]].astype("category"),
                   drop_absorbed=True, check_rank=False)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def did(df, event, treated="high", control="none"):
    """Pooled 2x2 DiD: average post-event change for treated vs. control ZIPs."""
    d, e = _sample(df, event, treated, control)
    d["treat_post"] = d["treat"] * (d["t"] >= e)
    res = _fit(d, ["treat_post"])
    b, se = res.params["treat_post"], res.std_errors["treat_post"]
    return {
        "event": event, "treated": treated, "control": control,
        "coef_log": b, "se": se, "pct_effect": 100 * (np.exp(b) - 1),
        "ci_low_pct": 100 * (np.exp(b - 1.96 * se) - 1),
        "ci_high_pct": 100 * (np.exp(b + 1.96 * se) - 1),
        "p_value": res.pvalues["treat_post"],
        "n_zips_treated": d.loc[d["treat"] == 1, "zip"].nunique(),
        "n_zips_control": d.loc[d["treat"] == 0, "zip"].nunique(),
        "post_months": int(d["t"].max() - e + 1),
    }


def did_continuous(df, event):
    """Dose-response version: exposure score (0 to 1) times post, all ZIPs."""
    e = pd.Period(event, "M").ordinal
    d = df[(df["t"] >= e - LEADS - 12) & (df["t"] <= e + LAGS)].copy()
    d["exp_post"] = d["exposure"] * (d["t"] >= e)
    res = _fit(d, ["exp_post"])
    return {"event": event, "coef_per_unit_exposure": res.params["exp_post"],
            "se": res.std_errors["exp_post"], "p_value": res.pvalues["exp_post"]}


def event_study(df, event, treated="high", control="none"):
    """Month-by-month coefficients relative to k = -1, plus a joint pre-trend test."""
    d, e = _sample(df, event, treated, control)
    ks = [k for k in sorted(d["rel"].unique()) if k != -1]
    names = []
    for k in ks:
        name = f"k_{'m' if k < 0 else 'p'}{abs(k)}"
        d[name] = d["treat"] * (d["rel"] == k)
        names.append(name)
    res = _fit(d, names)
    rows = [{"k": k, "coef": res.params[n], "se": res.std_errors[n]} for k, n in zip(ks, names)
            if n in res.params.index]
    rows.append({"k": -1, "coef": 0.0, "se": 0.0})
    out = pd.DataFrame(rows).sort_values("k")
    # Exclude the binned endpoint (-LEADS) from the pre-trend test
    leads = [n for k, n in zip(ks, names) if -LEADS < k < -1 and n in res.params.index]
    pre_p = res.wald_test(formula=" = ".join(leads) + " = 0").pval if len(leads) > 1 else np.nan
    return out, pre_p


def group_paths(df, event):
    """Mean log ZHVI by group, normalized to 0 in the month before the event."""
    e = pd.Period(event, "M").ordinal
    d = df[(df["t"] >= e - 24) & (df["t"] <= e + LAGS)]
    m = d.groupby(["group", "t"])["log_zhvi"].mean().unstack(0)
    m = m - m.loc[e - 1]
    m.index = m.index - e
    return 100 * m  # percent (log points)
