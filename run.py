"""Run the full pipeline: exposure scores -> panel -> DiD and event studies -> charts.

    python run.py                # uses the paths in config.py
    python run.py --raw data/raw # point at another raw-data folder
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from exposure import add_fair_plan, zip_exposure
from load import load_fair_plan, load_fhsz, load_zcta, load_zhvi
from model import build_panel, did, did_continuous, event_study, group_paths

INK, MUTED, ACCENT, GRID = "#1f2933", "#7b8794", "#c2410c", "#e4e7eb"
GROUP_COLORS = {"high": "#c2410c", "low": "#e8a87c", "none": "#52606d"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_event_study(es, name, pre_p, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pct = 100 * es["coef"]
    ci = 196 * es["se"]
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(-0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.fill_between(es["k"], pct - ci, pct + ci, color=ACCENT, alpha=0.15, linewidth=0)
    ax.plot(es["k"], pct, color=ACCENT, marker="o", markersize=3.5, linewidth=1.6)
    ax.set_xlabel("Months relative to event", color=INK)
    ax.set_ylabel("High- vs. no-exposure ZIPs (log pts × 100)", color=INK)
    ttl = f"Home values around: {name.replace('_', ' ')}"
    ax.set_title(ttl, loc="left", color=INK, fontsize=11)
    if pd.notna(pre_p):
        ax.text(0.01, 0.03, f"pre-trend test p = {pre_p:.2f}", transform=ax.transAxes,
                color=MUTED, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_paths(paths, name, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = {"high": "High exposure", "low": "Low exposure", "none": "No exposure"}
    for g in ("none", "low", "high"):
        if g in paths:
            ax.plot(paths.index, paths[g], color=GROUP_COLORS[g], linewidth=1.8)
            ax.text(paths.index.max() + 0.4, paths[g].iloc[-1], labels[g],
                    color=GROUP_COLORS[g], fontsize=8, va="center")
    ax.axvline(-0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Months relative to event", color=INK)
    ax.set_ylabel("Mean home value change (%)", color=INK)
    ax.set_title(f"Raw paths by exposure group: {name.replace('_', ' ')}", loc="left",
                 color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(raw: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    f = {k: (raw / Path(v).name if not isinstance(v, list) else [raw / Path(x).name for x in v])
         for k, v in config.FILES.items()}

    prices = load_zhvi(f["zhvi"])
    print(f"ZHVI: {prices['zip'].nunique():,} CA ZIPs, {prices['month'].min()} to {prices['month'].max()}")

    zcta = load_zcta(f["zcta"], zips=prices["zip"].unique())
    fhsz = load_fhsz(f["fhsz"])
    exp = zip_exposure(zcta, fhsz)
    if Path(f["fair_plan"]).exists():
        exp = add_fair_plan(exp, load_fair_plan(f["fair_plan"]))
    exp.to_csv(out / "zip_exposure.csv", index=False)
    print("Exposure groups:", exp["group"].value_counts().to_dict())

    panel = build_panel(prices, exp)
    last = panel["month"].max().ordinal

    did_rows, cont_rows = [], []
    for name, ev in config.EVENTS.items():
        post = last - pd.Period(ev, "M").ordinal + 1
        if post < config.MIN_POST_MONTHS:
            print(f"Skip {name}: {post} post-event month(s) of data so far")
            continue
        r = did(panel, ev)
        r["event_name"] = name
        did_rows.append(r)
        cont_rows.append({**did_continuous(panel, ev), "event_name": name})
        es, pre_p = event_study(panel, ev)
        es.to_csv(out / f"event_study_{name}.csv", index=False)
        plot_event_study(es, name, pre_p, out / f"event_study_{name}.png")
        plot_paths(group_paths(panel, ev), name, out / f"paths_{name}.png")
        print(f"{name}: {r['pct_effect']:+.2f}% (95% CI {r['ci_low_pct']:+.2f} to "
              f"{r['ci_high_pct']:+.2f}), p={r['p_value']:.3f}, pre-trend p={pre_p:.2f}")

    pd.DataFrame(did_rows).to_csv(out / "did_results.csv", index=False)
    pd.DataFrame(cont_rows).to_csv(out / "did_continuous_results.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=config.RAW)
    ap.add_argument("--out", type=Path, default=config.OUT)
    a = ap.parse_args()
    main(a.raw, a.out)
