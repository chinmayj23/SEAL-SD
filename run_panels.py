"""Run SEAL on the four public longitudinal panels and render the rule-card figures.

Panels: World Bank (output per person), OWID CO2 (emissions per capita),
OWID COVID-19 (weekly deaths per million), EPA (monthly PM2.5).

Descriptors follow the paper. A static covariate contributes its value, a
categorical covariate one indicator per level, and a time-varying covariate its
value at a few reference times plus its net change over the window. The target and
its transformations are excluded from the descriptors.

Data: place the preprocessed panel CSVs under ``data/`` (or set the SEAL_DATA
environment variable to their directory). Outputs per panel go to
``results/<panel>/membership.csv``, ``results/<panel>/rules.txt``, and the figure
``figures/<panel>.pdf``.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seal import run_seal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SEAL_DATA", HERE / "data"))
RESULTS = HERE / "results"
FIG = HERE / "figures"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

PANELS = {  # ds -> (csv, target, title, ylabel, xlabel)
    "worldbank": ("worldbank/worldbank_preprocessed.csv", "target_gdp",
                  "World Bank", "GDP per capita (1990 = 1)", "year"),
    "co2":       ("co2/co2_preprocessed.csv", "target_co2",
                  "CO2 emissions", "CO2 emissions per capita (t)", "year"),
    "covid":     ("covid/covid_preprocessed.csv", "target_deaths_per_million",
                  "COVID mortality", "COVID deaths per million", "week"),
    "epa":       ("epa/epa_preprocessed.csv", "target_pm25",
                  "EPA PM2.5", "PM2.5 (ug/m3)", "month"),
}
FEATURE_YEARS = (1990, 2000, 2010, 2020)
CHANGE_SPAN = {"worldbank": (1990, 2023), "co2": (1990, 2022)}
PARAMS = dict(n_candidates=20000, k_range=[2, 8], tree_depths=[1, 2, 3, 4],
              leaf_fractions=[0.02, 0.30], lambda_complexity=0.005,
              diversity_pool=500, ari_max=0.8, seed=0)


def _entity_names():
    """ISO-code to country-name map for figure labels; empty if the data is absent."""
    try:
        o = pd.read_csv(DATA / "co2" / "owid_co2_data.csv",
                        usecols=["iso_code", "country"]).dropna().drop_duplicates("iso_code")
        return dict(zip(o["iso_code"], o["country"]))
    except Exception:
        return {}


NAME = _entity_names()


def load(ds):
    csv, target, *_ = PANELS[ds]
    df = pd.read_csv(DATA / csv, dtype={"entity": str} if ds == "epa" else None)
    traj = df.pivot(index="entity", columns="time", values=target).sort_index()
    inputs = [c for c in df.columns if c not in ("entity", "time", target)]
    if ds in ("worldbank", "co2"):
        wide = df.pivot(index="entity", columns="time", values=inputs).sort_index()
        years = list(traj.columns)
        feat = {}
        for cov in inputs:
            for y in FEATURE_YEARS:
                yy = y if y in years else min(years, key=lambda c: abs(c - y))
                feat[f"{cov}_{y}"] = wide[(cov, yy)]
            a, b = CHANGE_SPAN[ds]
            aa = a if a in years else min(years)
            bb = b if b in years else max(years)
            feat[f"{cov}_change_{a}_{b}"] = wide[(cov, bb)] - wide[(cov, aa)]
        X = pd.DataFrame(feat, index=traj.index)
        X = X.fillna(X.median())
    elif ds == "covid":
        X = df.groupby("entity")[inputs].first().loc[traj.index]
        X = X.fillna(X.median())
    else:  # epa: numeric site covariates + one-hot categoricals
        raw = df.groupby("entity")[inputs].first().loc[traj.index]
        cat = ["land_use", "location_setting", "state"]
        num = raw.drop(columns=cat).astype(float)
        X = pd.concat([num.fillna(num.median()),
                       pd.get_dummies(raw[cat], prefix={c: f"{c}_is" for c in cat}).astype(float)], axis=1)
    return traj, X


def _prettify(p):
    p = re.sub(r"_change_(\d+)_(\d+)", r" change \1-\2", p.strip())
    p = re.sub(r"_(\d{4})\b", r" in \1", p)
    p = re.sub(r"_is_", " = ", p).replace("_", " ")
    return re.sub(r"([\d.]+e[+\-]?\d+|\d+\.\d+)", lambda m: f"{float(m.group()):.2f}", p)


def _parse_rules(text):
    out = {}
    for seg in text.split("|"):
        seg = seg.strip()
        if seg.startswith("default"):
            continue
        m = re.match(r"G(\d+):\s*(.*)", seg)
        if not m:
            continue
        clauses = re.split(r"\)\s*OR\s*\(", m.group(2).strip())
        out[int(m.group(1))] = "\nor  ".join(
            ",  ".join(_prettify(x) for x in re.split(r"\s+AND\s+", c.strip().strip("()"))) for c in clauses)
    return out


def render(ds, traj, labels, rules_text):
    _, _, title, ylabel, xlabel = PANELS[ds]
    lab = pd.Series(labels, index=traj.index).astype(int)
    sizes = lab.value_counts()
    rules = _parse_rules(rules_text)
    no_rule = [g for g in sizes.index if g not in rules]
    default_id = no_rule[0] if no_rule else int(sizes.idxmax())
    nd = sorted([g for g in sizes.index if g != default_id])
    order = nd + [default_id]
    ncards = len(order)
    colmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(nd)}
    colmap[default_id] = "0.55"
    end = traj[traj.columns[-1]]

    HEADER, RGAP, RLINE, EX, NLINE, GAP, TOP = 0.34, 0.10, 0.205, 0.26, 0.24, 0.26, 0.16
    cards, rk = [], 0
    for g in order:
        if g == default_id:
            head, rule = "Default", None
        else:
            rk += 1
            head = f"Rule {rk}"
            pre = "" if rk == 1 else ("not Rule 1,  " if rk == 2 else f"not Rules 1-{rk-1},  ")
            rule = pre + rules.get(g, "")
        nrl = (1 + rule.count("\n")) if rule else 0
        mem = lab[lab == g].index
        ex = ", ".join(NAME.get(m, str(m)) for m in end[mem].sort_values(ascending=False).index[:5])
        cards.append(dict(g=g, head=head, rule=rule, nrl=nrl, ex=ex, n=len(mem)))

    def ch(c):
        return HEADER + (RGAP + RLINE * c["nrl"] if c["rule"] else 0.08) + EX + NLINE
    right_h = TOP + sum(ch(c) for c in cards) + GAP * (ncards - 1) + 0.15
    fig_h = max(3.0 + 0.62 * ncards, right_h)

    fig = plt.figure(figsize=(11.6, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.3], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    xs = np.arange(len(traj.columns))
    ax.plot(xs, traj.mean(axis=0).to_numpy(), "--", color="black", lw=1.5, zorder=6)
    for g in order:
        ax.plot(xs, traj.loc[lab[lab == g].index].mean(axis=0).to_numpy(),
                color=colmap[g], lw=1.4 if g == default_id else 2.2)
    step = max(1, len(traj.columns) // 6)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([str(traj.columns[i]) for i in range(0, len(traj.columns), step)],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)

    rx = fig.add_subplot(gs[1]); rx.axis("off"); rx.set_xlim(0, 1); rx.set_ylim(0, 1)
    yf = lambda inch: 1.0 - inch / fig_h
    cur = TOP
    for i, c in enumerate(cards):
        col = colmap[c["g"]]
        if i > 0:
            rx.plot([0, 1], [yf(cur - 0.10), yf(cur - 0.10)], color="0.88", lw=0.8)
        rx.plot(0.02, yf(cur + 0.13), "s", color=col, markersize=9, clip_on=False)
        rx.text(0.08, yf(cur + 0.13), c["head"], fontsize=10, fontweight="bold", color=col, va="center")
        cur += HEADER
        if c["rule"]:
            cur += RGAP
            rx.text(0.08, yf(cur), c["rule"], fontsize=6.6, va="top", color="0.12",
                    family="monospace", linespacing=1.3)
            cur += RLINE * c["nrl"]
        else:
            cur += 0.08
        rx.text(0.08, yf(cur), c["ex"], fontsize=8.0, va="top", color="0.12")
        cur += EX
        rx.text(0.08, yf(cur), f"n = {c['n']}", fontsize=8.2, va="top", color=col)
        cur += NLINE + GAP
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{ds}.pdf"); plt.close(fig)


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else list(PANELS)
    for ds in datasets:
        traj, X = load(ds)
        res = run_seal(traj, X, PARAMS)
        b = res["best"]
        d = RESULTS / ds
        d.mkdir(parents=True, exist_ok=True)
        pd.Series(b["labels"], index=traj.index, name="subgroup").rename_axis("entity")\
            .to_csv(d / "membership.csv")
        (d / "rules.txt").write_text(b["rules"], encoding="utf-8")
        render(ds, traj, b["labels"], b["rules"])
        print(f"[{ds}] n={len(traj)} T={traj.shape[1]} descriptors={X.shape[1]} "
              f"cost={b['cost']:.4f} groups={b['n_groups']} -> figures/{ds}.pdf")
