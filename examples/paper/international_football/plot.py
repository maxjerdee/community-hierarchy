"""International football: the paper figures, from the output of fit.py.

Reads:    results/scores_coupled.csv  [+ results/scores_independent.csv for --diffs]
          data/football_group_stage.gml
          (the *_quick.csv files with --quick, writing figures/*_quick)
Outputs:  figures/football_scores.{png,pdf}  scores by group
          figures/football_diffs.{png,pdf}   coupled vs independent scores  (--diffs)
Run with:
    python plot.py [--diffs] [--quick]

Each group has its own colour; the strongest nation of each is named.
"""
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import community_hierarchy as ch
import matplotlib.patheffects as pe

HERE    = Path(__file__).parent
DATA    = HERE / "data"
QUICK = "--quick" in sys.argv              # results of fit.py's QUICK settings
RESULTS = HERE / "results"
SUFFIX = "_quick" if QUICK else ""          # file-name suffix of results and figures
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

GAP   = 4.3        # gap between groups, in node spacings
FS    = 8

SHORT_NAME = {
    "United States": "USA",                "Bosnia and Herzegovina": "Bosnia",
    "Trinidad and Tobago": "T&T",          "Dominican Republic": "Dom. Rep.",
    "Turks and Caicos Islands": "Turks & Caicos",
    "United States Virgin Islands": "USVI","British Virgin Islands": "BVI",
    "Antigua and Barbuda": "Antigua",      "Saint Kitts and Nevis": "St. Kitts",
    "Saint Vincent and the Grenadines": "St. Vincent",
    "São Tomé and Príncipe": "S. Tomé",    "Central African Republic": "CAR",
    "South Africa": "S. Africa",           "South Korea": "S. Korea",
    "North Korea": "N. Korea",             "Saudi Arabia": "Saudi Arabia",
    "United Arab Emirates": "UAE",         "Republic of Ireland": "Ireland",
    "American Samoa": "Am. Samoa",         "Papua New Guinea": "PNG",
    "Solomon Islands": "Solomon Is.",      "Cook Islands": "Cook Is.",
    "New Caledonia": "N. Caledonia",
}

# ── Load scores and assign region-based group colors / labels ────────────────

G_NET = nx.read_gml(str(DATA / "football_group_stage.gml"), label="label")
_IN_WC = {n.replace("_", " "): int(d.get("in_wc2026", 0)) for n, d in G_NET.nodes(data=True)}


def load_scores(path):
    """node_properties_df() output -> one row per nation (nation, mean, std, group)."""
    if not path.exists():
        sys.exit(f"{path} not found: run `python fit.py` first.")
    d = pd.read_csv(path).rename(columns={"node": "nation", "score_mean": "mean", "score_std": "std"})
    d["nation"] = d["nation"].str.replace("_", " ")
    d["in_wc2026"] = d["nation"].map(_IN_WC).fillna(0).astype(int)
    return d


df = load_scores(RESULTS / f"scores_coupled{SUFFIX}.csv")

# one colour per group, in order of mean score (shared by both figures)
_order = df.groupby("group")["mean"].mean().sort_values().index
group_colors = {g: plt.get_cmap("tab20")(i % 20) for i, g in enumerate(_order)}

# ── Scores by group (community_hierarchy.plot_node_properties) ─────────────────

top = df.sort_values("mean").groupby("group").tail(1)["nation"]       # strongest nation per group
fig, ax = plt.subplots(figsize=(6.75, 4.1))
ch.plot_node_properties(
    df.rename(columns={"nation": "node", "mean": "score_mean", "std": "score_std"}), ax=ax,
    palette=group_colors, labels="none", group_gap=GAP,
    annotate={n: SHORT_NAME.get(n, n) for n in top},
    filled=set(df.loc[df.in_wc2026 == 1, "nation"]), annotation_fontsize=7,
    ylabel=r"$s_i$ (team strength)")
ax.yaxis.label.set_size(FS); ax.tick_params(axis="y", labelsize=FS - 1)
ax.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="#555", markerfacecolor="#555", markersize=2.5,
               linewidth=0, label="World Cup 2026 qualifier"),
    plt.Line2D([0], [0], marker="o", color="#555", markerfacecolor="none", markeredgewidth=0.6,
               markersize=2.5, linewidth=0, label="Non-qualifier"),
], loc="upper left", fontsize=7, framealpha=0.85, edgecolor="none", handletextpad=0.4)
for _ext in ("png", "pdf"):
    out = FIGURES / f"football_scores{SUFFIX}.{_ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out}")
plt.close(fig)

# ── Diffs: coupled vs independent scatter ────────────────────────────────────

if "--diffs" in sys.argv:
    na  = df.copy().rename(columns={"mean": "score_na", "std": "std_na"})
    ind = load_scores(RESULTS / f"scores_independent{SUFFIX}.csv")
    ind = ind.rename(columns={"mean": "score_ind", "std": "std_ind"})

    merged = na.merge(ind[["nation", "score_ind", "std_ind"]], on="nation")
    merged["score_diff"] = merged["score_na"] - merged["score_ind"]
    merged = merged.sort_values("score_na").reset_index(drop=True)

    colors = [group_colors.get(int(g), "#aaaaaa") for g in merged["group"]]
    is_wc  = merged["in_wc2026"].values == 1

    fig2, ax2 = plt.subplots(figsize=(3.375, 3.375))

    ax2.scatter(merged.loc[~is_wc, "score_ind"], merged.loc[~is_wc, "score_na"],
                facecolors="none", edgecolors=[c for c, m in zip(colors, ~is_wc) if m],
                s=10, alpha=0.35, linewidths=0.9, zorder=2)
    ax2.scatter(merged.loc[is_wc, "score_ind"], merged.loc[is_wc, "score_na"],
                c=[c for c, m in zip(colors, is_wc) if m],
                s=16, alpha=0.65, linewidths=0, zorder=3)

    lo = merged[["score_ind", "score_na"]].min().min() - 0.3
    hi = merged[["score_ind", "score_na"]].max().max() + 0.3
    ax2.plot([lo, hi], [lo, hi], color="#999999", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
    mid = (lo + hi) / 2; span = hi - lo
    ax2.text(mid - span * 0.22, mid + span * 0.22,
             "coupled > independent\n(group raises score)",
             fontsize=7, color="#777777", ha="center", va="center", rotation=45, style="italic")
    ax2.text(mid + span * 0.22, mid - span * 0.22,
             "coupled < independent\n(group lowers score)",
             fontsize=7, color="#777777", ha="center", va="center", rotation=45, style="italic")

    d = 0.5
    D = ( d, -d, "left",  "top")
    U = (-d,  d, "right", "bottom")
    LABEL_CONFIGS = {"Spain": D, "Argentina": U, "Morocco": D,
                     "Iran": D, "New Zealand": D, "New Caledonia": U, "Czech Republic": U}
    LABEL_DISPLAY = {"Czech Republic": "Czechia"}
    for nation, (dx, dy, ha, va) in LABEL_CONFIGS.items():
        row = merged[merged["nation"] == nation]
        if row.empty: continue
        row = row.iloc[0]
        tx, ty = row["score_ind"] + dx, row["score_na"] + dy
        ax2.plot([row["score_ind"], tx], [row["score_na"], ty], "-",
                 color="#555555", lw=0.5, zorder=4)
        ax2.text(tx, ty, LABEL_DISPLAY.get(nation, nation), fontsize=7, ha=ha, va=va,
                 path_effects=[pe.withStroke(linewidth=1.5, foreground="white")], zorder=5)

    ax2.set_xlabel("Independent model score", fontsize=FS)
    ax2.set_ylabel("Coupled model score", fontsize=FS)
    ax2.tick_params(labelsize=FS - 1)
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi); ax2.set_aspect("equal")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(handles=[
        plt.Line2D([0],[0], marker="o", color="#555", markerfacecolor="#555",
                   markersize=2.5, linewidth=0, label="WC 2026 qualifier"),
        plt.Line2D([0],[0], marker="o", color="#555", markerfacecolor="none",
                   markeredgewidth=0.6, markersize=2.5, linewidth=0, label="Non-qualifier"),
    ], fontsize=7, loc="upper left", frameon=False)
    fig2.subplots_adjust(left=0.16, right=0.97, bottom=0.13, top=0.96)

    for _ext in ("png", "pdf"):
        out2 = FIGURES / f"football_diffs{SUFFIX}.{_ext}"
        fig2.savefig(out2, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {out2}")
    plt.close(fig2)

