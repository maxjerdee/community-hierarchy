"""International football: the paper figures, from the output of fit.py.

Reads:    results/scores_coupled.csv  [+ results/scores_independent.csv for --diffs]
          data/football_group_stage.gml
Outputs:  figures/football_scores.{png,pdf}  scores by group (A) + world map (B)
          figures/football_diffs.{png,pdf}   coupled vs independent scores  (--diffs)
          figures/football_adj.{png,pdf}     adjacency matrix by group       (--adj)
Run with:
    python plot.py [--diffs] [--adj]

Group colours and labels are assigned automatically from the confederations of
each group's members, so they follow whatever partition the fit returns.  The
map (panel B) needs geopandas and downloads Natural Earth boundaries (public
domain) into .cache/ on first use.
"""
import sys, io, zipfile, urllib.request
from pathlib import Path
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE    = Path(__file__).parent
DATA    = HERE / "data"
RESULTS = HERE / "results"
CACHE   = HERE / ".cache"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

SCALE = 0.35
GAP   = 1.5
FS    = 8

# ── Region/confederation-based color & label assignment ──────────────────────
# Color families: blue=UEFA, orange-red=CAF, crimson=CONCACAF, green=AFC,
#                 teal=OFC, gold=global-elite.  Darker shade → higher skill.

_CONF_LOOKUP: dict[str, str] = {}
for _n in [
    "Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium",
    "Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
    "Denmark","England","Estonia","Faroe Islands","Finland","France","Georgia",
    "Germany","Gibraltar","Greece","Hungary","Iceland","Republic of Ireland",
    "Israel","Italy","Kazakhstan","Kosovo","Latvia","Liechtenstein","Lithuania",
    "Luxembourg","Malta","Moldova","Montenegro","Netherlands","North Macedonia",
    "Northern Ireland","Norway","Poland","Portugal","Romania","Russia",
    "San Marino","Scotland","Serbia","Slovakia","Slovenia","Spain","Sweden",
    "Switzerland","Turkey","Ukraine","Wales",
]: _CONF_LOOKUP[_n] = "UEFA"

for _n in [
    "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon",
    "Cape Verde","Central African Republic","Chad","Comoros","Congo","DR Congo",
    "Djibouti","Egypt","Equatorial Guinea","Eritrea","Eswatini","Ethiopia",
    "Gabon","Gambia","Ghana","Guinea","Guinea-Bissau","Ivory Coast","Kenya",
    "Lesotho","Liberia","Libya","Madagascar","Malawi","Mali","Mauritania",
    "Mauritius","Morocco","Mozambique","Namibia","Niger","Nigeria","Rwanda",
    "São Tomé and Príncipe","Senegal","Seychelles","Sierra Leone","Somalia",
    "South Africa","South Sudan","Sudan","Tanzania","Togo","Tunisia","Uganda",
    "Zambia","Zimbabwe",
]: _CONF_LOOKUP[_n] = "CAF"

for _n in [
    "Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador",
    "Paraguay","Peru","Uruguay","Venezuela",
]: _CONF_LOOKUP[_n] = "CONMEBOL"

_CARIBBEAN = {
    "Anguilla","Antigua and Barbuda","Aruba","Bahamas","Barbados","Bermuda",
    "Bonaire","British Virgin Islands","Cayman Islands","Cuba","Curaçao",
    "Dominica","Dominican Republic","French Guiana","Grenada","Guadeloupe",
    "Guyana","Haiti","Jamaica","Martinique","Montserrat","Puerto Rico",
    "Saint Kitts and Nevis","Saint Lucia","Saint Martin",
    "Saint Vincent and the Grenadines","Sint Maarten","Suriname",
    "Trinidad and Tobago","Turks and Caicos Islands","United States Virgin Islands",
}
_CENTRAL_AMERICA = {
    "Belize","Costa Rica","El Salvador","Guatemala","Honduras","Nicaragua","Panama",
}
_NORTH_AMERICA = {"Canada","Mexico","United States"}

for _n in [
    "Anguilla","Antigua and Barbuda","Aruba","Bahamas","Barbados","Belize",
    "Bermuda","Bonaire","British Virgin Islands","Canada","Cayman Islands",
    "Costa Rica","Cuba","Curaçao","Dominica","Dominican Republic","El Salvador",
    "French Guiana","Grenada","Guadeloupe","Guatemala","Guyana","Haiti",
    "Honduras","Jamaica","Martinique","Mexico","Montserrat","Nicaragua","Panama",
    "Puerto Rico","Saint Kitts and Nevis","Saint Lucia","Saint Martin",
    "Saint Vincent and the Grenadines","Sint Maarten","Suriname",
    "Trinidad and Tobago","Turks and Caicos Islands","United States",
    "United States Virgin Islands",
]: _CONF_LOOKUP[_n] = "CONCACAF"

for _n in [
    "Afghanistan","Australia","Bahrain","Bangladesh","Bhutan","Brunei",
    "Cambodia","China","Chinese Taipei","Taiwan","Guam","Hong Kong","India",
    "Indonesia","Iran","Iraq","Japan","Jordan","Kuwait","Kyrgyzstan","Laos",
    "Lebanon","Macau","Malaysia","Maldives","Mongolia","Myanmar","Nepal",
    "North Korea","Oman","Pakistan","Palestine","Philippines","Qatar",
    "Saudi Arabia","Singapore","South Korea","Sri Lanka","Syria","Tajikistan",
    "Thailand","Timor-Leste","Turkmenistan","United Arab Emirates","Uzbekistan",
    "Vietnam","Yemen",
]: _CONF_LOOKUP[_n] = "AFC"

for _n in [
    "American Samoa","Cook Islands","Fiji","New Caledonia","New Zealand",
    "Papua New Guinea","Samoa","Solomon Islands","Tahiti","Tonga","Vanuatu",
]: _CONF_LOOKUP[_n] = "OFC"

# Palettes ordered dark→light (high skill → low skill within each confederation)
# CAF trends orange (top) → yellow (bottom) for clear label contrast across groups
# Americas (CONMEBOL + CONCACAF) share a unified red→purple ramp, ranked jointly
_CONF_PALETTES: dict[str, list[str]] = {
    "UEFA":     ["#ffc107","#e07820"],        # Europe I gold, Europe II orange
    "CAF":      ["#0277bd","#039be5","#1565c0","#1a0070"],  # cool steel → indigo (Africa I dark blue)
    "AFC":      ["#33691e","#558b2f","#8bc34a","#c5e1a5"],  # B1: olive family (2 groups)
    "OFC":      ["#5d3a1a","#8d5524","#bc8a5f","#d4a574"],  # warm brown (1 group)
}
# Unified Americas ramp: red (north) → purple (south), geographic order
# Label order: N. America → C. America → Caribbean → S. America
_AMERICAS_RAMP = ["#7f0000","#e53935","#880e4f","#4a148c"]  # N.Am dark red → C.Am scarlet → Caribbean wine → S.Am purple
_AMERICAS_GEO_ORDER = ["N. America", "C. America", "Caribbean", "S. America"]
_AMERICAS_CONFS = {"CONMEBOL", "CONCACAF"}
_CONF_DISPLAY: dict[str, str] = {
    "UEFA": "Europe", "CAF": "Africa", "CONMEBOL": "S. America",
    "CONCACAF": "N. America", "AFC": "Asia", "OFC": "Oceania",
}
# Short abbreviations used in compound labels ("Afr./Am." not "Africa/Americas")
_CONF_SHORT: dict[str, str] = {
    "UEFA": "Eur.", "CAF": "Afr.", "CONMEBOL": "S.Am.",
    "CONCACAF": "Am.", "AFC": "Asia.", "OFC": "OFC",
}
_SCORE_TIERS = [
    (2.3,  "(elite)"),
    (1.2,  "(top)"),
    (0.3,  "(upper)"),
    (-0.2, "(mid)"),
    (-1.2, "(lower)"),
    (-3.5, "(weak)"),
    (-9.9, "(micro)"),
]


def _score_tier(mean_score: float) -> str:
    for threshold, label in _SCORE_TIERS:
        if mean_score >= threshold:
            return label
    return "(weak)"


def _assign_group_colors_labels(df: "pd.DataFrame") -> "tuple[dict, dict]":
    """Assign confederation-based colors and geo-sensible non-colliding labels."""
    group_means  = df.groupby("group")["mean"].mean()
    group_counts = df.groupby("group")["nation"].count()

    # Confederation plurality for each group
    group_conf: dict = {}      # group → top confederation string
    group_conf2: dict = {}     # group → second confederation string
    group_conf_pct: dict = {}  # group → fraction in top confederation
    for g in df["group"].unique():
        members = df[df["group"] == g]["nation"].tolist()
        counts  = Counter(_CONF_LOOKUP.get(n, "?") for n in members)
        total   = sum(counts.values())
        ranked  = counts.most_common()
        top_conf = ranked[0][0] if ranked else "?"
        top_pct  = ranked[0][1] / total if ranked else 0.0
        sec_conf = ranked[1][0] if len(ranked) > 1 else None
        group_conf[g]      = top_conf
        group_conf_pct[g]  = top_pct
        group_conf2[g]     = sec_conf

    # Rank groups within each plurality confederation by mean score (desc)
    conf_rank: dict[int, int] = {}
    from itertools import groupby as _groupby
    conf_buckets: dict[str, list] = {}
    for g in df["group"].unique():
        conf_buckets.setdefault(group_conf[g], []).append(g)
    for conf, gs in conf_buckets.items():
        gs_sorted = sorted(gs, key=lambda g: -group_means[g])
        for rank, g in enumerate(gs_sorted):
            conf_rank[g] = rank

    # ── Colors ───────────────────────────────────────────────────────────────
    # Joint Americas rank: geographic north→south order
    # We need base_labels to assign geo rank, but labels aren't built yet.
    # Use a temporary label pass for Americas groups only.
    def _americas_geo_label(g):
        members_set = set(df[df["group"] == g]["nation"].tolist())
        if group_conf[g] == "CONMEBOL":
            return "S. America"
        n_carib = len(members_set & _CARIBBEAN)
        n_cam   = len(members_set & _CENTRAL_AMERICA)
        n_nam   = len(members_set & _NORTH_AMERICA)
        if n_nam > 0 and n_nam >= n_carib and n_nam >= n_cam:
            return "N. America"
        elif n_cam >= n_carib and n_cam > 0:
            return "C. America"
        return "Caribbean"

    americas_rank = {}
    for g in df["group"].unique():
        if group_conf[g] in _AMERICAS_CONFS:
            lbl = _americas_geo_label(g)
            idx = _AMERICAS_GEO_ORDER.index(lbl) if lbl in _AMERICAS_GEO_ORDER else 3
            americas_rank[g] = idx

    colors: dict = {}
    global_elite_groups: set = set()
    for g in df["group"].unique():
        conf = group_conf[g]
        # Top global group (no single confederation dominates) → gold
        if (group_means[g] == group_means.max()
                and group_conf_pct[g] < 0.55
                and group_counts[g] >= 20):
            colors[g] = "#f9a825"
            global_elite_groups.add(g)
        elif conf in _AMERICAS_CONFS:
            rank = americas_rank.get(g, 0)
            colors[g] = _AMERICAS_RAMP[min(rank, len(_AMERICAS_RAMP) - 1)]
        elif conf in _CONF_PALETTES:
            pal  = _CONF_PALETTES[conf]
            rank = conf_rank.get(g, 0)
            colors[g] = pal[min(rank, len(pal) - 1)]
        else:
            colors[g] = "#9e9e9e"

    # ── Labels ───────────────────────────────────────────────────────────────
    base_labels: dict = {}   # group → raw label before disambiguation

    # Pass 1: assign base labels
    for g in sorted(df["group"].unique(), key=lambda g: -group_means[g]):
        n_members   = int(group_counts[g])
        mean_score  = float(group_means[g])
        conf        = group_conf[g]
        conf_pct    = group_conf_pct[g]
        conf2       = group_conf2[g]
        disp        = _CONF_DISPLAY.get(conf, conf)
        short       = _CONF_SHORT.get(conf, conf)

        if n_members == 1:
            base_labels[g] = ""
        elif g in global_elite_groups:
            base_labels[g] = "Global elite"
        elif conf == "CONCACAF" and conf_pct >= 0.50:
            # Split CONCACAF into sub-regions based on group composition
            members_set = set(df[df["group"] == g]["nation"].tolist())
            n_carib  = len(members_set & _CARIBBEAN)
            n_cam    = len(members_set & _CENTRAL_AMERICA)
            n_nam    = len(members_set & _NORTH_AMERICA)
            if n_nam > 0 and n_nam >= n_carib and n_nam >= n_cam:
                base_labels[g] = "N. America"
            elif n_cam >= n_carib and n_cam > 0:
                # Tie or majority C. America → label C. America
                base_labels[g] = "C. America"
            elif n_carib > 0:
                base_labels[g] = "Caribbean"
            else:
                base_labels[g] = disp
        elif conf_pct >= 0.50:
            base_labels[g] = disp       # dominant region name
        else:
            base_labels[g] = short      # abbreviated plurality

    # Pass 2: where a base label appears on ≥2 groups, add Roman numerals
    # I = lowest score, II = next, … (ascending by mean score)
    _ROMAN = ["I", "II", "III", "IV", "V"]
    from collections import defaultdict
    label_groups: dict[str, list] = defaultdict(list)
    for g, lbl in base_labels.items():
        if lbl:
            label_groups[lbl].append(g)

    labels: dict = {}
    for g, lbl in base_labels.items():
        if not lbl:
            labels[g] = ""
            continue
        peers = label_groups[lbl]
        if len(peers) == 1:
            labels[g] = lbl
        else:
            # sort peers by mean score ascending → I is lowest
            peers_sorted = sorted(peers, key=lambda x: group_means[x])
            rank = peers_sorted.index(g)
            labels[g] = f"{lbl} {_ROMAN[rank]}"

    return colors, labels

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

NAME_MAP = {
    "United States":         "United States of America",
    "DR Congo":              "Democratic Republic of the Congo",
    "Congo":                 "Republic of the Congo",
    "Republic of Ireland":   "Ireland",
    "China":                 "People's Republic of China",
    "Timor-Leste":           "East Timor",
    "Gambia":                "The Gambia",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Cape Verde":            "Cabo Verde",
}

NE_SHP    = CACHE / "ne_countries" / "ne_50m_admin_0_countries.shp"
NE_URL    = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
NE_MU_SHP = CACHE / "ne_mapunits" / "ne_10m_admin_0_map_units.shp"
NE_MU_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_map_units.zip"
LON_0     = 11
ROBINSON  = (f"+proj=robin +lon_0={LON_0} +x_0=0 +y_0=0 "
             f"+datum=WGS84 +units=m +no_defs +over")
_FORCE_RIGHT = {"Cook Islands", "French Polynesia"}

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


df = load_scores(RESULTS / "scores_coupled.csv")

group_colors, group_labels = _assign_group_colors_labels(df)

# ── Map helpers ───────────────────────────────────────────────────────────────

def _wrap_antimeridian(gdf, name_col=None):
    import shapely.affinity
    from shapely.geometry import box
    from shapely.ops import unary_union
    cut = LON_0 - 180
    east_clip = box(cut, -90, 180, 90)
    west_clip = box(-180, -90, cut, 90)
    names = list(gdf[name_col]) if name_col and name_col in gdf.columns else [None] * len(gdf)
    new_geoms = []
    for geom, name in zip(gdf.geometry, names):
        if geom is None or geom.is_empty:
            new_geoms.append(geom); continue
        if name in _FORCE_RIGHT:
            new_geoms.append(shapely.affinity.translate(geom, xoff=360)); continue
        e = geom.intersection(east_clip)
        w = geom.intersection(west_clip)
        parts = ([e] if not e.is_empty else []) + (
                 [shapely.affinity.translate(w, xoff=360)] if not w.is_empty else [])
        new_geoms.append(unary_union(parts) if parts else geom)
    result = gdf.copy(); result.geometry = new_geoms; return result


def _ensure_shapefiles():
    for path, url in [(NE_SHP, NE_URL), (NE_MU_SHP, NE_MU_URL)]:
        if not path.exists():
            print(f"Downloading {path.name}…")
            with urllib.request.urlopen(url, timeout=120) as r:
                content = r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(io.BytesIO(content)).extractall(path.parent)


def draw_map(ax, df, group_colors):
    try:
        import geopandas as gpd
    except ImportError:
        ax.text(0.5, 0.5, "geopandas not installed", ha="center", va="center",
                transform=ax.transAxes)
        return (-14.5e6, -5.5e6, 17e6, 8.5e6)

    _ensure_shapefiles()

    geo_to_color = {}
    for _, row in df.iterrows():
        geo = NAME_MAP.get(row["nation"], row["nation"])
        geo_to_color[geo] = group_colors.get(int(row["group"]), "#efefef")

    world      = gpd.read_file(NE_SHP)
    world_proj = _wrap_antimeridian(world, name_col="NAME_EN").to_crs(ROBINSON)
    colors_a0  = [geo_to_color.get(n, "#efefef") for n in world["NAME_EN"]]
    world_proj.plot(color=colors_a0, linewidth=0.2, edgecolor="white", ax=ax)

    mu      = gpd.read_file(NE_MU_SHP)
    mu_proj = _wrap_antimeridian(mu).to_crs(ROBINSON)
    uk_mask  = mu["SOV_A3"] == "GB1"
    uk_proj  = mu_proj[uk_mask.values]
    uk_names = mu.loc[uk_mask, "GEOUNIT"].values
    uk_proj.plot(color=[geo_to_color.get(n, "#efefef") for n in uk_names],
                 linewidth=0.2, edgecolor="white", ax=ax)

    # Centroids
    def centroids(gdf_, name_col_):
        return {row[name_col_]: (row.geometry.centroid.x, row.geometry.centroid.y)
                for _, row in gdf_.iterrows()}
    cents = centroids(world_proj.assign(NAME_EN=world["NAME_EN"].values), "NAME_EN")
    cents.update(centroids(uk_proj.assign(NAME_EN=uk_names), "NAME_EN"))

    # Match network overlay
    nation_group = {NAME_MAP.get(r["nation"], r["nation"]): int(r["group"])
                    for _, r in df.iterrows()}
    G_net = G_NET
    pairs = {}
    for u, v in G_net.edges():
        key = tuple(sorted([NAME_MAP.get(u.replace("_"," "), u.replace("_"," ")),
                             NAME_MAP.get(v.replace("_"," "), v.replace("_"," "))]))
        pairs[key] = pairs.get(key, 0) + 1
    no_ant = world_proj[~world["NAME_EN"].str.contains("Antarc", na=False).values]
    xmin, ymin, xmax, ymax = no_ant.total_bounds
    xmin = max(xmin, -15_000_000); xmax = min(xmax, 18_000_000); ymin = max(ymin, -5_800_000)
    max_count = max(pairs.values()) if pairs else 1
    for (gu, gv), count in pairs.items():
        if gu not in cents or gv not in cents: continue
        x0, y0 = cents[gu]; x1, y1 = cents[gv]
        if not (xmin <= x0 <= xmax and xmin <= x1 <= xmax): continue
        same = (nation_group.get(gu, -1) == nation_group.get(gv, -1) and nation_group.get(gu,-1) != -1)
        lw = 0.1 + 0.15 * (count / max_count)
        ax.plot([x0, x1], [y0, y1], color="#333333", lw=lw,
                alpha=0.05 if same else 0.09, solid_capstyle="round", zorder=1)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_edgecolor("#333333")
    return (xmin, ymin, xmax, ymax)


def draw_scores(ax, df, group_colors, group_labels):
    group_means    = df.groupby("group")["mean"].mean().sort_values()
    ordered_groups = list(group_means.index)

    widths = {g: SCALE * (df["group"] == g).sum() for g in ordered_groups}
    starts = {}; pos = 0.0
    for g in ordered_groups:
        starts[g] = pos; pos += widths[g] + GAP

    group_floor   = {g: (df[df["group"] == g]["mean"] - df[df["group"] == g]["std"]).min()
                     for g in ordered_groups}
    group_ceiling = {g: (df[df["group"] == g]["mean"] + df[df["group"] == g]["std"]).max()
                     for g in ordered_groups}

    nation_x = {}; nation_y = {}

    for g in ordered_groups:
        members  = df[df["group"] == g].sort_values("mean").reset_index(drop=True)
        n        = len(members)
        xs       = starts[g] + np.linspace(0, widths[g], max(n, 1))
        color    = group_colors.get(g, "#aaaaaa")
        alpha_eb = max(0.20, 0.55 - 0.005 * n)
        lw_eb    = max(0.15, 0.45 - 0.004 * n)

        for i, (_, row) in enumerate(members.iterrows()):
            name_raw = row["nation"]
            is_wc    = int(row.get("in_wc2026", 0)) == 1
            ms, mfc, mew = (2.6, color, 0) if is_wc else (2.0, "none", 0.6)
            ax.errorbar(xs[i], row["mean"], yerr=row["std"],
                        fmt="o", color=color, alpha=alpha_eb,
                        markersize=ms, markerfacecolor=mfc,
                        markeredgewidth=mew, markeredgecolor=color,
                        linewidth=lw_eb, capsize=0, zorder=2)
            nation_x[name_raw] = xs[i]; nation_y[name_raw] = row["mean"]

        ax.hlines(group_means[g], xs[0] - 0.12, xs[-1] + 0.12,
                  colors=color, linewidth=1.6, zorder=3)

    # Group labels: each placed a fixed distance below its own group mean score
    LABEL_OFFSET = 2.8   # data units below group mean
    # Per-label y nudges in points (positive = down)
    LABEL_NUDGE_PT: dict[str, float] = {
        "N. America": 3,
        "C. America": 10,
        "Caribbean":  10,
        "S. America": -1,
        "Asia I":     8,
        "Asia II":    2,
        "Africa I":   10,
        "Africa II":  8,
        "Africa III": 3,
        "Africa IV":  9,
    }

    label_ys: list[float] = []
    for g in ordered_groups:
        lbl = group_labels.get(g, "")
        if not lbl:
            continue
        cx = starts[g] + widths[g] / 2
        cy = float(group_means[g]) - LABEL_OFFSET
        nudge = LABEL_NUDGE_PT.get(lbl, 0)
        ax.annotate(lbl, xy=(cx, cy), xytext=(0, -nudge),
                    textcoords="offset points",
                    ha="center", va="top", fontsize=7,
                    color=group_colors.get(g, "#333333"), fontweight="bold",
                    clip_on=False, annotation_clip=False)
        label_ys.append(cy - 0.5)

    # Expand y downward for labels; cap top at data ceiling + small buffer
    ylo, yhi = ax.get_ylim()
    data_max = max(group_ceiling[g] for g in ordered_groups)
    new_lo = min(ylo, min(label_ys)) if label_ys else ylo
    new_hi = data_max + 0.55
    ax.set_ylim(new_lo, new_hi)

    # Top-nation label per group
    for g in ordered_groups:
        top_row  = df[df["group"] == g].sort_values("mean", ascending=False).iloc[0]
        name_raw = top_row["nation"]
        if name_raw not in nation_x: continue
        label = SHORT_NAME.get(name_raw, name_raw)
        ax.annotate(label, xy=(nation_x[name_raw], nation_y[name_raw]), xytext=(0.5, 10),
                    textcoords="offset points", fontsize=7,
                    ha="center", va="bottom", color="#222222",
                    rotation=90, annotation_clip=False,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    ax.axhline(0, color="#aaaaaa", lw=0.7, alpha=0.6, zorder=1)
    ax.set_xticks([]); ax.set_ylabel(r"$s_i$ (team strength)", fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS - 1)
    ax.set_xlim(-GAP / 2, pos - GAP / 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[
        plt.Line2D([0],[0], marker="o", color="#555", markerfacecolor="#555",
                   markersize=2.0, linewidth=0, label="World Cup 2026 qualifier"),
        plt.Line2D([0],[0], marker="o", color="#555", markerfacecolor="none",
                   markeredgewidth=0.6, markersize=2.0, linewidth=0, label="Non-qualifier"),
    ], loc="upper left", fontsize=7, framealpha=0.85, edgecolor="none", handletextpad=0.4)


# ── Main two-panel figure ─────────────────────────────────────────────────────

fig, ax_scores = plt.subplots(figsize=(6.75, 4.1))
fig.subplots_adjust(left=0.090, right=1.0, top=0.990, bottom=0.0)

draw_scores(ax_scores, df, group_colors, group_labels)

_fig_w, _fig_h = fig.get_size_inches()
_pos = ax_scores.get_position()
_ax_w = _fig_w * (_pos.x1 - _pos.x0)
_ax_h = _fig_h * (_pos.y1 - _pos.y0)
_iw   = 0.529
ax_map = ax_scores.inset_axes([1.0 - _iw, 0.0, _iw, 0.5])
bounds = draw_map(ax_map, df, group_colors)
_xmn, _ymn, _xmx, _ymx = bounds
_data_ar = (_xmx - _xmn) / (_ymx - _ymn)
_ih = (_iw * _ax_w) / (_data_ar * _ax_h)
_pt_y = 1 / (_fig_h * 72)

ax_map.set_position([
    _pos.x1 - _iw * _pos.width,
    0.5 * _pt_y,
    _iw * _pos.width,
    _ih * _pos.height,
])
ax_map.set_axes_locator(None)

ax_map.text(0.011, 0.97, "B", transform=ax_map.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left", clip_on=False)

xl, xr = ax_scores.get_xlim()
ax_scores.set_xlim(xl, xr + (xr - xl) * 0.012)

# Place "A" aligned with the ylabel centre x, at top of axes
fig.canvas.draw()
_rend      = fig.canvas.get_renderer()
_fig_bbox  = fig.get_window_extent(_rend)
_yl_bbox   = ax_scores.yaxis.label.get_window_extent(_rend)
_ax_bbox   = ax_scores.get_window_extent(_rend)
_a_x_fig   = (_yl_bbox.x0 + _yl_bbox.x1) * 0.5 / _fig_bbox.width
_a_y_fig   = _ax_bbox.y1 / _fig_bbox.height - 0.02
fig.text(_a_x_fig, _a_y_fig, "A",
         fontsize=11, fontweight="bold", va="top", ha="center",
         transform=fig.transFigure, clip_on=False)

for _ext in ("png", "pdf"):
    out = FIGURES / f"football_scores.{_ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Saved {out}")
plt.close(fig)

# ── Diffs: coupled vs independent scatter ────────────────────────────────────

if "--diffs" in sys.argv:
    na  = df.copy().rename(columns={"mean": "score_na", "std": "std_na"})
    ind = load_scores(RESULTS / "scores_independent.csv")
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
        out2 = FIGURES / f"football_diffs.{_ext}"
        fig2.savefig(out2, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {out2}")
    plt.close(fig2)

# ── Adjacency matrix (bivariate: asymmetry=color, volume=opacity) ─────────────

if "--adj" in sys.argv:
    G_net = G_NET

    # node order: groups sorted by mean score, within group sorted by mean score
    group_means_s = df.groupby("group")["mean"].mean().sort_values()
    ordered_groups_adj = list(group_means_s.index)
    node_order = []
    for g in ordered_groups_adj:
        node_order.extend(
            df[df["group"] == g].sort_values("mean")["nation"].tolist()
        )

    n   = len(node_order)
    idx = {nd: i for i, nd in enumerate(node_order)}
    A   = np.zeros((n, n))
    for u, v, d in G_net.edges(data=True):
        nu = u.replace("_", " "); nv = v.replace("_", " ")
        i, j = idx.get(nu), idx.get(nv)
        if i is not None and j is not None:
            A[i, j] += d.get("weight", 1)

    groups_ord = [int(df.set_index("nation").loc[nd, "group"]) for nd in node_order]
    boundaries = []
    cur = None
    for i, g in enumerate(groups_ord):
        if g != cur:
            boundaries.append(i)
            cur = g
    boundaries.append(n)

    eps      = 1e-9
    asym     = (A - A.T) / (A + A.T + eps)
    # red = row beats col, blue = col beats row
    cmap_rbk = LinearSegmentedColormap.from_list(
        "rbk", [(0.85, 0.10, 0.10), (0.00, 0.00, 0.00), (0.10, 0.25, 0.85)], N=512)
    rgba         = cmap_rbk((asym + 1.0) / 2.0)
    amax         = np.percentile(A[A > 0], 99) if (A > 0).any() else 1.0
    rgba[..., 3] = np.clip(A / amax, 0, 1) ** 0.7
    rgba[A == 0, 3] = 0

    # Short nation labels
    abbrevs = [SHORT_NAME.get(nd, nd[:12]) for nd in node_order]

    BG     = "#f6f6f6"
    BG_rgb = np.array(mcolors.to_rgb(BG))

    fig3, ax3 = plt.subplots(figsize=(7.0, 9.5))
    fig3.subplots_adjust(left=0.14, right=0.97, top=0.78, bottom=0.22)
    ax3.set_facecolor(BG)
    ax3.imshow(rgba, aspect="equal", origin="upper")

    for b in boundaries[1:-1]:
        ax3.axvline(b - 0.5, color="black", linewidth=0.6, zorder=5)
        ax3.axhline(b - 0.5, color="black", linewidth=0.6, zorder=5)

    TICK_FS = 2.2
    ax3.set_xticks(range(n)); ax3.set_yticks(range(n))
    ax3.xaxis.tick_top(); ax3.xaxis.set_label_position("top")
    ax3.set_xticklabels(abbrevs, rotation=90, fontsize=TICK_FS, va="bottom")
    ax3.set_yticklabels(abbrevs, fontsize=TICK_FS)
    ax3.tick_params(axis="x", which="both", pad=12, length=0,
                    top=True, bottom=False, labeltop=True, labelbottom=False)
    ax3.tick_params(axis="y", which="both", pad=10, length=0)
    ax3.set_xlabel("Opponent",  fontsize=7.5, labelpad=14, color="#444")
    ax3.set_ylabel("Nation", fontsize=7.5, labelpad=10, color="#444")

    fig3.canvas.draw()
    p3       = ax3.get_position()
    _fw3, _fh3 = fig3.get_size_inches()
    STRIP_W3 = 0.010; STRIP_H3 = STRIP_W3 * _fw3 / _fh3; STRIP_GAP3 = 0.003
    ax_top3  = fig3.add_axes([p3.x0, p3.y1 + STRIP_GAP3, p3.width, STRIP_H3])
    ax_left3 = fig3.add_axes([p3.x0 - STRIP_W3 - STRIP_GAP3, p3.y0, STRIP_W3, p3.height])
    for ax_s, vertical in [(ax_top3, False), (ax_left3, True)]:
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis("off")
        for k in range(len(boundaries) - 1):
            lo_b, hi_b = boundaries[k], boundaries[k + 1]
            col = group_colors.get(groups_ord[lo_b], "#aaaaaa")
            if not vertical:
                ax_s.add_patch(mpatches.Rectangle(
                    (lo_b / n, 0), (hi_b - lo_b) / n, 1,
                    color=col, transform=ax_s.transAxes))
            else:
                ax_s.add_patch(mpatches.Rectangle(
                    (0, (n - hi_b) / n), 1, (hi_b - lo_b) / n,
                    color=col, transform=ax_s.transAxes))


    for _ext in ("png", "pdf"):
        out3 = FIGURES / f"football_adj.{_ext}"
        fig3.savefig(out3, dpi=300, bbox_inches="tight")
        print(f"Saved {out3}")
    plt.close(fig3)
