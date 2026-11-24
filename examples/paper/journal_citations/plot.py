"""Journal citations: the paper figure, from the output of fit.py.

Reads:    results/scores_coupled.csv, data/journal_citations.gml
Outputs:  figures/journal_scores.{png,pdf}  journal scores by group (A) with the
                                            citation matrix ordered by group (B)
          figures/journal_adj.{png,pdf}     full-size citation matrix   (--adj)
Run with:
    python plot.py [--adj] [--no-inset]

Group names are carried over from the paper's fit: each named group of the paper
(PAPER_GROUPS below) is matched to the inferred group sharing the most journals
with it, so a new fit is labelled consistently even though group indices differ.
Inferred groups with no match are left unlabelled.
"""
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

HERE    = Path(__file__).parent
DATA    = HERE / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)
GML     = DATA / "journal_citations.gml"

adj   = "--adj" in sys.argv
inset = "--no-inset" not in sys.argv

SCALE = 0.28
GAP   = 0.18
FS    = 8                       # as the football figure: y label 8 pt, ticks 7 pt

ABBREV = {
    # ── Flagship / top generalist ──────────────────────────────────────────────
    "Nature":                                           "Nature",
    "Science":                                          "Science",
    "Cell":                                             "Cell",
    "The Lancet":                                       "Lancet",
    "New England Journal of Medicine":                  "NEJM",
    "JAMA":                                             "JAMA",
    "JAMA: The Journal of the American Medical Association": "JAMA",
    "Nature Medicine":                                  "Nat.Med.",
    "Nature Communications":                            "Nat.Comm.",
    "Proceedings of the National Academy of Sciences":  "PNAS",
    # ── Broad open-access / multidisciplinary ─────────────────────────────────
    "Scientific Reports":                               "Sci.Rep.",
    "PLoS ONE":                                         "PLoS ONE",
    "PLOS ONE":                                         "PLoS ONE",
    "Science Advances":                                 "Sci.Adv.",
    "Advanced Science":                                 "Adv.Sci.",
    "eLife":                                            "eLife",
    "Heliyon":                                          "Heliyon",
    "BMJ":                                              "BMJ",
    "BMJ Open":                                         "BMJ Open",
    "The BMJ":                                          "BMJ",
    "JAMA Network Open":                                "JAMA Netw.Open",
    # ── Medicine / clinical ───────────────────────────────────────────────────
    "Journal of Clinical Oncology":                     "JCO",
    "Journal of Clinical Medicine":                     "J.Clin.Med.",
    "Cancers":                                          "Cancers",
    "Circulation":                                      "Circulation",
    "Frontiers in Immunology":                          "Front.Immunol.",
    "Frontiers in Pharmacology":                        "Front.Pharmacol.",
    "Frontiers in Oncology":                            "Front.Oncol.",
    "Frontiers in Psychology":                          "Front.Psychol.",
    "Nutrients":                                        "Nutrients",
    "Cells":                                            "Cells",
    # ── Biology / biochemistry ────────────────────────────────────────────────
    "Nucleic Acids Research":                           "NAR",
    "Cell Reports":                                     "Cell Rep.",
    "Frontiers in Microbiology":                        "Front.Micro.",
    "Frontiers in Plant Science":                       "Front.Plant.Sci.",
    "Molecules":                                        "Molecules",
    "International Journal of Molecular Sciences":      "IJMS",
    "International Journal of Biological Macromolecules": "IJBM",
    "Carbohydrate Polymers":                            "Carbohyd.Polym.",
    # ── Chemistry ─────────────────────────────────────────────────────────────
    "Journal of the American Chemical Society":         "JACS",
    "Angewandte Chemie International Edition":          "Angew.Chem.",
    "Chemical Reviews":                                 "Chem.Rev.",
    "Chemical Society Reviews":                         "Chem.Soc.Rev.",
    "Analytical Chemistry":                             "Anal.Chem.",
    "ACS Catalysis":                                    "ACS Catal.",
    "ACS Nano":                                         "ACS Nano",
    "ACS Applied Materials & Interfaces":               "ACS AMI",
    "ACS Omega":                                        "ACS Omega",
    "ACS Sustainable Chemistry & Engineering":          "ACS SC&E",
    "RSC Advances":                                     "RSC Adv.",
    "Chemical Engineering Journal":                     "Chem.Eng.J.",
    "Applied Catalysis B: Environmental":               "Appl.Catal.B",
    # ── Materials science ─────────────────────────────────────────────────────
    "Advanced Materials":                               "Adv.Mat.",
    "Advanced Functional Materials":                    "Adv.Funct.Mat.",
    "Small":                                            "Small",
    "Journal of Materials Chemistry A":                 "JMCA",
    "Journal of Alloys and Compounds":                  "J.Alloys Comp.",
    "Journal of Colloid and Interface Science":         "JCIS",
    "Applied Surface Science":                          "Appl.Surf.Sci.",
    "Ceramics International":                           "Ceram.Int.",
    "Separation and Purification Technology":           "Sep.Purif.Tech.",
    "Nanomaterials":                                    "Nanomaterials",
    "Nano Energy":                                      "Nano Energy",
    "Materials":                                        "Materials",
    "Materials Today Proceedings":                      "Mat.Today Proc.",
    "Polymers":                                         "Polymers",
    # ── Energy ────────────────────────────────────────────────────────────────
    "Advanced Energy Materials":                        "Adv.Energy Mat.",
    "Journal of Energy Storage":                        "J.Energy Stor.",
    "Renewable Energy":                                 "Renew.Energy",
    "Renewable and Sustainable Energy Reviews":         "RSER",
    "International Journal of Hydrogen Energy":         "IJHE",
    "Applied Energy":                                   "Appl.Energy",
    "Energies":                                         "Energies",
    "Fuel":                                             "Fuel",
    # ── Environment / sustainability ──────────────────────────────────────────
    "Environmental Science & Technology":               "ES&T",
    "The Science of The Total Environment":             "Sci.Tot.Env.",
    "Chemosphere":                                      "Chemosphere",
    "Environmental Pollution":                          "Environ.Pollut.",
    "Environmental Research":                           "Environ.Res.",
    "Environmental Science and Pollution Research":     "ESPR",
    "Journal of Hazardous Materials":                   "J.Hazard.Mat.",
    "Journal of Environmental Management":              "J.Environ.Mgmt.",
    "Journal of Cleaner Production":                    "J.Clean.Prod.",
    "Sustainability":                                   "Sustainability",
    "Bioresource Technology":                           "Bioresour.Tech.",
    "Water Research":                                   "Water Res.",
    "Water":                                            "Water",
    "Construction and Building Materials":              "Constr.Build.Mat.",
    "International Journal of Environmental Research and Public Health": "IJERPH",
    # ── Food / agriculture ────────────────────────────────────────────────────
    "Food Chemistry":                                   "Food Chem.",
    "Foods":                                            "Foods",
    # ── Physics / astronomy ───────────────────────────────────────────────────
    "Physical Review Letters":                          "PRL",
    "Physical review. B./Physical review. B":           "PRB",
    "Physical review. D/Physical review. D.":           "PRD",
    "Journal of Physics Conference Series":             "J.Phys.Conf.",
    "The Astrophysical Journal":                        "ApJ",
    "Monthly Notices of the Royal Astronomical Society": "MNRAS",
    # ── CS / engineering / sensing ────────────────────────────────────────────
    "IEEE Access":                                      "IEEE Access",
    "IEEE Internet of Things Journal":                  "IEEE IoT J.",
    "Expert Systems with Applications":                 "Expert Syst.",
    "Remote Sensing":                                   "Remote Sens.",
    "Sensors":                                          "Sensors",
    "Applied Sciences":                                 "Appl.Sci.",
}

# ── Group names: membership of each named group in the paper's fit ────────────
PAPER_GROUPS: dict[str, list[str]] = {
    "Multidisciplinary": ["Nature", "Science", "Proceedings of the National Academy of Sciences",
                          "Science Advances"],
    "Physics & astro.":  ["Physical Review Letters", "Physical review. B./Physical review. B",
                          "Monthly Notices of the Royal Astronomical Society", "The Astrophysical Journal"],
    "Chemistry II":      ["Advanced Energy Materials", "ACS Catalysis", "Chemical Reviews", "Nano Energy",
                          "Chemical Society Reviews"],
    "OA multi.":         ["Nature Communications", "Scientific Reports"],
    "Chemistry I":       ["Angewandte Chemie International Edition", "Advanced Materials",
                          "Journal of the American Chemical Society", "Advanced Functional Materials",
                          "ACS Nano", "Small", "Advanced Science"],
    "Life sciences":     ["Frontiers in Immunology", "Cell", "Cancers", "Journal of Clinical Medicine",
                          "Cells", "Frontiers in Oncology", "eLife", "Frontiers in Pharmacology",
                          "Cell Reports"],
    "Chem. engineering": ["Chemical Engineering Journal", "ACS Applied Materials & Interfaces"],
    "Analytical chem.":  ["Food Chemistry", "Carbohydrate Polymers", "Analytical Chemistry"],
    "Materials II":      ["Journal of Alloys and Compounds", "Journal of Materials Chemistry A",
                          "Applied Catalysis B: Environmental", "Applied Surface Science",
                          "Ceramics International", "Journal of Colloid and Interface Science"],
    "Environment":       ["Journal of Hazardous Materials", "Chemosphere",
                          "Environmental Science & Technology", "Environmental Research"],
    "Water":             ["Journal of Environmental Management", "Environmental Pollution",
                          "Bioresource Technology", "Water Research", "Water"],
    "Renewables":        ["Construction and Building Materials", "Energies", "Applied Energy",
                          "Renewable and Sustainable Energy Reviews", "Renewable Energy",
                          "Materials Today Proceedings"],
    "OA life":           ["PLoS ONE", "International Journal of Environmental Research and Public Health",
                          "Nutrients", "Remote Sensing", "Frontiers in Microbiology",
                          "Frontiers in Plant Science", "Foods"],
    "Energy":            ["International Journal of Hydrogen Energy", "Fuel", "Journal of Energy Storage",
                          "Separation and Purification Technology", "ACS Sustainable Chemistry & Engineering"],
    "Sustainability":    ["The Science of The Total Environment", "Sustainability",
                          "Journal of Cleaner Production", "Environmental Science and Pollution Research"],
    "Materials I":       ["Materials", "Polymers", "RSC Advances", "ACS Omega", "Nanomaterials"],
    "OA engineering":    ["IEEE Access", "Sensors", "Applied Sciences", "Heliyon"],
    "OA molecular":      ["International Journal of Molecular Sciences", "Molecules",
                          "International Journal of Biological Macromolecules"],
}
# Horizontal nudges of group names (in x units), to clear neighbours / the y axis
LABEL_X_NUDGE = {"Multidisciplinary": -0.8, "OA molecular": 0.35}


def match_group_names(df):
    """Map inferred group -> paper group name, greedily by number of shared journals."""
    members = df.groupby("group")["journal_name"].apply(set)
    pairs = sorted(((len(members[g] & set(js)), g, name)
                    for g in members.index for name, js in PAPER_GROUPS.items()), reverse=True)
    names, used = {}, set()
    for overlap, g, name in pairs:
        if overlap and g not in names and name not in used:
            names[g] = name
            used.add(name)
    return names


# ── Load the fit ──────────────────────────────────────────────────────────────
scores_csv = RESULTS / "scores_coupled.csv"
if not scores_csv.exists():
    sys.exit(f"{scores_csv} not found: run `python fit.py` first.")
df = pd.read_csv(scores_csv)
if "journal_name" not in df.columns:
    df["journal_name"] = df["name"]
if "group_score_mean" not in df.columns:
    df["group_score_mean"] = df.groupby("group")["score_mean"].transform("mean")
df["score_mean_centred"] = df["score_mean"]      # raw scores, no recentering
# Scores are identified up to sign by convention: put the flagship journals on top
_TOP_JOURNALS = {"Nature", "Science"}
_top_mask = df["journal_name"].isin(_TOP_JOURNALS)
if _top_mask.any() and df.loc[_top_mask, "score_mean"].mean() < 0:
    df["score_mean"]          = -df["score_mean"]
    df["score_mean_centred"]  = -df["score_mean_centred"]
    df["group_score_mean"]    = -df["group_score_mean"]
_active_group_labels = match_group_names(df)

group_means    = df.groupby("group")["score_mean_centred"].mean().sort_values()
ordered_groups = list(group_means.index)
cmap_t20       = plt.get_cmap("tab20")
gc             = {g: cmap_t20(i / len(ordered_groups)) for i, g in enumerate(ordered_groups)}

# ── Scatter ───────────────────────────────────────────────────────────────────

x_pos   = {}
x_start = 0.0
for g in ordered_groups:
    members = df[df["group"] == g].sort_values("score_mean_centred")
    for j, (_, row) in enumerate(members.iterrows()):
        x_pos[row["node"]] = x_start + j * SCALE
    x_start += len(members) * SCALE + GAP

fig, ax = plt.subplots(figsize=(6.75, 4.1))     # same size as the football figure
fig.subplots_adjust(left=0.085, right=0.995, top=0.960, bottom=0.0)

for g in ordered_groups:
    sub   = df[df["group"] == g].sort_values("score_mean_centred")
    col   = gc[g]
    xs    = [x_pos[nd] for nd in sub["node"]]
    n_g   = len(sub)
    alpha = max(0.20, 0.55 - 0.005 * n_g)
    lw    = max(0.35, 0.80 - 0.005 * n_g)

    ax.errorbar(xs, sub["score_mean_centred"], yerr=sub["score_std"],
                fmt="none", color=col, elinewidth=lw, capsize=0, alpha=alpha, zorder=3)
    ax.scatter(xs, sub["score_mean_centred"].tolist(),
               color=col, s=12, alpha=min(alpha + 0.2, 1.0), zorder=4, linewidths=0)
    ax.hlines(group_means[g], min(xs) - SCALE * 0.3, max(xs) + SCALE * 0.3,
              colors=col, linewidth=1.6, zorder=2)

    for xi, (_, row) in zip(xs, sub.iterrows()):
        jname = row.get("journal_name", str(row["node"]))
        abbr  = ABBREV.get(jname, str(jname)[:12])
        ax.annotate(abbr, xy=(xi, row["score_mean_centred"] + 0.02), xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", rotation=90, fontsize=4.5, color="#222222", zorder=4,
                    path_effects=[pe.withStroke(linewidth=0.3, foreground="white")])

# ── Group labels (fixed offset below group mean) ──────────────────────────────
_x_starts  = {}
_x_ends    = {}
x_start = 0.0
for g in ordered_groups:
    members = df[df["group"] == g]
    _x_starts[g] = x_start
    _x_ends[g]   = x_start + (len(members) - 1) * SCALE
    x_start += len(members) * SCALE + GAP

# Labels: evenly ascending from left to right at fixed vertical step
LABEL_Y_BASE = df["score_mean_centred"].min() - df["score_std"].max() - 0.03
LABEL_Y_STEP = 0.125   # data units per group, left→right (fits 7 pt labels)

labeled_groups = [(g, (_x_starts[g] + _x_ends[g]) / 2)
                  for g in ordered_groups if _active_group_labels.get(g, "")]
labeled_groups.sort(key=lambda t: t[1])   # sort by x centre

label_ys: list[float] = []
for i, (g, cx) in enumerate(labeled_groups):
    lbl = _active_group_labels[g]
    cy  = LABEL_Y_BASE + i * LABEL_Y_STEP
    ax.text(cx + LABEL_X_NUDGE.get(lbl, 0.0), cy, lbl, ha="center", va="top", fontsize=7.0,
            color=gc[g], fontweight="bold", clip_on=False)
    label_ys.append(cy - 0.3)

ax.axhline(0, color="#aaaaaa", linewidth=0.7, alpha=0.6, zorder=1)
ax.set_ylabel(r"$s_i$ (journal status)", fontsize=FS)
ax.tick_params(axis="y", labelsize=FS - 1)
ax.set_xticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
# Tighten both ends
data_max = df["score_mean"].max() + df.loc[df["score_mean"].idxmax(), "score_std"]
y_bot = min(label_ys) + 0.5 if label_ys else ax.get_ylim()[0]
y_bot = min(y_bot, -2.0)
ax.set_ylim(y_bot, data_max + 0.1)
xlo, xhi = ax.get_xlim()
ax.set_xlim(xlo, xhi - (xhi - xlo) * 0.03)


if inset:
    # ── Build miniature adjacency matrix for inset ────────────────────────────
    _gml_path = GML
    if _gml_path is not None:
        _G = nx.read_gml(str(_gml_path), label="label")

        _node_order = []
        for _g in ordered_groups:
            _node_order.extend(df[df["group"] == _g].sort_values("score_mean_centred")["node"].tolist())

        _n   = len(_node_order)
        _idx = {nd: i for i, nd in enumerate(_node_order)}
        _A   = np.zeros((_n, _n))
        for _u, _v, _d in _G.edges(data=True):
            _i, _j = _idx.get(_u), _idx.get(_v)
            if _i is not None and _j is not None:
                _A[_i, _j] = _d.get("weight", 1)

        _grps_ord = [df.set_index("node").loc[nd, "group"] for nd in _node_order]
        _bounds = []
        _cur = None
        for _i, _gg in enumerate(_grps_ord):
            if _gg != _cur:
                _bounds.append(_i)
                _cur = _gg
        _bounds.append(_n)

        _eps   = 1e-9
        _asym  = (_A - _A.T) / (_A + _A.T + _eps)
        # Clamp so ratio 0.2/0.8 (asym ≈ ±0.6) saturates to full red/blue
        _ASYM_MAX = 0.6
        _asym_clamped = np.clip(_asym, -_ASYM_MAX, _ASYM_MAX)
        _cmap_rbk = LinearSegmentedColormap.from_list(
            "rbk2", [(0.10, 0.25, 0.85), (0.00, 0.00, 0.00), (0.85, 0.10, 0.10)], N=512)
        _rgba         = _cmap_rbk((_asym_clamped + _ASYM_MAX) / (2 * _ASYM_MAX))
        _amax         = np.percentile(_A[_A > 0], 99) if (_A > 0).any() else 1.0
        _rgba[..., 3] = np.clip(_A / _amax, 0, 1)   # linear opacity
        _rgba[_A == 0, 3] = 0

        # ── Inset layout (figure fractions, figure = 7.0 × 4.5 in) ─────────────
        # Reserve left space for bivariate legend + gap, then adj matrix.
        _FW, _FH = fig.get_size_inches()
        _LEG_IN   = 0.43   # legend square size in inches
        _LEG_GAP  = 0.18   # gap between legend and adj axes
        _ADJ_W_IN = 1.15   # adj axes width in inches
        _ADJ_H_IN = 1.15   # adj axes height in inches (shorter figure)

        _leg_w = _LEG_IN   / _FW;  _leg_h = _LEG_IN   / _FH
        _lgap  = _LEG_GAP  / _FW
        _adj_w = _ADJ_W_IN / _FW;  _adj_h = _ADJ_H_IN / _FH
        _ins_h = _adj_h            # total inset height driven by adj

        # Align top of B panel with top of main axes
        _ax_pos = ax.get_position()
        _ins_b  = _ax_pos.y1 - _ins_h - 18 / (72 * _FH)   # adj position inside box (room for top label/strip)
        _ins_l  = 0.090 + 45 / (72 * 7.0)             # left anchor (+45pt from left margin)
        _leg_l  = _ins_l - 10 / (72 * 7.0)          # legend 10pt left of adj anchor
        _adj_l  = _ins_l + _leg_w + _lgap           # adj axes left edge
        _leg_b  = _ins_b + (_ins_h - _leg_h) / 2   # legend vertically centred

        # ── Adjacency matrix axes ─────────────────────────────────────────────
        ax_ins = fig.add_axes([_adj_l, _ins_b, _adj_w, _adj_h])
        ax_ins.imshow(_rgba, aspect="equal", origin="upper",
                      interpolation="nearest")
        ax_ins.set_xlim(-0.5, _n - 0.5)
        ax_ins.set_ylim(_n - 0.5, -0.5)

        for _b in _bounds:
            ax_ins.axvline(_b - 0.5, color="#888888", lw=0.2, alpha=0.6, zorder=5, clip_on=False)
            ax_ins.axhline(_b - 0.5, color="#888888", lw=0.2, alpha=0.6, zorder=5, clip_on=False)

        ax_ins.set_xticks([]); ax_ins.set_yticks([])
        ax_ins.set_xlabel("Cited journal",  fontsize=7, labelpad=5, color="#444")
        ax_ins.set_ylabel("Citing journal", fontsize=7, labelpad=5, color="#444")
        ax_ins.xaxis.set_label_position("top")
        # xlabel/ylabel used only to measure positions; cleared and replaced below
        for _sp in ax_ins.spines.values():
            _sp.set_visible(False)   # outer box drawn below

        # ── Color strips: pixel-accurate from rendered matrix bounds ──────────
        fig.canvas.draw()
        _rend = fig.canvas.get_renderer()
        _fb   = fig.get_window_extent(_rend)
        _ab   = ax_ins.get_window_extent(_rend)
        _mat_px = min(_ab.width, _ab.height)
        _mat_x0 = _ab.x0 + (_ab.width  - _mat_px) / 2
        _mat_y0 = _ab.y0 + (_ab.height - _mat_px) / 2
        _ml = _mat_x0 / _fb.width;  _mb = _mat_y0 / _fb.height
        _mw = _mat_px / _fb.width;  _mh = _mat_px / _fb.height

        _SPX = 4; _SGX = 0   # flush: no gap between strips and matrix
        _sw = _SPX / _fb.width;  _sh = _SPX / _fb.height
        _gw = _SGX / _fb.width;  _gh = _SGX / _fb.height

        _strip_top  = np.zeros((1, _n, 4))
        _strip_left = np.zeros((_n, 1, 4))
        for _k in range(len(_bounds) - 1):
            _lo, _hi = _bounds[_k], _bounds[_k + 1]
            _gcol = mcolors.to_rgba(gc[_grps_ord[_lo]])
            _strip_top[0, _lo:_hi]  = _gcol
            _strip_left[_lo:_hi, 0] = _gcol

        _ax_top = fig.add_axes([_ml, _mb + _mh + _gh, _mw, _sh])
        _ax_top.imshow(_strip_top, aspect="auto", origin="upper",
                       extent=[-0.5, _n - 0.5, 0, 1])
        _ax_top.set_axis_off()

        _ax_left_strip = fig.add_axes([_ml - _gw - _sw, _mb, _sw, _mh])
        _ax_left_strip.imshow(_strip_left, aspect="auto", origin="upper",
                              extent=[0, 1, _n - 0.5, -0.5])
        _ax_left_strip.set_axis_off()

        # ── Bivariate legend to the left of the adj ───────────────────────────
        # Same design as the full adj figure: X=citation ratio (red→black→blue),
        # Y=# citations (transparent→opaque).
        # Legend x runs over clamped ratio range [0.2, 0.8]; at edges cmap is saturated
        # Legend x = citation ratio 0→1; color saturates at 0.2 and 0.8
        _BG_rgb = np.array([1.0, 1.0, 1.0])
        _NX, _NY = 120, 60
        _leg_img = np.ones((_NY, _NX, 3))
        for _iy, _v in enumerate(np.linspace(0, 1, _NY)):
            for _ix, _r in enumerate(np.linspace(0, 1, _NX)):
                # citation ratio r → asym = 2r-1, clamped at ±_ASYM_MAX
                _av = np.clip(2 * _r - 1, -_ASYM_MAX, _ASYM_MAX)
                _cp = (_av + _ASYM_MAX) / (2 * _ASYM_MAX)
                _c  = np.array(_cmap_rbk(_cp)[:3])
                _leg_img[_iy, _ix] = _c * _v + _BG_rgb * (1 - _v)

        ax_leg = fig.add_axes([_leg_l, _leg_b, _leg_w, _leg_h])
        ax_leg.imshow(_leg_img, aspect="auto", origin="lower",
                      extent=[-1, 1, 0, 1], interpolation="bilinear")
        ax_leg.set_xticks([-1, 0, 1])
        ax_leg.set_xticklabels(["0", "0.5", "1"], fontsize=4.5)
        _amax_int = int(round(_amax))
        _amid_int = int(round(_amax * 0.5))   # linear: 50% opacity = 50% of max weight
        ax_leg.set_yticks([0, 0.5, 1])
        ax_leg.set_yticklabels(["0", str(_amid_int), str(_amax_int)], fontsize=4.5)
        ax_leg.set_ylabel("# citations", fontsize=4.5, labelpad=1)
        ax_leg.set_xlabel("Citation ratio", fontsize=4.5, labelpad=2)
        ax_leg.xaxis.tick_top(); ax_leg.xaxis.set_label_position("top")
        ax_leg.tick_params(length=1.5, pad=1, width=0.4)
        for _sp in ax_leg.spines.values():
            _sp.set_visible(True); _sp.set_linewidth(0.6); _sp.set_edgecolor("#333333")

        # ── Outer box: use tight bbox of legend to get true left edge ────────
        fig.canvas.draw()                        # re-draw so legend decorations are placed
        _rend2   = fig.canvas.get_renderer()
        _fb2     = fig.get_window_extent(_rend2)
        _leg_tb  = ax_leg.get_tightbbox(_rend2)  # includes ylabel, ticks, ticklabels
        _PAD_PX  = 6
        _PAD_X_PX = _PAD_PX + 10   # 10pt extra whitespace left and right
        _pad_x   = _PAD_X_PX / _fb2.width
        _pad_y   = _PAD_PX / _fb2.height
        _box_x0  = (_leg_tb.x0 / _fb2.width)  - _pad_x
        _box_y0  = _mb                         - _pad_y
        _box_x1  = _ml + _mw                  + _pad_x - 5 / (72 * _FW)
        _box_y1  = _ax_pos.y1 + _pad_y - 5 / (72 * _FH)  # pinned, -5pt top
        fig.patches.append(mpatches.FancyBboxPatch(
            (_box_x0, _box_y0), _box_x1 - _box_x0, _box_y1 - _box_y0,
            boxstyle="square,pad=0",
            linewidth=0.6, edgecolor="#333333", facecolor="none",
            transform=fig.transFigure, zorder=10, clip_on=False,
        ))

        # ── A / B panel labels ────────────────────────────────────────────────
        # A / B: same vertical position — top of B box
        _yl_bbox  = ax.yaxis.label.get_window_extent(_rend2)
        _a_x_fig  = (_yl_bbox.x0 + _yl_bbox.x1) * 0.5 / _fb2.width
        _b_y_fig  = _box_y1 - _pad_y * 0.4 - 2 / (72 * _FH)  # B top-left corner y, -2pt
        fig.text(_a_x_fig, _b_y_fig, "A",
                 fontsize=11, fontweight="bold", va="top", ha="center",
                 transform=fig.transFigure, clip_on=False)

        # B: top-left corner of the outer box (just inside)
        fig.text(_box_x0 + _pad_x * 0.6 - 3 / (72 * _FW), _b_y_fig, "B",
                 fontsize=11, fontweight="bold", va="top", ha="left",
                 transform=fig.transFigure, clip_on=False)

        # ── (low) / (high) end labels + repositioned axis labels ────────────
        _ins_ax_bb = ax_ins.get_window_extent(_rend2)
        _xl_ins_bb = ax_ins.xaxis.label.get_window_extent(_rend2)
        _yl_ins_bb = ax_ins.yaxis.label.get_window_extent(_rend2)
        _rk_ins_fs = 4.5

        # Rank label anchor positions (centre of original xlabel/ylabel)
        _xl_ins_cy = (_xl_ins_bb.y0 + _xl_ins_bb.y1) * 0.5 / _fb2.height
        _yl_ins_cx = (_yl_ins_bb.x0 + _yl_ins_bb.x1) * 0.5 / _fb2.width

        # Clear matplotlib-placed labels; replace with fig.text so we control position
        ax_ins.set_xlabel(""); ax_ins.set_ylabel("")

        _pt   = 1 / (72 * _FH)   # 1pt in figure-height fracs
        _ptx  = 1 / (72 * _FW)

        # "Cited journal": bottom at rank-label y minus 2pt (label sits just above rank labels)
        _adj_cx = (_ins_ax_bb.x0 + _ins_ax_bb.x1) * 0.5 / _fb2.width
        fig.text(_adj_cx, _xl_ins_cy - 2 * _pt, "Cited journal",
                 ha="center", va="bottom", fontsize=7, color="#444",
                 transform=fig.transFigure, clip_on=False)

        # "Citing journal": va="center" matches original ylabel centering
        _adj_cy = (_ins_ax_bb.y0 + _ins_ax_bb.y1) * 0.5 / _fb2.height
        fig.text(_yl_ins_cx, _adj_cy, "Citing journal",
                 ha="center", va="center", fontsize=7, color="#444",
                 rotation=90, transform=fig.transFigure, clip_on=False)

        # Rank labels: (low)/(high) on Cited journal axis raised 2pt
        _rk_cy = _xl_ins_cy + 1 * _pt
        fig.text(_ins_ax_bb.x0 / _fb2.width, _rk_cy, "(low)",
                 ha="left", va="center", fontsize=_rk_ins_fs, color="#666",
                 transform=fig.transFigure, clip_on=False)
        fig.text(_ins_ax_bb.x1 / _fb2.width, _rk_cy, "(high)",
                 ha="right", va="center", fontsize=_rk_ins_fs, color="#666",
                 transform=fig.transFigure, clip_on=False)
        fig.text(_yl_ins_cx, _ins_ax_bb.y1 / _fb2.height, "(low)",
                 ha="center", va="top", fontsize=_rk_ins_fs, color="#666",
                 rotation=90, transform=fig.transFigure, clip_on=False)
        fig.text(_yl_ins_cx, _ins_ax_bb.y0 / _fb2.height, "(high)",
                 ha="center", va="bottom", fontsize=_rk_ins_fs, color="#666",
                 rotation=90, transform=fig.transFigure, clip_on=False)

        _stem = "journal_scores"
    else:
        print("  GML not found — skipping inset")
        _stem = "journal_scores_no_inset"
else:
    _stem = "journal_scores_no_inset"


# y-axis line runs the full figure height, from the lowest group label to the top of the highest journal
# label (as in the football figure); ticks stay where they are
fig.canvas.draw()
_rr = fig.canvas.get_renderer()
_fhpx = fig.get_window_extent(_rr).height
_tbs = [t.get_window_extent(_rr) for t in ax.texts]
_ya, _yb = min(b.y0 for b in _tbs) / _fhpx, max(b.y1 for b in _tbs) / _fhpx
_lsp = ax.spines["left"]
fig.add_artist(matplotlib.lines.Line2D([ax.get_position().x0] * 2, [_ya, _yb], transform=fig.transFigure,
                                       color=_lsp.get_edgecolor(), linewidth=_lsp.get_linewidth(),
                                       solid_capstyle="butt"))
_lsp.set_visible(False)

for ext in ("png", "pdf"):
    out = FIGURES / f"{_stem}.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Saved {out}")
plt.close(fig)

# ── Adjacency matrix (optional) ───────────────────────────────────────────────

if adj:
    G        = nx.read_gml(str(GML), label="label")

    node_order = []
    for g in ordered_groups:
        node_order.extend(df[df["group"] == g].sort_values("score_mean_centred")["node"].tolist())

    n   = len(node_order)
    idx = {nd: i for i, nd in enumerate(node_order)}
    A   = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        i, j = idx.get(u), idx.get(v)
        if i is not None and j is not None:
            A[i, j] = d.get("weight", 1)

    groups_ord = [df.set_index("node").loc[nd, "group"] for nd in node_order]
    boundaries = []
    cur = None
    for i, g in enumerate(groups_ord):
        if g != cur:
            boundaries.append(i)
            cur = g
    boundaries.append(n)

    eps      = 1e-9
    asym     = (A - A.T) / (A + A.T + eps)
    ASYM_MAX = 0.6   # ratio 0.2/0.8 saturates to full blue/red
    asym_clamped = np.clip(asym, -ASYM_MAX, ASYM_MAX)
    cmap_rbk = LinearSegmentedColormap.from_list(
        "rbk", [(0.10, 0.25, 0.85), (0.00, 0.00, 0.00), (0.85, 0.10, 0.10)], N=512)
    rgba          = cmap_rbk((asym_clamped + ASYM_MAX) / (2 * ASYM_MAX))
    amax          = np.percentile(A[A > 0], 99) if (A > 0).any() else 1.0
    rgba[..., 3]  = np.clip(A / amax, 0, 1) ** 0.7
    rgba[A == 0, 3] = 0

    abbrevs = [ABBREV.get(df.set_index("node").loc[nd, "journal_name"],
                          str(df.set_index("node").loc[nd, "journal_name"])[:12])
               for nd in node_order]
    BG     = "#f6f6f6"
    BG_rgb = np.array(mcolors.to_rgb(BG))

    fig2, ax2 = plt.subplots(figsize=(7.0, 9.5))
    fig2.subplots_adjust(left=0.14, right=0.97, top=0.78, bottom=0.22)
    ax2.set_facecolor(BG)
    ax2.imshow(rgba, aspect="equal", origin="upper")

    for b in boundaries:
        ax2.axvline(b - 0.5, color="black", linewidth=0.6, zorder=5)
        ax2.axhline(b - 0.5, color="black", linewidth=0.6, zorder=5)

    TICK_FS = 4.0
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.xaxis.tick_top(); ax2.xaxis.set_label_position("top")
    ax2.set_xticklabels(abbrevs, rotation=90, fontsize=TICK_FS, va="bottom")
    ax2.set_yticklabels(abbrevs, fontsize=TICK_FS)
    ax2.tick_params(axis="x", which="both", pad=12, length=0,
                    top=True, bottom=False, labeltop=True, labelbottom=False)
    ax2.tick_params(axis="y", which="both", pad=10, length=0)
    ax2.set_xlabel("Cited journal",  fontsize=7.5, labelpad=14, color="#444")
    ax2.set_ylabel("Citing journal", fontsize=7.5, labelpad=10, color="#444")

    fig2.canvas.draw()
    _rend_a = fig2.canvas.get_renderer()
    _fb_a   = fig2.get_window_extent(_rend_a)
    _ax2_bb = ax2.get_window_extent(_rend_a)
    _xl_bb  = ax2.xaxis.label.get_window_extent(_rend_a)
    _yl_bb  = ax2.yaxis.label.get_window_extent(_rend_a)
    # (low rank) / (high rank) at ends of Cited journal (top x-axis)
    _xl_cy  = (_xl_bb.y0 + _xl_bb.y1) * 0.5 / _fb_a.height
    fig2.text(_ax2_bb.x0 / _fb_a.width,  _xl_cy, "(low rank)",
              ha="left",  va="center", fontsize=TICK_FS, color="#444",
              transform=fig2.transFigure, clip_on=False)
    fig2.text(_ax2_bb.x1 / _fb_a.width,  _xl_cy, "(high rank)",
              ha="right", va="center", fontsize=TICK_FS, color="#444",
              transform=fig2.transFigure, clip_on=False)

    p       = ax2.get_position()
    _fw, _fh = fig2.get_size_inches()
    STRIP_W = 0.010; STRIP_H = STRIP_W * _fw / _fh; STRIP_GAP = 0.003
    # (low rank) / (high rank) at ends of Citing journal (left y-axis)
    # place (low rank) above the top strip so it doesn't overlap
    _yl_cx  = (_yl_bb.x0 + _yl_bb.x1) * 0.5 / _fb_a.width
    fig2.text(_yl_cx, _ax2_bb.y1 / _fb_a.height + STRIP_H + STRIP_GAP,
              "(low rank)",
              ha="center", va="top", fontsize=TICK_FS, color="#444",
              rotation=90, transform=fig2.transFigure, clip_on=False)
    fig2.text(_yl_cx, _ax2_bb.y0 / _fb_a.height, "(high rank)",
              ha="center", va="bottom", fontsize=TICK_FS, color="#444",
              rotation=90, transform=fig2.transFigure, clip_on=False)
    ax_top  = fig2.add_axes([p.x0, p.y1 + STRIP_GAP, p.width, STRIP_H])
    ax_left = fig2.add_axes([p.x0 - STRIP_W - STRIP_GAP, p.y0, STRIP_W, p.height])
    for ax_s, vertical in [(ax_top, False), (ax_left, True)]:
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis("off")
        for k in range(len(boundaries) - 1):
            lo, hi = boundaries[k], boundaries[k + 1]
            col    = gc[groups_ord[lo]]
            if not vertical:
                ax_s.add_patch(mpatches.Rectangle((lo / n, 0), (hi - lo) / n, 1,
                               color=col, transform=ax_s.transAxes))
            else:
                ax_s.add_patch(mpatches.Rectangle((0, (n - hi) / n), 1, (hi - lo) / n,
                               color=col, transform=ax_s.transAxes))

    # ── Bivariate legend (taller, with numeric scale) ─────────────────────────
    # Citing ratio computed for the scatter plot (not shown in adj figure)
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    total    = row_sums + col_sums
    citing_ratio = np.where(total > 0, row_sums / total, 0.5)

    NX, NY    = 200, 100
    legend_img = np.ones((NY, NX, 3))
    for iy, v in enumerate(np.linspace(0, 1, NY)):
        for ix, r in enumerate(np.linspace(0, 1, NX)):
            # citation ratio r → asym = 2r-1, clamped at ±ASYM_MAX
            av = np.clip(2 * r - 1, -ASYM_MAX, ASYM_MAX)
            cp = (av + ASYM_MAX) / (2 * ASYM_MAX)
            c  = np.array(cmap_rbk(cp)[:3])
            legend_img[iy, ix] = c * v + BG_rgb * (1 - v)  # linear opacity gradient
    sw_w = 0.07; sw_h = sw_w * _fw / _fh  # square on this figure
    sw_x = 0.094
    sw_y = 1.0 - sw_h - 0.152
    ax_sw = fig2.add_axes([sw_x, sw_y, sw_w, sw_h])
    ax_sw.imshow(legend_img, aspect="auto", origin="lower",
                 extent=[-1, 1, 0, 1], interpolation="bilinear")
    ax_sw.set_xticks([-1, 0, 1])
    ax_sw.set_xticklabels(["0", "0.5", "1"], fontsize=4.0)
    _w_mid = int(round(amax * 0.5 ** (1 / 0.7)))  # weight giving 50% opacity under x^0.7
    _w_max = int(round(amax))
    ax_sw.set_yticks([0, 0.5, 1])
    ax_sw.set_yticklabels(["0", str(_w_mid), str(_w_max)], fontsize=4.0)
    ax_sw.set_ylabel("# citations", fontsize=4.0, labelpad=1)
    ax_sw.set_xlabel("Citation ratio", fontsize=4.0, labelpad=3)
    ax_sw.xaxis.tick_top()
    ax_sw.xaxis.set_label_position("top")
    ax_sw.tick_params(length=2, pad=2)
    for sp in ax_sw.spines.values(): sp.set_linewidth(0.5)

    _adj_stem = "journal_adj"
    for ext in ("png", "pdf"):
        out2 = FIGURES / f"{_adj_stem}.{ext}"
        fig2.savefig(out2, dpi=300, bbox_inches="tight")
        print(f"Saved {out2}")
    plt.close(fig2)

