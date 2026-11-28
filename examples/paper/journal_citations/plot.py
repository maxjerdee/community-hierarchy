"""Journal citations: the paper figure, from the output of fit.py.

Reads:    results/scores_coupled.csv
          (the *_quick.csv file with --quick, writing figures/*_quick)
Outputs:  figures/journal_scores.{png,pdf}  journal scores by group
Run with:
    python plot.py [--quick]

Each group has its own colour; every journal is named.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import community_hierarchy as ch

HERE = Path(__file__).parent
SUFFIX = "_quick" if "--quick" in sys.argv else ""  # results of fit.py's QUICK settings
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

ABBREV = {
    # ── Flagship / top generalist ──────────────────────────────────────────────
    "Nature": "Nature",
    "Science": "Science",
    "Cell": "Cell",
    "The Lancet": "Lancet",
    "New England Journal of Medicine": "NEJM",
    "JAMA": "JAMA",
    "JAMA: The Journal of the American Medical Association": "JAMA",
    "Nature Medicine": "Nat.Med.",
    "Nature Communications": "Nat.Comm.",
    "Proceedings of the National Academy of Sciences": "PNAS",
    # ── Broad open-access / multidisciplinary ─────────────────────────────────
    "Scientific Reports": "Sci.Rep.",
    "PLoS ONE": "PLoS ONE",
    "PLOS ONE": "PLoS ONE",
    "Science Advances": "Sci.Adv.",
    "Advanced Science": "Adv.Sci.",
    "eLife": "eLife",
    "Heliyon": "Heliyon",
    "BMJ": "BMJ",
    "BMJ Open": "BMJ Open",
    "The BMJ": "BMJ",
    "JAMA Network Open": "JAMA Netw.Open",
    # ── Medicine / clinical ───────────────────────────────────────────────────
    "Journal of Clinical Oncology": "JCO",
    "Journal of Clinical Medicine": "J.Clin.Med.",
    "Cancers": "Cancers",
    "Circulation": "Circulation",
    "Frontiers in Immunology": "Front.Immunol.",
    "Frontiers in Pharmacology": "Front.Pharmacol.",
    "Frontiers in Oncology": "Front.Oncol.",
    "Frontiers in Psychology": "Front.Psychol.",
    "Nutrients": "Nutrients",
    "Cells": "Cells",
    # ── Biology / biochemistry ────────────────────────────────────────────────
    "Nucleic Acids Research": "NAR",
    "Cell Reports": "Cell Rep.",
    "Frontiers in Microbiology": "Front.Micro.",
    "Frontiers in Plant Science": "Front.Plant.Sci.",
    "Molecules": "Molecules",
    "International Journal of Molecular Sciences": "IJMS",
    "International Journal of Biological Macromolecules": "IJBM",
    "Carbohydrate Polymers": "Carbohyd.Polym.",
    # ── Chemistry ─────────────────────────────────────────────────────────────
    "Journal of the American Chemical Society": "JACS",
    "Angewandte Chemie International Edition": "Angew.Chem.",
    "Chemical Reviews": "Chem.Rev.",
    "Chemical Society Reviews": "Chem.Soc.Rev.",
    "Analytical Chemistry": "Anal.Chem.",
    "ACS Catalysis": "ACS Catal.",
    "ACS Nano": "ACS Nano",
    "ACS Applied Materials & Interfaces": "ACS AMI",
    "ACS Omega": "ACS Omega",
    "ACS Sustainable Chemistry & Engineering": "ACS SC&E",
    "RSC Advances": "RSC Adv.",
    "Chemical Engineering Journal": "Chem.Eng.J.",
    "Applied Catalysis B: Environmental": "Appl.Catal.B",
    # ── Materials science ─────────────────────────────────────────────────────
    "Advanced Materials": "Adv.Mat.",
    "Advanced Functional Materials": "Adv.Funct.Mat.",
    "Small": "Small",
    "Journal of Materials Chemistry A": "JMCA",
    "Journal of Alloys and Compounds": "J.Alloys Comp.",
    "Journal of Colloid and Interface Science": "JCIS",
    "Applied Surface Science": "Appl.Surf.Sci.",
    "Ceramics International": "Ceram.Int.",
    "Separation and Purification Technology": "Sep.Purif.Tech.",
    "Nanomaterials": "Nanomaterials",
    "Nano Energy": "Nano Energy",
    "Materials": "Materials",
    "Materials Today Proceedings": "Mat.Today Proc.",
    "Polymers": "Polymers",
    # ── Energy ────────────────────────────────────────────────────────────────
    "Advanced Energy Materials": "Adv.Energy Mat.",
    "Journal of Energy Storage": "J.Energy Stor.",
    "Renewable Energy": "Renew.Energy",
    "Renewable and Sustainable Energy Reviews": "RSER",
    "International Journal of Hydrogen Energy": "IJHE",
    "Applied Energy": "Appl.Energy",
    "Energies": "Energies",
    "Fuel": "Fuel",
    # ── Environment / sustainability ──────────────────────────────────────────
    "Environmental Science & Technology": "ES&T",
    "The Science of The Total Environment": "Sci.Tot.Env.",
    "Chemosphere": "Chemosphere",
    "Environmental Pollution": "Environ.Pollut.",
    "Environmental Research": "Environ.Res.",
    "Environmental Science and Pollution Research": "ESPR",
    "Journal of Hazardous Materials": "J.Haz.",
    "Journal of Environmental Management": "J.Environ.Mgmt.",
    "Journal of Cleaner Production": "J.Clean.Prod.",
    "Sustainability": "Sustainability",
    "Bioresource Technology": "Bioresour.Tech.",
    "Water Research": "Water Res.",
    "Water": "Water",
    "Construction and Building Materials": "Constr.Build.Mat.",
    "International Journal of Environmental Research and Public Health": "IJERPH",
    # ── Food / agriculture ────────────────────────────────────────────────────
    "Food Chemistry": "Food Chem.",
    "Foods": "Foods",
    # ── Physics / astronomy ───────────────────────────────────────────────────
    "Physical Review Letters": "PRL",
    "Physical review. B./Physical review. B": "PRB",
    "Physical review. D/Physical review. D.": "PRD",
    "Journal of Physics Conference Series": "J.Phys.Conf.",
    "The Astrophysical Journal": "ApJ",
    "Monthly Notices of the Royal Astronomical Society": "MNRAS",
    # ── CS / engineering / sensing ────────────────────────────────────────────
    "IEEE Access": "IEEE Access",
    "IEEE Internet of Things Journal": "IEEE IoT J.",
    "Expert Systems with Applications": "Expert Syst.",
    "Remote Sensing": "Remote Sens.",
    "Sensors": "Sensors",
    "Applied Sciences": "Appl.Sci.",
}


scores_csv = RESULTS / f"scores_coupled{SUFFIX}.csv"
if not scores_csv.exists():
    sys.exit(f"{scores_csv} not found: run `python fit.py` first.")
df = pd.read_csv(scores_csv)
fig, ax = plt.subplots(figsize=(6.75, 4.1))
ch.plot_node_properties(
    df,
    ax=ax,
    labels="none",
    group_gap=1.5,
    annotate={n: ABBREV.get(nm, nm[:12]) for n, nm in zip(df.node, df.name)},
    annotation_fontsize=4.5,
    ylabel=r"$s_i$ (journal status)",
)
ax.yaxis.label.set_size(8)
ax.tick_params(axis="y", labelsize=7)
for ext in ("png", "pdf"):
    out = FIGURES / f"journal_scores{SUFFIX}.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out}")
plt.close(fig)
