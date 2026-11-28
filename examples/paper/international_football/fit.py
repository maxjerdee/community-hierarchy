"""Fit the model to the international football network.

Writes to results/ (file names end in _quick for the QUICK settings), for the
coupled model and, for
comparison, the independent model: scores_<model>.csv (one row per nation),
parameters_<model>.csv (posterior means and 90% HDIs), knockouts_<model>.csv
(predictions for the 32 knockout matches of the 2026 World Cup) and loo_<model>.csv
(leave-one-match-out cross-validation of the group-stage results); see predict.py.

Two settings below.  QUICK (the default) takes under a minute for both models:
enough to explore the results, but too short for the partition to settle, so the
groups (and to a lesser extent the numbers) vary from run to run.  To
regenerate the committed results/ and the paper figures, comment out QUICK
and uncomment REPRODUCE (fixed seed; about 5 minutes per model on a 16-core
desktop, the four chains running in parallel).
"""
from pathlib import Path

import networkx as nx

import community_hierarchy as ch

from predict import knockout_probabilities, loo

HERE = Path(__file__).parent

# Quick check (under a minute for both models).
SUFFIX = "_quick"                  # writes results/*_quick.csv
SETTINGS = dict(num_chains=4, num_samples=300, sweeps_per_sample=10,
                num_warmup=100, seed=1, timeout=120)

# Figure reproduction: the run behind results/, the paper figures and the MCMC table
# (4 chains x 1,000 burn-in + 5,000 recorded samples, 10 sweeps per sample).
# SUFFIX = ""                     # writes results/*.csv (committed)
# SETTINGS = dict(num_chains=4, num_samples=5000, sweeps_per_sample=10,
#                 num_warmup=1000,         # x sweeps_per_sample = 10,000 sweeps of burn-in
#                 seed=1, timeout=36000)

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

G = nx.read_gml(HERE / "data" / "football_group_stage.gml", label="label")

for interaction in ("coupled", "independent"):
    result = ch.fit(
        ch.Config(groups_model="traditional_SBM", interaction=interaction),
        G,
        **SETTINGS,
    )
    result.node_properties_df().to_csv(OUT / f"scores_{interaction}{SUFFIX}.csv", index=False)
    result.network_properties_df().to_csv(OUT / f"parameters_{interaction}{SUFFIX}.csv", index=False)
    knockout_probabilities(result).to_csv(OUT / f"knockouts_{interaction}{SUFFIX}.csv", index=False)
    loo(result, G).to_csv(OUT / f"loo_{interaction}{SUFFIX}.csv", index=False)
    print(result.network_properties_df().round(3).to_string(index=False))
