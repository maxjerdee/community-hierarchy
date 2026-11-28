"""Fit the model to the citation network of 80 journals.

Edges point from the citing journal to the cited one, weighted by the citation
count (2019-2023, OpenAlex), so the group model is the weighted SBM and being
cited is the win (source_high_rank=False).  Writes
results/scores_coupled.csv (one row per journal) and results/parameters_coupled.csv
(posterior means and 90% HDIs) for the coupled model; with the QUICK settings the
file names end in _quick.

The weighted network mixes slowly: almost every pair of journals is linked, the
citation counts pin the scores tightly, and the chains can settle in competing
partitions.  Both settings therefore use parallel tempering (num_tempering_chains:
copies of each chain at a ladder of temperatures that swap states, so the
beta = 1 chain can cross between partitions).

Two settings below.  QUICK (the default) takes about a minute: enough to see the
field structure and rough parameters, but far too short to converge, so the
groups of the journals between fields and the numbers vary from run to run.
To regenerate the committed results/ and the paper figure, comment out QUICK and
uncomment REPRODUCE (fixed seed; about 35 minutes on a 16-core desktop, the
four chains running in parallel).
"""
from pathlib import Path

import networkx as nx

import community_hierarchy as ch

HERE = Path(__file__).parent

# Quick check (about a minute): 4 temperatures.
SUFFIX = "_quick"                  # writes results/*_quick.csv
SETTINGS = dict(num_chains=4, samples_per_chain=200, sweeps_per_sample=10,
                burn_sweeps=50, num_tempering_chains=4, seed=1, timeout=180)

# Figure reproduction: the run behind results/ and the paper figure.
# SUFFIX = ""                     # writes results/*.csv (committed)
# SETTINGS = dict(num_chains=4, samples_per_chain=5000, sweeps_per_sample=10,
#                 burn_sweeps=1000,         # x sweeps_per_sample = 10,000 sweeps of burn-in
#                 num_tempering_chains=8,   # 8 temperatures per chain
#                 seed=1, timeout=36000)

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

G = nx.read_gml(HERE / "data" / "journal_citations.gml", label="label")

result = ch.fit(
    ch.Config(groups_model="traditional_SBM", weighted_SBM=True),
    G,
    source_high_rank=False,     # an edge u -> v is a citation of v by u: v wins
    **SETTINGS,
)
scores = result.node_properties_df()
scores["name"] = scores["node"].map(nx.get_node_attributes(G, "name"))
scores.to_csv(OUT / f"scores_coupled{SUFFIX}.csv", index=False)
result.network_properties_df().to_csv(OUT / f"parameters_coupled{SUFFIX}.csv", index=False)
print(result.network_properties_df().round(3).to_string(index=False))
