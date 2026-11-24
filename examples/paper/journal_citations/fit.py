"""Fit the model to the citation network of 85 journals.

Edges are weighted citation counts between journals (2019-2023 citing window,
OpenAlex), so the group model is the weighted SBM.  Writes
results/scores_<model>.csv (one row per journal) and results/parameters_<model>.csv
(posterior means and 90% HDIs) for the coupled and independent models.
"""
from pathlib import Path

import networkx as nx

import community_hierarchy as ch

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

G = nx.read_gml(HERE / "data" / "journal_citations.gml", label="label")

for interaction in ("coupled", "independent"):
    result = ch.fit(
        ch.Config(groups_model="traditional_SBM", weighted_SBM=True, interaction=interaction),
        G,
        num_chains=4,
        samples_per_chain=5000,
        sweeps_per_sample=10,
        burn_sweeps=500,          # x sweeps_per_sample = 5000 sweeps of burn-in
        seed=900,
        timeout=3600,
    )
    scores = result.node_properties_df()
    scores["name"] = scores["node"].map(nx.get_node_attributes(G, "name"))
    scores.to_csv(OUT / f"scores_{interaction}.csv", index=False)
    result.network_properties_df().to_csv(OUT / f"parameters_{interaction}.csv", index=False)
    print(result.network_properties_df().round(3).to_string(index=False))
