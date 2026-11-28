# Paper figures

Code to reproduce the main figures of the paper for the two public datasets:
international football and journal citations. The high-school friendship
networks (Add Health) cannot be redistributed, so they are not included.

Each folder has the same layout:

| File | What it does |
|---|---|
| `data/` | the network, as GML |
| `fit.py` | minimal: loads the network, fits the model, writes `results/` |
| `results/` | `*.csv`: the output of `fit.py` with its `REPRODUCE` settings (committed, so the plots run without refitting); `*_quick.csv`: with its default `QUICK` settings (not committed) |
| `figures/` | the output of `plot.py` (`*_quick` with `--quick`) |
| `plot.py` | the full figure code: reads `results/` (the `_quick` files with `--quick`), writes `figures/` |

```bash
pip install community-hierarchy arviz   # arviz for the football cross-validation
cd international_football
python fit.py                    # quick look, under a minute (see below): results/*_quick.csv
python plot.py --diffs           # the committed full run: figures/football_scores.png, football_diffs.png
python plot.py --diffs --quick   # your quick run: figures/*_quick.png
python predict.py                # predictions: 2026 knockouts vs the market, and cross-validation
cd ../journal_citations
python fit.py            # quick look, about a minute: results/*_quick.csv
python plot.py           # figures/journal_scores.png (add --quick for your run)
```

Each `fit.py` has two settings. The active one, `QUICK`, runs in about a
minute: enough to explore the results, but too short for the group structure to
settle. Its fixed seed makes a run repeatable, but its groups and numbers differ
from the paper's. The
commented `REPRODUCE` block is the run behind the committed `results/*.csv`
and the paper's figures; uncomment it (and comment out `QUICK`) to regenerate
them:

- football: 4 chains, 1,000 burn-in samples and 5,000 recorded samples, 10
  sweeps per sample, seed 1 (about 5 minutes per model);
- journals: the same, with each chain run at 8 temperatures (parallel
  tempering), seed 1 (about 35 minutes). The weighted network mixes slowly
  between competing partitions without tempering.

Times are for a 16-core desktop, with the four chains running in parallel.
Label positions in `plot.py` are tuned to the committed fit; a different seed
can need small adjustments.

`fit.py` saves two tables per model (football fits the coupled and independent
models, the journals only the coupled one):

- `scores_<model>.csv` from `Result.node_properties_df()`: each node's posterior
  mean score and standard deviation, its group in the consensus partition
  (maximum-overlap consensus of Peixoto 2021, with the number of groups free),
  and the group's mean score.
- `parameters_<model>.csv` from `Result.network_properties_df()`: posterior
  means and 90% highest-density intervals of the network-level parameters
  (σ_G, σ_I, ICC, the ties parameter, densities, number of groups).

`<model>` is `coupled` (group mean scores and individual scores fitted jointly,
the model of the paper) or `independent` (the community and hierarchy models
fitted separately, for comparison).

## International football

`data/football_group_stage.gml`: 216 national teams. One edge per match in
competitive international fixtures (World Cup, continental championships, Nations
Leagues and their qualifiers; no friendlies) from 1 September 2023 to the end of
the 2026 World Cup group stage. A decisive match is a `dominant` edge from winner
to loser; a draw is a `neutral` edge. Node attribute `in_wc2026` marks the 48
teams in the 2026 World Cup. Source: M. Jürisoo, *International football results
from 1872*, https://github.com/martj42/international_results (CC0).

`predict.py` compares the coupled and independent models' predictions (`fit.py`
computes them while the samples are in memory, and writes
`results/knockouts_<model>.csv` and `results/loo_<model>.csv`):

- the 32 knockout matches of the 2026 World Cup, which are out of sample, against
  the betting market (its de-vigged pre-match probabilities are listed in
  `predict.py`): how many each called correctly and the log-likelihood of the
  results;
- leave-one-match-out cross-validation of the 2,134 group-stage results
  (PSIS-LOO, Vehtari, Gelman & Gabry 2017; needs `arviz`): each model's expected
  log predictive density, and accuracy and log-likelihood on the decisive
  matches.

`plot.py` draws the scores with `community_hierarchy.plot_node_properties`: one
colour per inferred group, filled markers for the World Cup teams, and the
strongest team of each group named.

## Journal citations

`data/journal_citations.gml`: 80 journals across fields. An edge u → v with
weight w means that articles u published in 2023 cite articles v published in
2019–2023 w times (1.7 million citations in all; pairs exchanging a single
citation are dropped). The journals are selected from the 90 most-cited of
2019–2023 by repeatedly removing the one exchanging the fewest citations with
the rest, until 80 remain; the first removed are the clinical journals (NEJM,
The Lancet, JAMA, BMJ and others), heavily cited from outside the set but barely
connected to it. Node attributes: `name`, `cited_by`. The model is fitted with
`weighted_SBM=True`. Source: OpenAlex (CC0), https://openalex.org.

Each edge is a citation from the citing journal to the cited one, and `fit.py`
passes `source_high_rank=False` so that being cited counts as the win: higher
scores are higher status, with no sign flip. `plot.py` draws the scores with `community_hierarchy.plot_node_properties`, one
colour per inferred group.
