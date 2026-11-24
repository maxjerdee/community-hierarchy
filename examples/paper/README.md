# Paper figures

Code to reproduce the main figures of the paper for the two public datasets:
international football and journal citations. The high-school friendship
networks (Add Health) cannot be redistributed, so they are not included.

Each folder has the same layout:

| File | What it does |
|---|---|
| `data/` | the network, as GML |
| `fit.py` | minimal: loads the network, fits the model, writes `results/` |
| `results/` | the output of `fit.py` (committed, so the plots run without refitting) |
| `figures/` | the output of `plot.py` |
| `plot.py` | the full figure code: reads `results/`, writes `figures/` |

```bash
pip install community-hierarchy geopandas   # geopandas only for the football map
cd international_football
python fit.py            # about 20 minutes: coupled and independent models
python plot.py --diffs   # figures/football_scores.png, figures/football_diffs.png
cd ../journal_citations
python fit.py            # about 20 minutes
python plot.py           # figures/journal_scores.png
```

Every fit uses 4 chains of 5,000 samples, 10 sweeps per sample, and 5,000
sweeps of burn-in, with a fixed seed. Times are for a 16-core desktop, with
the four chains running in parallel. Label positions in `plot.py` are tuned to
the committed fit; a different seed can need small adjustments.

`fit.py` saves two tables per model:

- `scores_<model>.csv` from `Result.node_properties_df()`: each node's posterior
  mean score and standard deviation, its group in the consensus partition
  (maximum-overlap consensus at the posterior modal number of groups), and the
  group's mean score.
- `parameters_<model>.csv` from `Result.network_properties_df()`: posterior
  means and 90% highest-density intervals of the network-level parameters
  (σ_G, σ_I, ICC, the ties parameter, densities, number of groups).

`<model>` is `coupled` (group mean scores and individual scores fitted jointly,
the model of the paper) or `independent` (the community and hierarchy models
fitted separately, for comparison).

## International football

`data/football_group_stage.gml`: 215 national teams. One edge per match in
competitive international fixtures (World Cup, continental championships, Nations
Leagues and their qualifiers; no friendlies) from 1 September 2023 to the end of
the 2026 World Cup group stage. A decisive match is a `dominant` edge from winner
to loser; a draw is a `neutral` edge. Node attribute `in_wc2026` marks the 48
teams in the 2026 World Cup. Source: M. Jürisoo, *International football results
from 1872*, https://github.com/martj42/international_results (CC0).

`plot.py` colours and names each inferred group by the confederations of its
members, so it works on any partition the fit returns. The world map in panel B
uses Natural Earth boundaries (public domain), downloaded to `.cache/` on first
use.

## Journal citations

`data/journal_citations.gml`: 85 high-volume journals across fields. An edge
u → v with weight w means articles in u published 2019–2023 cite v w times
(1.8 million citations in all). Node attributes: `name`, `cited_by`. The model is fitted with
`weighted_SBM=True`. Source: OpenAlex (CC0), https://openalex.org.

The model treats each citation like a win for the citing journal, so `plot.py`
flips the sign of the scores to put the most-cited journals (Nature, Science) at
the top. Group names are those of the paper's fit; `plot.py` matches each to the
inferred group sharing the most journals with it.
