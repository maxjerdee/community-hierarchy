# communityhierarchy (R)

R interface to [community-hierarchy](https://github.com/maxjerdee/community-hierarchy):
joint inference of communities and ranking in directed networks of wins, losses and
draws, and of how much of the ranking the communities explain.  It runs the same C++
sampler as the Python package; with the same seed, the two give identical samples.

## Install

```r
# from R-universe (binaries for Windows, macOS and Linux)
install.packages("communityhierarchy",
                 repos = c("https://maxjerdee.r-universe.dev", "https://cloud.r-project.org"))
# or from source on GitHub (needs a C++17 compiler)
remotes::install_github("maxjerdee/community-hierarchy", subdir = "R-package")
```

## Example

```r
library(communityhierarchy)

games <- data.frame(winner = c("a", "a", "b", "c", "d", "a"),
                    loser  = c("b", "c", "c", "d", "e", "e"),
                    type   = c("dominant", "dominant", "dominant", "neutral", "dominant", "dominant"))
fit <- ch_fit(games, ch_config(), num_chains = 4, seed = 1)

network_properties(fit)          # sigma_I, sigma_G, sigma_T, ICC, nu, number of groups
node_properties(fit)             # score and consensus group of every node
direction_probabilities(fit, "a", "e")
```

`G` can also be a directed igraph graph.  An edge u -> v means u beat v; pass
`source_high_rank = FALSE` when the target is the winner (citations, nominations).
Edges with `type = "neutral"` are draws, and `weight` counts repeated interactions.
The model and its options are documented at
<https://community-hierarchy.readthedocs.io>; in R, see `?ch_fit` and `?ch_config`.

## Differences from the Python package

| Python | R |
|---|---|
| `fit(Config(...), G, ...)` | `ch_fit(G, ch_config(...), ...)` |
| groups numbered from 0 | groups numbered from 1 |
| `result.to_frame()` | `samples(fit)`, `samples(fit, "scores")`, `samples(fit, "groups")` |
| `direction_probabilities` keys `a_to_b`, `b_to_a` | names `a_wins`, `b_wins` |
| default seed from numpy | default seed from R's generator (`set.seed()` makes it repeatable) |

Both start from the same default state, computed by the C++ core (greedy-modularity
groups and win-fraction scores), so with the same seed they give identical samples
(Python 0.6.0 or later).

## Development

The C++ core in `src/` is a copy of the repository's `src/`.  Edit it there, then

```bash
python tools/sync_r_core.py               # copy into R-package/src/
Rscript -e 'Rcpp::compileAttributes("R-package"); roxygen2::roxygenise("R-package")'
R CMD build R-package && R CMD check --as-cran communityhierarchy_*.tar.gz
```

`tests/python/test_r_core_sync.py` fails if the copy is out of date.  The parity
fixture in `tests/testthat/fixtures/` is made by `tools/make_r_parity_fixture.py`.
