"""Predictive comparison of the coupled and independent models.

Two tests, both on matches the model scores from the fitted posterior:

1. The 32 knockout matches of the 2026 World Cup, which are out of sample (the network
   ends with the group stage).  KNOCKOUTS lists each (winner, loser, round, p_market); a
   match decided on penalties counts as won by the team that advanced.  p_market is
   the betting market's probability that the winner beats the loser: de-vigged
   pre-match 1X2 odds (OddsPortal, Pinnacle for some round-of-16 matches), with the
   draw probability split in proportion to the two win probabilities.  The model's
   probability is the Bradley-Terry probability with no draws,
   exp(s_w) / (exp(s_w) + exp(s_l)), averaged over the posterior samples.

2. Leave-one-match-out cross-validation (PSIS-LOO) of the 2,134 group-stage results.
   For every posterior sample and match, the log-probability of the observed result
   under the fitted model of directions with draws,
       P(a beats b) = e^{s_a} / Z,   P(draw) = nu e^{(s_a + s_b)/2} / Z,
       Z = e^{s_a} + e^{s_b} + nu e^{(s_a + s_b)/2};
   Pareto-smoothed importance sampling (Vehtari, Gelman & Gabry 2017, arviz.loo)
   turns these into each match's leave-one-out log predictive density without
   refitting (only the result is left out; the match stays in the network).  The same
   weights give each decisive match's leave-one-out probability that the winner beats
   the loser given that the match is decided (p_win): accuracy and log-likelihood are
   reported over the decisive matches, the elpd difference over all and decisive ones.

fit.py calls knockout_probabilities() and loo() on each fit (they need the samples) and
writes results/knockouts_<model>.csv and results/loo_<model>.csv;
    python predict.py [--quick]
prints the comparison.
"""
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

KNOCKOUTS = [
    # Round of 32
    ("Canada",        "South Africa",           "Round of 32",    0.7286),
    ("Brazil",        "Japan",                  "Round of 32",    0.7498),
    ("Paraguay",      "Germany",                "Round of 32",    0.1132),
    ("Morocco",       "Netherlands",            "Round of 32",    0.4268),
    ("Norway",        "Ivory Coast",            "Round of 32",    0.6336),
    ("France",        "Sweden",                 "Round of 32",    0.8943),
    ("Mexico",        "Ecuador",                "Round of 32",    0.6471),
    ("England",       "DR Congo",               "Round of 32",    0.9145),
    ("Belgium",       "Senegal",                "Round of 32",    0.6827),
    ("United States", "Bosnia and Herzegovina", "Round of 32",    0.8679),
    ("Spain",         "Austria",                "Round of 32",    0.8861),
    ("Portugal",      "Croatia",                "Round of 32",    0.7703),
    ("Switzerland",   "Algeria",                "Round of 32",    0.6923),
    ("Egypt",         "Australia",              "Round of 32",    0.5902),
    ("Argentina",     "Cape Verde",             "Round of 32",    0.9524),
    ("Colombia",      "Ghana",                  "Round of 32",    0.8612),
    # Round of 16
    ("France",        "Paraguay",               "Round of 16",    0.9375),
    ("Morocco",       "Canada",                 "Round of 16",    0.7337),
    ("Norway",        "Brazil",                 "Round of 16",    0.2674),
    ("England",       "Mexico",                 "Round of 16",    0.5390),
    ("Spain",         "Portugal",               "Round of 16",    0.6667),
    ("Belgium",       "United States",          "Round of 16",    0.4669),
    ("Argentina",     "Egypt",                  "Round of 16",    0.9002),
    ("Switzerland",   "Colombia",               "Round of 16",    0.4000),   # on penalties
    # Quarter-finals
    ("France",        "Morocco",                "Quarter-finals", 0.8043),
    ("Spain",         "Belgium",                "Quarter-finals", 0.7756),
    ("England",       "Norway",                 "Quarter-finals", 0.7059),
    ("Argentina",     "Switzerland",            "Quarter-finals", 0.7751),
    # Semi-finals
    ("Spain",         "France",                 "Semi-finals",    0.4590),
    ("Argentina",     "England",                "Semi-finals",    0.4750),
    # Third place
    ("England",       "France",                 "Third place",    0.3214),
    # Final
    ("Spain",         "Argentina",              "Final",          0.6160),
]


def knockout_probabilities(result):
    """One row per knockout: winner, loser, round, p_market and p, the model's
    probability that the winner beats the loser."""
    df = pd.DataFrame(KNOCKOUTS, columns=["winner", "loser", "round", "p_market"])
    df["p"] = [result.direction_probabilities(w.replace(" ", "_"), l.replace(" ", "_"),
                                              ties_allowed=False)["a_to_b"]
               for w, l in zip(df.winner, df.loser)]
    return df


def loo(result, G):
    """One row per match: winner (or first team of a draw), loser, draw, elpd (the
    match's leave-one-out log predictive density), pareto_k (above 0.7, that match's
    estimate is unreliable) and p_win (decisive matches: leave-one-out probability
    that the winner beats the loser, draws excluded)."""
    edges = list(G.edges(data=True))
    a = [u for u, _, _ in edges]
    b = [v for _, v, _ in edges]
    draw = np.array([d["type"] == "neutral" for *_, d in edges])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = result.to_frame()
    sa, sb = s[[f"{n}_score" for n in a]].to_numpy(), s[[f"{n}_score" for n in b]].to_numpy()
    log_nu = np.log(s["ties_parameter"].to_numpy())[:, None]
    log_z = np.logaddexp(np.logaddexp(sa, sb), log_nu + (sa + sb) / 2)
    ll = np.where(draw, log_nu + (sa + sb) / 2, sa) - log_z            # (samples, matches)
    chains = s["chain_id"].to_numpy()
    n = min((chains == c).sum() for c in np.unique(chains))           # equal-length chains
    ll = np.stack([ll[chains == c][:n] for c in np.unique(chains)])   # (chains, draws, matches)
    idata = az.from_dict(log_likelihood={"y": ll}, posterior={"mu": np.zeros(ll.shape[:2])})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = az.loo(idata, pointwise=True)
    # leave-one-out P(winner beats loser | decided): PSIS-weighted posterior average
    log_w, _ = az.psislw(-ll.reshape(-1, ll.shape[2]).T)              # (matches, samples)
    d = (sa - sb)[np.concatenate([np.flatnonzero(chains == c)[:n] for c in np.unique(chains)])]
    p_win = (np.exp(log_w) / (1 + np.exp(-d.T))).sum(1)
    return pd.DataFrame(dict(winner=a, loser=b, draw=draw, elpd=L.loo_i.values,
                             pareto_k=L.pareto_k.values, p_win=np.where(draw, np.nan, p_win)))


if __name__ == "__main__":
    RESULTS = Path(__file__).parent / "results"
    SUFFIX = "_quick" if "--quick" in sys.argv else ""

    df = pd.DataFrame(KNOCKOUTS, columns=["winner", "loser", "round", "market"])
    for model in ("coupled", "independent"):
        df[model] = pd.read_csv(RESULTS / f"knockouts_{model}{SUFFIX}.csv")["p"].values

    print(f"{len(df)} knockout matches of the 2026 World Cup: probability given to the winner")
    print(df[["round", "winner", "loser", "coupled", "independent", "market"]].round(3).to_string(index=False))
    print(f"\n{'':12s} {'correct':>8s} {'log-likelihood':>15s} {'mean per match':>15s}")
    for model in ("coupled", "independent", "market"):
        p = df[model]
        print(f"{model:12s} {int((p > 0.5).sum()):5d}/{len(p)} {np.log(p).sum():15.2f} {np.log(p).mean():15.3f}")

    print("\nLeave-one-match-out cross-validation of the group-stage results (PSIS-LOO)")
    loo = {m: pd.read_csv(RESULTS / f"loo_{m}{SUFFIX}.csv") for m in ("coupled", "independent")}
    for m, d in loo.items():
        dec = d[~d.draw]
        print(f"{m:12s} elpd {d.elpd.sum():9.1f}   decisive matches ({len(dec)}): called correctly "
              f"{(dec.p_win > 0.5).mean():.1%}, mean log P(winner | decided) {np.log(dec.p_win).mean():.3f}"
              f"   Pareto k > 0.7: {(d.pareto_k > 0.7).sum()}")
    diff = loo["coupled"].elpd - loo["independent"].elpd
    for label, d in (("all matches", diff), ("decisive only", diff[~loo["coupled"].draw])):
        se = np.sqrt(len(d)) * d.std()
        print(f"coupled - independent, {label:13s}: {d.sum():+6.1f} (se {se:.1f}, z {d.sum() / se:+.2f}) over {len(d)} matches")
