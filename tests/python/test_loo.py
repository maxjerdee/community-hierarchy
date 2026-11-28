"""Leave-one-out cross-validation of the edge directions (src/loo.h)."""
import pickle
import random
import warnings

import networkx as nx
import numpy as np
import pytest

import community_hierarchy as ch
from community_hierarchy import MCMC_core


def _graph(seed=3, n=12, m=40):
    rng = random.Random(seed)
    nodes = [f"n{i}" for i in range(n)]
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    for _ in range(m):
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v, type="neutral" if rng.random() < 0.2 else "dominant",
                   weight=rng.choice([1, 1, 2]))
    return G


def _fit(G, **kw):
    kw = dict(num_chains=2, num_samples=200, num_warmup=20, seed=5,
              show_progress=False, timeout=600) | kw
    return ch.fit(ch.Config(), G, **kw)


def test_fixed_scores_give_the_plain_log_likelihood():
    # Every sample is the same, so the weights are uniform and the leave-one-out
    # density is the likelihood itself.
    G = _graph()
    s = {node: 0.3 * i - 1.5 for i, node in enumerate(G.nodes())}
    nx.set_node_attributes(G, s, "s")
    nu = 0.7
    res = ch.fit(ch.Config(hierarchy_attribute="s", ties_parameter=nu), G, num_chains=1,
                 num_samples=30, num_warmup=2, seed=1, show_progress=False)
    d = res.direction_loo_df()
    assert len(d) == G.number_of_edges()
    for row, (u, v, e) in zip(d.itertuples(), G.edges(data=True)):
        sa, sb = s[row.winner], s[row.loser]
        z = np.exp(sa) + np.exp(sb) + nu * np.exp((sa + sb) / 2)
        p = (nu * np.exp((sa + sb) / 2) if row.draw else np.exp(sa)) / z
        assert row.elpd == pytest.approx(e["weight"] * np.log(p), rel=1e-12)
        assert (row.winner, row.loser) == ((u, v) if row.draw or e["type"] == "dominant" else (v, u))
        if row.draw:
            assert np.isnan(row.p_win)
        else:
            assert row.p_win == pytest.approx(1 / (1 + np.exp(sb - sa)), rel=1e-12)


def test_source_high_rank_false_puts_the_target_first():
    G = _graph()
    d = _fit(G, source_high_rank=False).direction_loo_df()
    for row, (u, v, e) in zip(d.itertuples(), G.edges(data=True)):
        assert (row.winner, row.loser) == ((u, v) if e["type"] == "neutral" else (v, u))


def test_survives_pickling():
    res = _fit(_graph())
    again = pickle.loads(pickle.dumps(res))
    assert np.array_equal(again.direction_loo_df().elpd, res.direction_loo_df().elpd)


def test_psis_matches_arviz():
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(0)
    for S in (5, 30, 100, 1000, 4000):
        for r_eff in (1.0, 0.6):
            lr = rng.standard_t(df=rng.uniform(0.5, 10), size=S) * rng.uniform(0.1, 5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lw_a, k_a = az.psislw(lr.copy(), reff=r_eff)
            lw_c, k_c = MCMC_core.psis(list(lr), r_eff)
            np.testing.assert_allclose(lw_c, lw_a, rtol=0, atol=1e-10)
            assert k_c == pytest.approx(float(k_a), abs=1e-10) if np.isfinite(k_a) else np.isinf(k_c)


def test_direction_loo_matches_arviz_loo():
    az = pytest.importorskip("arviz")
    G = _graph()
    res = _fit(G)
    d = res.direction_loo_df()
    s = res.to_frame()
    sa = s[[f"{n}_score" for n in d.winner]].to_numpy()
    sb = s[[f"{n}_score" for n in d.loser]].to_numpy()
    log_nu = np.log(s["ties_parameter"].to_numpy())[:, None]
    log_z = np.logaddexp(np.logaddexp(sa, sb), log_nu + (sa + sb) / 2)
    ll = d.weight.to_numpy() * (np.where(d.draw, log_nu + (sa + sb) / 2, sa) - log_z)
    ll = ll.reshape(2, -1, ll.shape[1])                       # (chains, draws, edges)
    idata = az.from_dict(log_likelihood={"y": ll}, posterior={"mu": np.zeros(ll.shape[:2])})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = az.loo(idata, pointwise=True)
    np.testing.assert_allclose(d.elpd, L.loo_i.values, rtol=0, atol=1e-10)
    np.testing.assert_allclose(d.pareto_k, L.pareto_k.values, rtol=0, atol=1e-10)
