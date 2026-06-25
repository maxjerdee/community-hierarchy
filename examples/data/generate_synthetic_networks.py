"""Generate synthetic directed SBM networks with planted partition and scores.

Filename convention: snr{SNR*100:.0f}_sigC{sigC:.0f}_rep{rep:02d}.gml

Parameters
----------
n         : total nodes (equally split into q groups)
q         : number of groups
mean_degree: mean undirected degree per node
snr       : signal-to-noise ratio = (a - b) / sqrt(q * b),
            where a = within-group mean degree, b = between-group mean degree
            to each other group. snr=1 is the Kesten-Stigum threshold.
sigma_C   : std of group mean scores (group hierarchy depth)
sigma_I   : std of individual score noise around group mean
nu        : Davidson tie parameter (0 = no ties, larger = more ties)
rep       : random seed (replicate index)
"""

from __future__ import annotations
import argparse
import math
import numpy as np
import networkx as nx
from pathlib import Path


def _solve_ab(mean_degree: float, snr: float, q: int) -> tuple[float, float]:
    """Solve for within-group (a) and between-group (b) mean degrees.

    Constraints:
        a + (q-1)*b = mean_degree
        (a - b) / sqrt(q * b) = snr
    """
    # Let x = sqrt(b): q*x^2 + snr*sqrt(q)*x - mean_degree = 0
    sq = math.sqrt(q)
    disc = snr**2 * q + 4 * q * mean_degree
    x = (-snr * sq + math.sqrt(disc)) / (2 * q)
    b = x**2
    a = mean_degree - (q - 1) * b
    if a <= 0 or b < 0:
        raise ValueError(f"snr={snr} too large for mean_degree={mean_degree}, q={q}: a={a:.4f}")
    return a, b


def generate(
    n: int = 200,
    q: int = 4,
    mean_degree: float = 5.0,
    snr: float = 1.5,
    sigma_C: float = 5.0,
    sigma_I: float = 1.0,
    nu: float = 0.1,
    rep: int = 0,
) -> nx.DiGraph:
    rng = np.random.default_rng(rep)
    assert n % q == 0, "n must be divisible by q"
    n_r = n // q

    # --- Solve for edge probabilities ---
    a, b = _solve_ab(mean_degree, snr, q)
    rho_in  = a / (n_r - 1)   # probability per within-group pair
    rho_out = b / n_r          # probability per between-group pair

    # --- Assign groups ---
    groups = np.repeat(np.arange(q), n_r)

    # --- Sample group means and individual scores ---
    group_means = rng.normal(0.0, sigma_C, size=q)
    scores = group_means[groups] + rng.normal(0.0, sigma_I, size=n)

    # --- Build directed graph ---
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, group=int(groups[i]), status=float(scores[i]))

    for i in range(n):
        for j in range(i + 1, n):
            p_edge = rho_in if groups[i] == groups[j] else rho_out
            if rng.random() >= p_edge:
                continue

            # Davidson (1970) direction model
            si, sj = scores[i], scores[j]
            # Normalise to avoid overflow: subtract max(si, sj)
            m = max(si, sj)
            ei = math.exp(si - m)
            ej = math.exp(sj - m)
            et = nu * math.exp((si + sj) / 2 - m)
            Z = ei + ej + et
            r = rng.random() * Z
            if r < ei:
                G.add_edge(i, j)          # i beats j
            elif r < ei + ej:
                G.add_edge(j, i)          # j beats i
            else:
                G.add_edge(i, j)          # tie: both directions
                G.add_edge(j, i)

    return G


def _filename(snr: float, sigma_C: float, rep: int) -> str:
    return f"snr{snr * 100:03.0f}_sigC{sigma_C:.0f}_rep{rep:02d}.gml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",           type=int,   default=200)
    parser.add_argument("--q",           type=int,   default=4)
    parser.add_argument("--mean-degree", type=float, default=5.0)
    parser.add_argument("--snr",         type=float, default=1.5)
    parser.add_argument("--sigma-C",     type=float, default=5.0)
    parser.add_argument("--sigma-I",     type=float, default=1.0)
    parser.add_argument("--nu",          type=float, default=0.1)
    parser.add_argument("--rep",         type=int,   default=9)
    parser.add_argument("--out-dir",     type=Path,
                        default=Path(__file__).parent / "networks")
    args = parser.parse_args()

    G = generate(
        n=args.n, q=args.q, mean_degree=args.mean_degree,
        snr=args.snr, sigma_C=args.sigma_C, sigma_I=args.sigma_I,
        nu=args.nu, rep=args.rep,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fname = args.out_dir / _filename(args.snr, args.sigma_C, args.rep)
    nx.write_gml(G, fname)
    n_und = sum(1 for u, v in G.to_undirected().edges())
    n_ties = sum(1 for u, v in G.edges() if G.has_edge(v, u)) // 2
    a, b = _solve_ab(args.mean_degree, args.snr, args.q)
    print(f"Saved: {fname}")
    print(f"  n={G.number_of_nodes()}, directed edges={G.number_of_edges()}, "
          f"undirected pairs={n_und}, ties={n_ties}")
    print(f"  Target a={a:.4f}, b={b:.4f}, rho_in={a/(args.n//args.q-1):.5f}, "
          f"rho_out={b/(args.n//args.q):.5f}")


if __name__ == "__main__":
    main()
