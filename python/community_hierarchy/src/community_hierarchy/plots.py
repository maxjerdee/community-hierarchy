"""Plots of a fit."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .result import Result


def plot_node_properties(
    result,
    *,
    partition: str = "auto",
    consensus_method: str = "moc",
    ax=None,
    palette=None,
    xlabel: str = "Community (inferred)",
    ylabel: str = r"$s_i$ (status score)",
    group_labels=None,
    labels: str = "ticks",
    flip: bool = False,
    annotate=None,
    filled=None,
    group_gap: float = 5.0,
    ladder_step=None,
    annotation_fontsize: float = 6.0,
    output_filename: Optional[str] = None,
):
    """Plot posterior node scores ranked within each inferred community.

    Each node is a dot at its rank within its community, with a vertical error bar
    of +-1 posterior standard deviation; a horizontal line marks each community's
    mean score.  Communities run left to right in ascending order of mean score,
    and each takes as much width as it has nodes.

    Parameters
    ----------
    result : Result or pandas.DataFrame
        A fit from :func:`community_hierarchy.fit`, or a table in the format of
        :meth:`Result.node_properties_df` (columns ``node``, ``score_mean``,
        ``score_std``, ``group``), e.g. one saved to CSV.
    partition, consensus_method : str
        For a ``Result``: passed to :meth:`Result.node_properties_df`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If *None* a new figure is created.
    palette : list, dict, or colormap, optional
        Community colours: a list in ascending mean-score order, a dict
        ``{group: colour}``, or a colormap.  Defaults to ``tab10`` (``tab20`` for
        more than 10 communities).
    xlabel, ylabel : str
        Axis labels (``xlabel`` is not shown with ``labels="ladder"``).
    group_labels : list or dict, optional
        Community names: a list in ascending mean-score order or a dict
        ``{group: name}``.  Defaults to "G1", "G2", ...
    labels : {"ticks", "ladder", "none"}
        Where community names go: as x-axis ticks (default), as coloured labels
        below the data, rising one rung per community from left to right so that
        neighbouring names never collide, or nowhere (communities are told apart
        by colour only).
    flip : bool
        Plot minus the scores, so that low rank is at the top.  Not needed
        to correct edge direction: pass ``source_high_rank=False`` to
        :func:`community_hierarchy.fit` instead, after which higher scores
        always mean higher rank.  Default False.
    annotate : dict, optional
        ``{node: text}``: rotated labels above these nodes' error bars.
    filled : collection, optional
        Nodes drawn as filled markers; the rest are hollow.  *None*: all filled.
    group_gap : float
        Gap between communities, in units of the spacing between nodes.
    ladder_step : float, optional
        Rise of the label ladder per community, in score units.  Defaults to
        3.5% of the score range.
    annotation_fontsize : float
        Font size of the ``annotate`` labels.
    output_filename : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, ax : tuple
    """
    import matplotlib.patheffects as pe


    if isinstance(result, Result):
        node_df = result.node_properties_df(partition=partition, consensus_method=consensus_method)
    elif isinstance(result, pd.DataFrame):
        node_df = result
    else:
        raise TypeError(
            "result must be a community_hierarchy.Result or a node_properties_df() table."
        )
    if labels not in ("ticks", "ladder", "none"):
        raise ValueError(f"labels must be 'ticks', 'ladder' or 'none', got {labels!r}")
    df = node_df[["node", "score_mean", "score_std", "group"]].copy()
    if flip:
        df["score_mean"] = -df["score_mean"]

    grp_order = df.groupby("group")["score_mean"].mean().sort_values().index.tolist()
    K = len(grp_order)

    if palette is None:  # tab10, or tab20 for more than 10 communities
        cmap_tab, n_col = (plt.get_cmap("tab10"), 10) if K <= 10 else (plt.get_cmap("tab20"), 20)
        color_of = {g: cmap_tab(i % n_col) for i, g in enumerate(grp_order)}
    elif isinstance(palette, dict):
        color_of = {g: palette.get(g, "#9e9e9e") for g in grp_order}
    elif callable(palette):
        color_of = {g: palette(i / max(K - 1, 1)) for i, g in enumerate(grp_order)}
    else:
        color_of = {g: list(palette)[i] for i, g in enumerate(grp_order)}

    if group_labels is None:
        label_of = {g: f"G{i + 1}" for i, g in enumerate(grp_order)}
    elif isinstance(group_labels, dict):
        label_of = {g: group_labels.get(g, "") for g in grp_order}
    else:
        label_of = {g: str(group_labels[i]) for i, g in enumerate(grp_order)}

    if ax is None:
        w = min(14.0, max(8.0, len(df) * 0.12))
        fig, ax = plt.subplots(figsize=(w, w / 2))
    else:
        fig = ax.get_figure()
    filled = None if filled is None else set(filled)
    annotate = annotate or {}

    pos, centres, floor = 0.0, {}, np.inf
    for g in grp_order:
        grp = df[df["group"] == g].sort_values("score_mean")
        n_g = len(grp)
        xs = pos + np.arange(n_g)
        c = color_of[g]
        alpha, lw = max(0.25, 0.6 - 0.005 * n_g), max(0.3, 0.6 - 0.004 * n_g)
        ax.errorbar(
            xs,
            grp["score_mean"],
            yerr=grp["score_std"],
            fmt="none",
            color=c,
            elinewidth=lw,
            capsize=0,
            alpha=alpha,
            zorder=2,
        )
        full = np.array([filled is None or nd in filled for nd in grp["node"]])
        for sel, face in ((full, c), (~full, "none")):
            ax.scatter(
                xs[sel],
                grp["score_mean"].values[sel],
                s=10,
                facecolors=face,
                edgecolors=c,
                linewidths=0.6,
                alpha=min(alpha + 0.3, 1.0),
                zorder=3,
            )
        ax.hlines(
            grp["score_mean"].mean(), xs[0] - 0.4, xs[-1] + 0.4, colors=c, linewidth=1.6, zorder=4
        )
        for x, (_, row) in zip(xs, grp.iterrows()):
            if row["node"] in annotate:
                ax.annotate(
                    str(annotate[row["node"]]),
                    xy=(x, row["score_mean"] + row["score_std"]),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=annotation_fontsize,
                    color="#222222",
                    zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground="white")],
                )
        centres[g] = float(xs.mean())
        floor = min(floor, float((grp["score_mean"] - grp["score_std"]).min()))
        pos += n_g - 1 + group_gap

    ax.axhline(0, color="#aaaaaa", lw=0.7, alpha=0.6, zorder=1)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-group_gap / 2, pos - group_gap / 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if labels == "ticks":
        ax.set_xticks([centres[g] for g in grp_order])
        ax.set_xticklabels([label_of[g] for g in grp_order])
        ax.set_xlabel(xlabel)
    elif labels == "none":
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    else:
        ylo, yhi = ax.get_ylim()
        step = ladder_step if ladder_step is not None else 0.035 * (yhi - ylo)
        y0 = floor - step
        for i, g in enumerate(grp_order):
            if label_of[g]:
                ax.text(
                    centres[g],
                    y0 + i * step,
                    label_of[g],
                    ha="center",
                    va="top",
                    fontsize=7,
                    fontweight="bold",
                    color=color_of[g],
                    clip_on=False,
                )
        ax.set_ylim(min(ylo, y0 - 2 * step), yhi)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)

    if output_filename is not None:
        fig.savefig(output_filename)

    return fig, ax


def plot_adjacency_matrix(
    result: "Result",
    G: nx.Graph,
    *,
    partition: str = "auto",
    consensus_method: str = "moc",
    cmap=None,
    ax=None,
    title: str = "",
    strip_width: float = 0.06,
    palette=None,
    vmax_percentile: float = 98.0,
    output_filename: Optional[str] = None,
):
    """Plot the adjacency matrix of *G* reordered by inferred community.

    Nodes are sorted first by ascending community mean score, then by
    ascending individual score within each community. Colored strips on
    the top and left edges of the matrix indicate community membership.
    Vertical and horizontal lines mark community boundaries.

    Parameters
    ----------
    result : Result
        Inference result from :func:`community_hierarchy.fit`. Provides the
        node ordering and community labels; does not need to be the result
        of fitting *G* specifically (e.g. you can pass a grooming graph
        while ``result`` was fitted on an aggression graph).
    G : nx.Graph
        Graph whose adjacency to display. Edge weights are taken from the
        ``weight`` attribute; missing weights default to 1.0.
    partition : str
        Which partition to use; passed to :meth:`Result.node_properties_df`.
    consensus_method : {"moc", "moc_modal_k", "binder"}
        Consensus method; passed to :meth:`Result.node_properties_df`.
    cmap : str or colormap, optional
        Colormap for the matrix cells. Defaults to ``"Greens"``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If *None* a new figure is created.
    title : str
        Title text shown above the matrix (on the top color strip).
    strip_width : float
        Width of the community color strip as a fraction of the matrix
        axis size. Default 0.06.
    palette : list of colors or colormap, optional
        Colors for each community in ascending mean-score order.
        Defaults to ``tab10``.
    vmax_percentile : float
        Percentile of positive edge weights used as the color scale maximum.
        Default 98.
    output_filename : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, ax : tuple
    """
    import matplotlib.patches as mpatches


    if not isinstance(result, Result):
        raise TypeError("result must be a community_hierarchy.Result instance.")
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph.")

    node_df = result.node_properties_df(partition=partition, consensus_method=consensus_method)

    # Sort communities by ascending group mean score, nodes within by score
    grp_mean = node_df.groupby("group")["group_score_mean"].first().sort_values()
    grp_order = grp_mean.index.tolist()
    K = len(grp_order)

    sorted_df = (
        node_df.assign(_grp_rank=node_df["group"].map({g: i for i, g in enumerate(grp_order)}))
        .sort_values(["_grp_rank", "score_mean"])
        .reset_index(drop=True)
    )
    node_order = sorted_df["node"].tolist()
    grp_sizes = sorted_df.groupby("_grp_rank", sort=True).size().values
    boundaries = list(np.concatenate([[0], np.cumsum(grp_sizes)]))

    # Build palette
    if palette is None:
        cmap_tab = plt.get_cmap("tab10")
        colors = [cmap_tab(i % 10) for i in range(K)]
    elif hasattr(palette, "__call__"):
        colors = [palette(i / max(K - 1, 1)) for i in range(K)]
    else:
        colors = list(palette)

    # Build adjacency matrix
    n = len(node_order)
    idx = {nd: i for i, nd in enumerate(node_order)}
    A = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            w = d.get("weight", 1.0)
            A[idx[u], idx[v]] += w
            if not G.is_directed():
                A[idx[v], idx[u]] += w

    # Colormap
    if cmap is None:
        cmap = "Greens"

    # Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    vmax = float(np.percentile(A[A > 0], vmax_percentile)) if (A > 0).any() else 1.0
    ax.imshow(
        A, aspect="equal", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", origin="upper"
    )

    # Community grid lines
    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color="black", lw=0.5, zorder=3)
        ax.axhline(b - 0.5, color="black", lw=0.5, zorder=3)

    # Diagonal
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="black", lw=0.4, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])

    # Color strips — finalize layout first so get_position() is accurate
    fig.canvas.draw()
    pos = ax.get_position()
    sh = strip_width * pos.height
    sw = strip_width * pos.width

    ax_top = fig.add_axes([pos.x0, pos.y1, pos.width, sh])
    ax_left = fig.add_axes([pos.x0 - sw, pos.y0, sw, pos.height])

    for k, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        c = colors[k]
        ax_top.add_patch(mpatches.Rectangle((s, 0), e - s, 1, color=c, ec="none"))
        ax_left.add_patch(mpatches.Rectangle((0, s), 1, e - s, color=c, ec="none"))

    ax_top.set_xlim(0, n)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(n, 0)
    ax_left.axis("off")

    if title:
        ax_top.text(
            0.5,
            1.1,
            title,
            transform=ax_top.transAxes,
            fontsize=plt.rcParams.get("font.size", 9) - 1,
            va="bottom",
            ha="center",
        )

    if output_filename is not None:
        fig.savefig(output_filename, bbox_inches="tight")

    return fig, ax


def plot_MCMC_entropy(
    result: "Result",
    *,
    ax=None,
    output_filename: Optional[str] = None,
):
    """Trace plot of the description length over the samples, one line per chain.

    The description length of a sample is ``H_A_b + H_Adir_G_A_s_nu +
    H_s_G_b_depths`` in nats (see :meth:`Result.samples_df`); a trace that
    is still drifting, or chains that sit at different levels, mean the fit
    has not converged.

    Parameters
    ----------
    result : Result
        Inference result from :func:`community_hierarchy.fit`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If *None* a new figure is created.
    output_filename : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, ax : tuple
    """

    if not isinstance(result, Result):
        raise TypeError("result must be a community_hierarchy.Result instance.")

    samples_df = result.to_frame()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    dl = samples_df["H_A_b"] + samples_df["H_Adir_G_A_s_nu"] + samples_df["H_s_G_b_depths"]
    chains = samples_df["chain_id"].unique() if len(samples_df) else []
    for c in chains:
        sel = (samples_df["chain_id"] == c).to_numpy()
        ax.plot(
            np.arange(sel.sum()),
            dl[sel].to_numpy(),
            lw=1.0,
            label=f"chain {c}" if len(chains) > 1 else None,
        )
    if len(chains) > 1:
        ax.legend(frameon=False)
    ax.set_xlabel("Sample index within chain")
    ax.set_ylabel("Description length (nats)")
    ax.set_title("Convergence of MCMC")

    if output_filename is not None:
        fig.savefig(output_filename)

    return fig, ax
