"""
WASM parity tests: ds_wasm.cpp must reproduce Python's exact results for the same seed.

Three classes of regression are tested:

1. TestParameterDraws
   Checks that WASM draws vi, vo, id, gd, tp from mt19937_64(seed*2654435761) in
   the correct order.  Catches the libc++/libstdc++ uniform_real_distribution
   double-draw bug and the degree_correction eager-eval (value_or) bug.

2. TestInitialDLParity
   Passes an identical explicit initial state (partition + scores) to both the
   Python fit_backend and the WASM session with sweeps_per_sample=0.
   Compares the description length to 1e-6 relative tolerance.  Catches the
   group_adj pre-sizing bug (WASM was accidentally enabling two-layer mode,
   zeroing H_M/H_k/H_A_G_k_M_b).

3. TestConvergence
   Runs a short fit (burn=50, num_samples=10) with merge_split enabled on both
   backends.  Asserts WASM's MDL value is EXACTLY Python's (both RNG draws
   and container iteration order are now portable across libc++/libstdc++ --
   see src/rng.h -- so a seeded run is fully pinned; this used to be a loose
   20% band before that fix, back when the two backends could genuinely
   diverge on the same seed).

Run from the repository root:
    python -m pytest tests/test_wasm_parity.py -v
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import networkx as nx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WASM_JS = os.path.join(REPO, "bindings", "wasm", "js", "index.js")
PYTHON_SRC = os.path.join(REPO, "python", "community_hierarchy", "src")

if PYTHON_SRC not in sys.path:
    sys.path.insert(0, PYTHON_SRC)

# Skip the whole module when node or the WASM binary are unavailable
try:
    subprocess.run(["node", "--version"], capture_output=True, check=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    pytest.skip("node not found — skipping WASM parity tests", allow_module_level=True)

if not os.path.isfile(WASM_JS):
    pytest.skip(f"WASM bundle not built ({WASM_JS}) — run cmake/make first",
                allow_module_level=True)

# ---------------------------------------------------------------------------
# WASM helper
# ---------------------------------------------------------------------------
def wasm_fit(wins, ties, options, timeout=120):
    """
    Run a single WASM fit via a Node.js subprocess.
    wins / ties are edge lists: [[u, v, w], …] (weight optional).
    options maps directly to the js/index.js fit() options object;
    initialPartition / initialScores may be plain Python lists.
    Returns the parsed result JSON dict.
    """
    wasm_path = WASM_JS.replace("\\", "/")
    script = f"""
import {{ load }} from "{wasm_path}";
const m = await load();
const r = await m.fit(
  {json.dumps(wins)},
  {json.dumps(ties)},
  {json.dumps(options)}
).result;
m.terminate();
process.stdout.write(JSON.stringify(r));
"""
    with tempfile.NamedTemporaryFile(suffix=".mjs", mode="w", delete=False) as f:
        f.write(script)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            ["node", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        os.unlink(tmp_path)
    if proc.returncode != 0:
        raise RuntimeError(f"WASM subprocess failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Synthetic test network  (8 nodes, 2 planted communities, wins + ties)
#
#   Community A = {0,1,2,3},  community B = {4,5,6,7}
#   Intra-A wins:   0→1 (w=2), 0→2 (w=1), 2→3 (w=2)
#   Intra-B wins:   4→5 (w=1), 4→6 (w=2), 6→7 (w=2)
#   Inter wins (A beats B): 0→4 (w=1), 1→5 (w=1), 3→7 (w=1)
#   Intra-A ties:   0–3 (w=1), 1–3 (w=1)
#   Intra-B ties:   4–7 (w=1), 5–7 (w=1)
#
#   No pair appears in both WINS and TIES (avoids nx.DiGraph edge-type collision).
# ---------------------------------------------------------------------------
WINS = [
    [0, 1, 2], [0, 2, 1], [2, 3, 2],
    [4, 5, 1], [4, 6, 2], [6, 7, 2],
    [0, 4, 1], [1, 5, 1], [3, 7, 1],
]
TIES = [
    [0, 3, 1], [1, 3, 1],
    [4, 7, 1], [5, 7, 1],
]
N = 8
SEED = 42

# Explicit balanced 2-community partition supplied to both backends.
PARTITION = [0, 0, 0, 0, 1, 1, 1, 1]

# Scores with known unit-RMS norm (arbitrary relative ordering).
_RAW_SCORES = np.array([1.2, 0.8, 0.3, 0.5, -0.4, -0.7, -0.9, -0.6], dtype=float)
_RMS = float(np.sqrt(np.mean(_RAW_SCORES ** 2)))
SCORES_NORMED = (_RAW_SCORES / _RMS).tolist()   # unit-RMS, fed directly to both

# ---------------------------------------------------------------------------
# Reference parameter values from mt19937_64(seed * 2654435761).
#
# Draw order (no explicit scores → id drawn from init_rng):
#   draw1 → vi, draw2 → vo, draw3 → discarded (dc eager-eval in fit.cpp),
#   draw4 → id,  draw5 → gd,  draw6 → tp
#
# Values pre-computed with g++ mt19937_64(42*2654435761):
# ---------------------------------------------------------------------------
_DRAWS = [
    0.698254558812732,   # draw1 → vi
    0.053127790888864,   # draw2 → vo
    0.443816873552025,   # draw3 → discarded (dc value_or eager eval)
    0.587370122562773,   # draw4 → id  (auto scores) / gd (explicit scores)
    0.035126700331400,   # draw5 → gd  (auto scores) / tp (explicit scores)
    0.802060025783624,   # draw6 → tp  (auto scores)
]

EXPECTED_VI = _DRAWS[0]
EXPECTED_VO = _DRAWS[1]
EXPECTED_ID_AUTO  = _DRAWS[3]
EXPECTED_GD_AUTO  = _DRAWS[4]
EXPECTED_TP_AUTO  = _DRAWS[5]

DRAW_TOL = 1e-10


# ===========================================================================
# 1. Parameter draw correctness
# ===========================================================================
class TestParameterDraws:
    """
    Both backends must draw vi, vo, id, gd, tp from the same mt19937_64 stream.
    This directly validates the libc++ / libstdc++ uniformity fix and the
    degree_correction eager-eval fix in ds_wasm.cpp.
    """

    @pytest.fixture(scope="class")
    def wasm_sample(self):
        return wasm_fit(WINS, TIES, {
            "num_nodes": N,
            "seed": SEED,
            "num_warmup": 0,
            "sweeps_per_sample": 0,
            "num_samples": 1,
            "assortative": True,
            "variation_in": -1,     # inferred: general_ASBM
            "variation_out": -1,
            "merge_split_enabled": False,
        })["samples"][0]

    @pytest.fixture(scope="class")
    def python_params(self):
        import community_hierarchy as ch
        G = _build_nx_graph()
        result = ch.fit(
            ch.Config(groups_model="general_ASBM"), G,
            num_samples=1, seed=SEED,
            num_warmup=0, sweeps_per_sample=0,
            merge_split_enabled=False,
        )
        npdf = result.network_properties_df()
        return dict(zip(npdf["parameter"], npdf["mean"]))

    def test_wasm_vi(self, wasm_sample):
        assert abs(wasm_sample["variationIn"] - EXPECTED_VI) < DRAW_TOL

    def test_wasm_vo(self, wasm_sample):
        assert abs(wasm_sample["variationOut"] - EXPECTED_VO) < DRAW_TOL

    def test_wasm_dc_fixed(self, wasm_sample):
        assert abs(wasm_sample["degreeCorrection"] - 0.5) < DRAW_TOL

    def test_wasm_id(self, wasm_sample):
        assert abs(wasm_sample["individualDepth"] - EXPECTED_ID_AUTO) < DRAW_TOL

    def test_wasm_gd(self, wasm_sample):
        assert abs(wasm_sample["groupDepth"] - EXPECTED_GD_AUTO) < DRAW_TOL

    def test_wasm_tp(self, wasm_sample):
        assert abs(wasm_sample["tiesParameter"] - EXPECTED_TP_AUTO) < DRAW_TOL

    def test_python_vi(self, python_params):
        assert abs(python_params["variation_in"] - EXPECTED_VI) < DRAW_TOL

    def test_python_vo(self, python_params):
        assert abs(python_params["variation_out"] - EXPECTED_VO) < DRAW_TOL

    def test_python_id(self, python_params):
        assert abs(python_params["individual_depth"] - EXPECTED_ID_AUTO) < DRAW_TOL

    def test_python_gd(self, python_params):
        assert abs(python_params["group_depth"] - EXPECTED_GD_AUTO) < DRAW_TOL

    def test_python_tp(self, python_params):
        assert abs(python_params["ties_parameter"] - EXPECTED_TP_AUTO) < DRAW_TOL


# ===========================================================================
# 2. Initial DL parity — identical explicit state, both backends
# ===========================================================================
class TestInitialDLParity:
    """
    Feed the same explicit partition and scores to both backends with
    sweeps_per_sample=0 (initial state only, no MCMC moves).
    The description length must agree to 1e-6 relative tolerance.

    A structural bug such as the group_adj pre-sizing error (which zeroed
    H_M, H_k, H_A_G_k_M_b in WASM) would produce a DL ~3× lower than
    Python's and would be caught here.
    """

    _COMMON_OPTS = dict(
        seed=SEED,
        num_warmup=0,
        sweeps_per_sample=0,
        num_samples=1,
        assortative=True,
        variation_in=-1,      # inferred: general_ASBM
        variation_out=-1,
        merge_split_enabled=False,
    )

    @pytest.fixture(scope="class")
    def python_dl(self):
        """Python fit with explicit partition+scores, sweeps=0 → initial DL."""
        import community_hierarchy as ch
        G = _build_nx_graph()
        result = ch.fit(
            ch.Config(groups_model="general_ASBM"), G,
            num_samples=1, seed=SEED,
            num_warmup=0, sweeps_per_sample=0,
            merge_split_enabled=False,
            initial_partition=PARTITION,
            initial_scores=SCORES_NORMED,
        )
        return result.to_frame()["H_A_b"].iloc[0] \
             + result.to_frame()["H_Adir_G_A_s_nu"].iloc[0] \
             + result.to_frame()["H_s_G_b_depths"].iloc[0]

    @pytest.fixture(scope="class")
    def wasm_dl(self):
        result = wasm_fit(WINS, TIES, {
            "num_nodes": N,
            **self._COMMON_OPTS,
            "initialPartition": PARTITION,
            "initialScores": SCORES_NORMED,
        })
        return result["samples"][0]["descriptionLength"]

    def test_dl_matches_reference(self, python_dl):
        """Python DL must match the known-good reference value."""
        # 80.800219 before the multi-edge term was counted once per pair (Peixoto 2017):
        # the four weight-2 pairs (0,1), (2,3), (4,6), (6,7) had log 2! counted twice.
        reference = 78.027630
        rel_err = abs(python_dl - reference) / abs(reference)
        assert rel_err < 1e-5, (
            f"Python DL {python_dl:.6f} deviates from reference {reference}"
        )

    def test_wasm_dl_matches_python(self, python_dl, wasm_dl):
        """WASM DL for the explicit initial state matches Python to 1e-6."""
        rel_err = abs(wasm_dl - python_dl) / abs(python_dl)
        assert rel_err < 1e-6, (
            f"DL mismatch: WASM={wasm_dl:.6f}, Python={python_dl:.6f}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_wasm_dl_is_finite_and_positive(self, python_dl, wasm_dl):
        """DL must be finite and not suspiciously small (guards H_M=0 collapse)."""
        assert math.isfinite(wasm_dl), f"DL is not finite: {wasm_dl}"
        assert wasm_dl > python_dl * 0.5, (
            f"DL suspiciously low ({wasm_dl:.1f} vs Python {python_dl:.1f}); "
            "H_M / H_k may be zero — check group_adj or two_layer_mode"
        )


# ===========================================================================
# 3. Convergence parity — short run, both backends should reach similar MDL
# ===========================================================================
class TestConvergence:
    """
    A short run (50 burn + 10 samples) with merge_split enabled must reach the
    EXACT same MDL on both backends -- now that ds's RNG draws and container
    iteration order are portable (src/rng.h, ordered proposal_weights/
    std::set/sorted adjacency), a seeded run is fully pinned, so any
    divergence here is a real regression, not sampling noise. (Was a loose
    20% band before that fix landed; kept exact now specifically so a future
    reintroduction of an unordered_map or a raw std::uniform_*_distribution
    in the sweep/proposal path gets caught here instead of silently
    reappearing as the 68.75-vs-70.32-style drift this test was originally
    written to paper over.)
    """

    @pytest.fixture(scope="class")
    def python_result(self):
        import community_hierarchy as ch
        G = _build_nx_graph()
        return ch.fit(
            ch.Config(groups_model="general_ASBM"), G,
            num_samples=10, seed=SEED,
            num_warmup=50, sweeps_per_sample=1,
            merge_split_enabled=True,
        )

    @pytest.fixture(scope="class")
    def python_mdl(self, python_result):
        return python_result.mdl_value

    @pytest.fixture(scope="class")
    def wasm_result(self):
        return wasm_fit(WINS, TIES, {
            "num_nodes": N,
            "seed": SEED,
            "num_warmup": 50,
            "num_samples": 10,
            "sweeps_per_sample": 1,
            "assortative": True,
            "variation_in": -1,     # inferred: general_ASBM
            "variation_out": -1,
            "merge_split_enabled": True,
        })

    def test_wasm_mdl_matches_python_exactly(self, python_mdl, wasm_result):
        """WASM MDL must equal Python's to numerical precision after a short run."""
        wasm_mdl = wasm_result["mdlValue"]
        assert math.isfinite(wasm_mdl), f"WASM MDL is not finite: {wasm_mdl}"
        assert wasm_mdl == pytest.approx(python_mdl, rel=0, abs=1e-9), (
            f"WASM={wasm_mdl!r} != Python={python_mdl!r} "
            f"(diff={wasm_mdl - python_mdl:.3e}) -- a seeded run should be "
            "bit-for-bit identical now; see class docstring."
        )

    def test_wasm_consensus_matches_python_exactly(self, python_result, wasm_result):
        """The consensus partition must match too, not just the description length.

        These are different failure modes: the consensus is Dahl's pick at the
        modal group count followed by a greedy refinement at that count, so it can
        change with NO change to any sample's description length -- a stale wasm
        bundle passes an MDL-only comparison while returning a different
        partition. That is exactly what happened when the refinement landed
        (ds 0245bd5).
        """
        py = [int(x) for x in python_result.consensus_partition()]
        raw = wasm_result["consensusPartition"]
        wasm = [int(raw[str(i)] if isinstance(raw, dict) else raw[i]) for i in range(len(py))]
        assert len(wasm) == len(py)
        # Compare as partitions: labels are arbitrary, the grouping is not.
        def canon(p):
            m, out = {}, []
            for x in p:
                m.setdefault(x, len(m))
                out.append(m[x])
            return out
        assert canon(wasm) == canon(py), (
            f"consensus differs.\n  python: {canon(py)}\n  wasm  : {canon(wasm)}\n"
            "If the DL test passed but this failed, the wasm bundle is almost "
            "certainly stale -- rebuild it."
        )

    def test_wasm_k_at_least_two(self, wasm_result):
        """K should be ≥ 2 (guards against collapse to a single community)."""
        mdl_k = max(wasm_result["mdlPartition"].values()) + 1
        assert mdl_k >= 2, f"WASM collapsed to K=1 for N={N} nodes"


class TestConvergenceMultiSweep(TestConvergence):
    """As TestConvergence with sweeps_per_sample = 3, which pins the burn-in
    semantics: both backends burn in for num_warmup * sweeps_per_sample sweeps
    (the wasm driver once ran num_warmup sweeps, which sweeps_per_sample = 1
    could not detect)."""

    @pytest.fixture(scope="class")
    def python_result(self):
        import community_hierarchy as ch
        G = _build_nx_graph()
        return ch.fit(
            ch.Config(groups_model="general_ASBM"), G,
            num_samples=10, seed=SEED,
            num_warmup=20, sweeps_per_sample=3,
            merge_split_enabled=True,
        )

    @pytest.fixture(scope="class")
    def wasm_result(self):
        return wasm_fit(WINS, TIES, {
            "num_nodes": N,
            "seed": SEED,
            "num_warmup": 20,
            "num_samples": 10,
            "sweeps_per_sample": 3,
            "assortative": True,
            "variation_in": -1,     # inferred: general_ASBM
            "variation_out": -1,
            "merge_split_enabled": True,
        })


# ===========================================================================
# 4. Independent coupling parity — group_depth fixed at 0.0
# ===========================================================================
class TestIndependentCoupling:
    """
    DS Config(interaction='independent') fixes group_depth=0.0 (hierarchy
    decoupled from communities).  WASM must reproduce Python's exact MDL and
    consensus partition for the same seed, because:
    - The gd draw is still consumed (value_or is eager in fit.cpp), so the
      subsequent tp draw stays aligned.
    - Both backends then pass gd=0.0 + group_depth_fixed=True to the MCMC.

    Additionally: every sample must report group_depth == 0.0 exactly in
    both backends.
    """

    @pytest.fixture(scope="class")
    def python_result(self):
        import community_hierarchy as ch
        G = _build_nx_graph()
        return ch.fit(
            ch.Config(groups_model="general_ASBM", interaction="independent"), G,
            num_samples=10, seed=SEED,
            num_warmup=50, sweeps_per_sample=1,
            merge_split_enabled=True,
        )

    @pytest.fixture(scope="class")
    def wasm_result(self):
        return wasm_fit(WINS, TIES, {
            "num_nodes": N,
            "seed": SEED,
            "num_warmup": 50,
            "num_samples": 10,
            "sweeps_per_sample": 1,
            "assortative": True,
            "variation_in": -1,     # inferred: general_ASBM
            "variation_out": -1,
            "merge_split_enabled": True,
            "independent_coupling": 1,
        })

    def test_python_group_depth_is_zero(self, python_result):
        df = python_result.to_frame()
        assert "group_depth" in df.columns, "samples_df() must expose group_depth"
        for i, gd in enumerate(df["group_depth"]):
            assert gd == pytest.approx(0.0, abs=1e-12), \
                f"Python sample {i}: group_depth={gd}, expected 0.0"

    def test_wasm_group_depth_is_zero(self, wasm_result):
        for i, s in enumerate(wasm_result["samples"]):
            gd = s.get("groupDepth", None)
            assert gd is not None, f"sample {i} missing groupDepth field"
            assert gd == pytest.approx(0.0, abs=1e-12), \
                f"WASM sample {i}: groupDepth={gd}, expected 0.0"

    def test_wasm_mdl_matches_python_exactly(self, python_result, wasm_result):
        """WASM independent-mode MDL must equal Python's to numerical precision."""
        py_mdl = python_result.mdl_value
        wasm_mdl = wasm_result["mdlValue"]
        assert math.isfinite(wasm_mdl), f"WASM MDL is not finite: {wasm_mdl}"
        assert wasm_mdl == pytest.approx(py_mdl, rel=0, abs=1e-9), (
            f"WASM={wasm_mdl!r} != Python={py_mdl!r} "
            f"(diff={wasm_mdl - py_mdl:.3e})"
        )

    def test_wasm_consensus_matches_python(self, python_result, wasm_result):
        py = [int(x) for x in python_result.consensus_partition()]
        raw = wasm_result["consensusPartition"]
        wasm = [int(raw[str(i)] if isinstance(raw, dict) else raw[i]) for i in range(len(py))]
        assert len(wasm) == len(py)
        def canon(p):
            m, out = {}, []
            for x in p:
                m.setdefault(x, len(m))
                out.append(m[x])
            return out
        assert canon(wasm) == canon(py), (
            f"consensus differs.\n  python: {canon(py)}\n  wasm  : {canon(wasm)}"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_nx_graph():
    """
    Build a nx.DiGraph from the module-level WINS / TIES test network.
    Each tie pair is added in ONE direction only — adding both (u,v) and (v,u)
    as 'neutral' edges causes Python's undirected ties graph to accumulate
    weight=2 per pair, corrupting H_s_G_b_depths.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for u, v, w in WINS:
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w, type="dominant")
    for u, v, w in TIES:
        G.add_edge(u, v, weight=w, type="neutral")
        # Do NOT add (v, u) — Python builds G_ties as undirected from these edges
    return G


# ---------------------------------------------------------------------------
# 5. Model presets: fixed mixing variations
# ---------------------------------------------------------------------------
# Python's groups_model presets fix the mixing variations (and traditional_SBM
# turns assortativity off); the wasm binding takes the same fixed values
# (variation_in / variation_out, negative = inferred). Same seed, same MDL.
PRESETS = {
    "simple_ASBM":       dict(assortative=True,  variation_in=0.5, variation_out=0.5),
    "hybrid_ASBM":       dict(assortative=True,  variation_in=0.5, variation_out=0.0),
    "planted_partition": dict(assortative=True,  variation_in=0.0, variation_out=0.0),
    "traditional_SBM":   dict(assortative=False, variation_in=0.5, variation_out=0.5),
}


@pytest.mark.parametrize("model", sorted(PRESETS))
def test_wasm_preset_mdl_matches_python(model):
    import community_hierarchy as ch
    py = ch.fit(ch.Config(groups_model=model), _build_nx_graph(), num_samples=10, seed=SEED,
                num_warmup=50, sweeps_per_sample=1, merge_split_enabled=True)
    wasm = wasm_fit(WINS, TIES, {
        "num_nodes": N, "seed": SEED, "num_warmup": 50, "num_samples": 10,
        "sweeps_per_sample": 1, "merge_split_enabled": True, **PRESETS[model],
    })
    assert math.isfinite(wasm["mdlValue"])
    assert wasm["mdlValue"] == pytest.approx(py.mdl_value, rel=0, abs=1e-9), (
        f"{model}: WASM={wasm['mdlValue']!r} != Python={py.mdl_value!r}")
    for s in wasm["samples"]:
        assert s["variationIn"] == pytest.approx(PRESETS[model]["variation_in"], abs=1e-12)
        assert s["variationOut"] == pytest.approx(PRESETS[model]["variation_out"], abs=1e-12)
