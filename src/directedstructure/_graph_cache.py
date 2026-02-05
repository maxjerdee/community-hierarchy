# Graph caching utilities to avoid circular imports

import networkx as nx
import threading
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

# Global graph cache: maps graph-hash -> graph object
_wins_graph_cache: dict[str, nx.DiGraph] = {}
_ties_graph_cache: dict[str, nx.Graph] = {}
_graph_cache_lock = threading.RLock()

def _store_wins_graph(G: nx.DiGraph) -> str:
    """Store a graph in the cache and return its hash key."""
    hash_key = weisfeiler_lehman_graph_hash(G)
    with _graph_cache_lock:
        _wins_graph_cache[hash_key] = G
    return hash_key

def _get_wins_graph(hash_key: str) -> nx.DiGraph | None:
    """Return cached graph (or None)."""
    with _graph_cache_lock:
        return _wins_graph_cache.get(hash_key)
    
def _store_ties_graph(G: nx.Graph) -> str:
    """Store a graph in the cache and return its hash key."""
    hash_key = weisfeiler_lehman_graph_hash(G)
    with _graph_cache_lock:
        _ties_graph_cache[hash_key] = G
    return hash_key

def _get_ties_graph(hash_key: str) -> nx.Graph | None:
    """Return cached graph (or None)."""
    with _graph_cache_lock:
        return _ties_graph_cache.get(hash_key)