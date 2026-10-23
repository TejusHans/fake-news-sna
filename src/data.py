"""Data loading and input-masking utilities for UPFD (PolitiFact, GossipCop)."""
import networkx as nx
import torch
from torch_geometric.datasets import UPFD

SOURCES = ("politifact", "gossipcop")
SPLITS = ("train", "val", "test")
MODES = ("full", "root_only", "users_only", "structure_only")


def load_split(name, split, root):
    """Load one official UPFD split with BERT node features."""
    return UPFD(root=root, name=name, feature="bert", split=split)


def load_all(root):
    """Return {source: {split: dataset}} for every source and split."""
    return {n: {s: load_split(n, s, root) for s in SPLITS} for n in SOURCES}


def feature_dim(mode):
    """Node feature size the model must accept for a given input mode."""
    return 3 if mode == "structure_only" else 768


def _structural_x(g):
    """Replace node features with [is_root, log(1 + out-degree), depth]."""
    n = g.num_nodes
    ei = g.edge_index.numpy()
    D = nx.DiGraph()
    D.add_nodes_from(range(n))
    D.add_edges_from(zip(ei[0], ei[1]))
    depth = nx.single_source_shortest_path_length(D, 0)
    is_root = torch.zeros(n)
    is_root[0] = 1
    outdeg = torch.tensor([D.out_degree(i) for i in range(n)], dtype=torch.float)
    dep = torch.tensor([depth.get(i, 0) for i in range(n)], dtype=torch.float)
    return torch.stack([is_root, torch.log1p(outdeg), dep], dim=1)


def apply_mode(ds, mode):
    """Return a list of graphs with the node inputs restricted by `mode`.

    full            news node + user nodes (unchanged)
    root_only       user-node features zeroed; news node (index 0) kept
    users_only      news-node features zeroed; user nodes kept
    structure_only  features replaced by tree-shape features only
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    out = []
    for g in ds:
        g = g.clone()
        if mode == "root_only":
            g.x[1:] = 0
        elif mode == "users_only":
            g.x[0] = 0
        elif mode == "structure_only":
            g.x = _structural_x(g)
        out.append(g)
    return out
