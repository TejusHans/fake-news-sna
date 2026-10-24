"""Streamlit demo: browse UPFD GossipCop test cascades and see the GNN's prediction.

Run from the repo root:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
import torch

from src.data import load_split
from src.models import SAGENet

MODEL_PATH = "models/gossipcop_full.pt"
# Label convention assumed from the UPFD data (0 = real, 1 = fake); verify against the UPFD paper.
LABELS = {0: "Real", 1: "Fake"}

st.set_page_config(page_title="Fake News Cascade Explorer", layout="wide")
st.title("Fake News Cascade Explorer")
st.caption(
    "GraphSAGE trained on the UPFD GossipCop training split (single seed, test accuracy 0.955 "
    "when saved). The 3-seed mean reported in the README is 0.951. "
    "This is a research demo, not a fact-checking tool."
)

root = st.sidebar.text_input("UPFD data folder", "data/UPFD")


@st.cache_resource
def get_data(root):
    return load_split("gossipcop", "test", root)


@st.cache_resource
def get_model():
    model = SAGENet(768)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def tree_stats(g):
    n = g.num_nodes
    ei = g.edge_index.numpy()
    D = nx.DiGraph()
    D.add_nodes_from(range(n))
    D.add_edges_from(zip(ei[0], ei[1]))
    depth = nx.single_source_shortest_path_length(D, 0)
    levels = np.bincount(list(depth.values()))
    outdeg = np.array([D.out_degree(i) for i in range(n)])
    stats = {
        "Nodes": n,
        "Max depth": int(max(depth.values())),
        "Widest level": int(levels.max()),
        "Root out-degree": int(outdeg[0]),
        "Leaf fraction": round(float((outdeg == 0).mean()), 2),
    }
    return D, depth, stats


def draw_tree(D, depth):
    for node in D.nodes:
        D.nodes[node]["layer"] = depth.get(node, 0)
    pos = nx.multipartite_layout(D, subset_key="layer", align="horizontal")
    pos = {k: (x, -y) for k, (x, y) in pos.items()}
    colors = ["#d62728" if n == 0 else "#7f9fc7" for n in D.nodes]
    sizes = [220 if n == 0 else 40 for n in D.nodes]
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_edges(D, pos, ax=ax, arrows=False, alpha=0.4)
    nx.draw_networkx_nodes(D, pos, ax=ax, node_color=colors, node_size=sizes)
    ax.set_title("Propagation tree (red = news node, blue = users who shared it)")
    ax.axis("off")
    return fig


try:
    ds = get_data(root)
    model = get_model()
except Exception as e:
    st.error(f"Could not load data or model: {e}")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx = 0

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Random"):
    st.session_state.idx = int(np.random.randint(len(ds)))
idx = st.sidebar.number_input("Test graph index", 0, len(ds) - 1, key="idx")

g = ds[int(idx)]
batch = torch.zeros(g.num_nodes, dtype=torch.long)
with torch.no_grad():
    probs = torch.softmax(model(g.x, g.edge_index, batch), dim=1)[0]
pred = int(probs.argmax())
true = int(g.y)

c1, c2, c3 = st.columns(3)
c1.metric("True label", LABELS[true])
c2.metric("Model prediction", LABELS[pred], f"{float(probs[pred]) * 100:.1f}% confidence")
c3.metric("Result", "Correct" if pred == true else "Wrong")

D, depth, stats = tree_stats(g)
left, right = st.columns([2, 1])
with left:
    st.pyplot(draw_tree(D, depth))
with right:
    st.subheader("Cascade structure")
    for k, v in stats.items():
        st.write(f"**{k}:** {v}")
    st.caption(
        "Earlier experiments in this repo found that structure alone separates fake from real "
        "on GossipCop at about 80%; the user-node features carry the rest of the signal."
    )
