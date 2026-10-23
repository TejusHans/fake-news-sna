# Fake News Detection Using Social Network Analysis

Fake news detection on the UPFD version of FakeNewsNet (PolitiFact and GossipCop), comparing text-only, network-only, and combined models, plus a GraphSAGE GNN over news propagation trees. The project includes feature-group ablations, input ablations, and a cross-domain experiment.

## Dataset

UPFD, loaded through PyTorch Geometric with BERT node features and the official 20/10/70 train/val/test split. Each graph is a propagation tree: the news node (index 0, the root) plus the users who shared it.

| Source | Graphs | Real | Fake | Avg. nodes | Max nodes |
|---|---|---|---|---|---|
| PolitiFact | 314 | 157 | 157 | 130.7 | 497 |
| GossipCop | 5464 | 2732 | 2732 | 57.5 | 199 |

Classes are balanced, so accuracy is a fair metric. PolitiFact trains on only 62 graphs, so its results are noisy.

## Methods

- **Text-only:** BERT embedding of the news node, with Logistic Regression and Random Forest.
- **Network-only:** 16 hand-built features per cascade (tree structure, centrality, PageRank, Louvain community statistics), with the same classifiers.
- **Combined:** text embedding plus the network features.
- **GNN:** 2-layer GraphSAGE with mean+max pooling over all nodes.
- **GNN input ablations:** `full`, `root_only` (user-node features zeroed), `users_only` (news-node features zeroed), `structure_only` (features replaced by is-root, log out-degree, depth).
- **Cross-domain:** train on PolitiFact, GossipCop, or both merged; test on each source.

GNN model selection uses validation accuracy only. Test accuracy is reported at the best-validation epoch.

## Results

All numbers are test accuracy.

### Classical models (single run, seed 42; LogReg / RandomForest)

| Features | PolitiFact | GossipCop |
|---|---|---|
| Text-only | 0.801 / 0.787 | 0.737 / 0.726 |
| Cascade size only | 0.489 / 0.507 | 0.730 / 0.704 |
| Network-only | 0.552 / 0.529 | 0.822 / 0.805 |
| Text + network | 0.828 / 0.765 | 0.809 / 0.840 |

### GraphSAGE input ablation (mean ± std over 3 seeds)

| GNN input | PolitiFact | GossipCop |
|---|---|---|
| Full | 0.821 ± 0.021 | 0.951 ± 0.006 |
| News node only | 0.828 ± 0.008 | 0.825 ± 0.033 |
| User nodes only | 0.759 ± 0.007 | 0.971 ± 0.002 |
| Tree structure only | 0.535 ± 0.021 | 0.801 ± 0.005 |

### Cross-domain (GNN, mean over 3 seeds; full input / news node only)

| Train on | Test on PolitiFact | Test on GossipCop |
|---|---|---|
| PolitiFact | 0.827 / 0.825 | 0.575 / 0.544 |
| GossipCop | 0.496 / 0.528 | 0.952 / 0.818 |
| Merged | 0.576 / 0.614 | 0.943 / 0.831 |

## Key findings

1. **GossipCop:** cascade structure alone reaches about 80% accuracy. Two different models agree (a GNN on three structural features, and a Random Forest on 16 hand-built features). Cascade size alone gets about 73%, so the structure features add signal beyond size.
2. **GossipCop:** the GNN's 95% comes mainly from the user-node features, not the news text. Using only user nodes gives 97%, and using only the news node gives 82%.
3. **PolitiFact:** the news-node embedding carries the signal. Network and structure features are at chance, and the GNN gives no measurable gain over text. Small training data (62 graphs) limits what can be concluded.
4. **No transfer between domains:** models trained on one source score 50 to 57% on the other, and naive merging does not help.

## Limitations

- PolitiFact's training set has 62 graphs and its validation set 31, so its numbers vary a lot and small differences are not meaningful.
- Seed standard deviations reflect model randomness only, not variation across data splits.
- The classical-model table is a single run with one seed.
- UPFD's graphs are precomputed trees that do not expose user IDs, so users cannot be linked across news items. Community detection runs inside each cascade, where trees have little community structure, so those features add little.
- In the merged-training experiment, epochs were selected on a validation set that is about 95% GossipCop. This may contribute to the PolitiFact drop. Selecting epochs per domain is untested.
- The classical-model results and the cross-domain results currently come from notebooks, not from the script below.

## Reproduce

```bash
git clone https://github.com/TejusHans/fake-news-sna.git
cd fake-news-sna
pip install torch torch_geometric scikit-learn pandas numpy networkx
python run_experiments.py --root data/UPFD --experiment modes
python run_experiments.py --root data/UPFD --experiment cross_domain
```

UPFD downloads automatically on first run. A GPU is recommended. Outputs are written to `results/` as `repro_*.csv`. The GNN input-ablation table above was reproduced from this script to within about 0.3 points.

## Repository layout

```
src/               data loading and masking, model, training loop
run_experiments.py reproduces the GNN experiments
notebooks/         original exploratory notebook
results/           result CSVs for every table above
```

## Status

Work in progress: a research paper and a demo app are planned.
