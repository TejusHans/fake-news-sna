"""Reproduce the GNN experiments from the repo root.

Usage:
    python run_experiments.py --root data/UPFD --experiment modes
    python run_experiments.py --root data/UPFD --experiment cross_domain
    python run_experiments.py --root data/UPFD --experiment all

Output CSVs are written to results/ with a `repro_` prefix, so the original
result files are never overwritten and the two can be compared.
"""
import argparse
import os

import pandas as pd

from src.data import SOURCES, apply_mode, feature_dim, load_all
from src.train import fit


def model_kwargs(mode):
    """Model settings used in the original experiments."""
    if mode == "structure_only":
        return dict(hid=64, dropout=0.3)
    return dict(hid=128, dropout=0.5)


def run_modes(root, out, seeds):
    """Per-source GNN with each input mode (full / root_only / users_only / structure_only)."""
    data = load_all(root)
    rows = []
    for name in SOURCES:
        bs = 32 if name == "politifact" else 128
        for mode in ["full", "root_only", "users_only", "structure_only"]:
            tr = apply_mode(data[name]["train"], mode)
            va = apply_mode(data[name]["val"], mode)
            te = apply_mode(data[name]["test"], mode)
            for seed in range(seeds):
                res = fit(tr, va, {name: te}, feature_dim(mode), seed,
                          epochs=60, batch_size=bs, **model_kwargs(mode))
                acc, f1 = res["test"][name]
                rows.append((name, mode, seed, round(res["val_acc"], 4),
                             round(acc, 4), round(f1, 4)))
                print(f"{name:11s} {mode:14s} seed={seed} val={res['val_acc']:.4f} test_acc={acc:.4f}")
    df = pd.DataFrame(rows, columns=["dataset", "mode", "seed", "val_acc", "test_acc", "test_f1"])
    df.to_csv(os.path.join(out, "repro_gnn_modes.csv"), index=False)
    print(df.groupby(["dataset", "mode"])[["test_acc", "test_f1"]].agg(["mean", "std"]).round(4))


def run_cross_domain(root, out, seeds):
    """Train on PolitiFact / GossipCop / merged; test on both sources."""
    data = load_all(root)
    setups = {"politifact": ["politifact"], "gossipcop": ["gossipcop"],
              "merged": ["politifact", "gossipcop"]}
    rows = []
    for mode in ["full", "root_only"]:
        d = {n: {s: apply_mode(data[n][s], mode) for s in data[n]} for n in data}
        tests = {n: d[n]["test"] for n in d}
        for train_on, srcs in setups.items():
            tr = sum([d[n]["train"] for n in srcs], [])
            va = sum([d[n]["val"] for n in srcs], [])
            bs = 32 if train_on == "politifact" else 128
            for seed in range(seeds):
                res = fit(tr, va, tests, feature_dim(mode), seed,
                          epochs=40, batch_size=bs, **model_kwargs(mode))
                for n, (acc, f1) in res["test"].items():
                    rows.append((mode, train_on, n, seed, round(res["val_acc"], 4),
                                 round(acc, 4), round(f1, 4)))
                t = res["test"]
                print(f"{mode:9s} train={train_on:10s} seed={seed} "
                      f"pf={t['politifact'][0]:.4f} gc={t['gossipcop'][0]:.4f}")
    df = pd.DataFrame(rows, columns=["mode", "train_on", "test_on", "seed",
                                     "val_acc", "test_acc", "test_f1"])
    df.to_csv(os.path.join(out, "repro_cross_domain.csv"), index=False)
    print(df.groupby(["mode", "train_on", "test_on"])[["test_acc", "test_f1"]]
          .agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/UPFD")
    ap.add_argument("--out", default="results")
    ap.add_argument("--experiment", choices=["modes", "cross_domain", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.experiment in ("modes", "all"):
        run_modes(a.root, a.out, a.seeds)
    if a.experiment in ("cross_domain", "all"):
        run_cross_domain(a.root, a.out, a.seeds)
