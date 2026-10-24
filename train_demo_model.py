"""Train one GraphSAGE model (full input) on a single source and save its weights.

Usage:
    python train_demo_model.py --root data/UPFD --source gossipcop
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.data import load_split
from src.models import SAGENet
from src.train import evaluate, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/UPFD")
    ap.add_argument("--source", choices=["politifact", "gossipcop"], default="gossipcop")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    device = get_device()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tr = load_split(a.source, "train", a.root)
    va = load_split(a.source, "val", a.root)
    te = load_split(a.source, "test", a.root)
    bs = 32 if a.source == "politifact" else 128
    train_loader = DataLoader(tr, batch_size=bs, shuffle=True)
    val_loader = DataLoader(va, batch_size=256)
    test_loader = DataLoader(te, batch_size=256)

    model = SAGENet(768).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

    best_val, best_state = -1.0, None
    for _ in range(a.epochs):
        model.train()
        for d in train_loader:
            d = d.to(device)
            opt.zero_grad()
            F.cross_entropy(model(d.x, d.edge_index, d.batch), d.y).backward()
            opt.step()
        val_acc, _ = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.source}_full.pt")
    torch.save(best_state, path)
    print(f"saved {path} | best val acc {best_val:.4f} | test acc {test_acc:.4f} | test f1 {test_f1:.4f}")


if __name__ == "__main__":
    main()
