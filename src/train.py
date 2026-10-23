"""Training and evaluation loop shared by every GNN experiment."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader

from src.models import SAGENet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (accuracy, F1) of `model` on every graph in `loader`."""
    model.eval()
    preds, ys = [], []
    for d in loader:
        d = d.to(device)
        preds.append(model(d.x, d.edge_index, d.batch).argmax(1).cpu())
        ys.append(d.y.cpu())
    p = torch.cat(preds).numpy()
    y = torch.cat(ys).numpy()
    return accuracy_score(y, p), f1_score(y, p)


def fit(train_ds, val_ds, test_sets, in_dim, seed, epochs=60, batch_size=128,
        hid=128, dropout=0.5, lr=0.005, weight_decay=1e-3, device=None):
    """Train one GraphSAGE model and report test results at the best-validation epoch.

    The epoch is chosen on validation accuracy only; test sets are scored
    just when validation improves, so test data never drives model selection.

    test_sets: {name: list_of_graphs}.
    Returns {"val_acc": float, "test": {name: (accuracy, f1)}}.
    """
    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loaders = {n: DataLoader(ds, batch_size=256) for n, ds in test_sets.items()}

    model = SAGENet(in_dim, hid=hid, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_test = -1.0, None
    for _ in range(epochs):
        model.train()
        for d in train_loader:
            d = d.to(device)
            opt.zero_grad()
            F.cross_entropy(model(d.x, d.edge_index, d.batch), d.y).backward()
            opt.step()
        val_acc, _ = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_test = {n: evaluate(model, ld, device) for n, ld in test_loaders.items()}
    return {"val_acc": best_val, "test": best_test}
