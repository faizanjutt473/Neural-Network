"""PyTorch equivalent of train_numpy.py (for comparison only).

Same architecture, hyperparameters, initial weights (loaded from the NumPy
model) and the same mini-batch order, so differences come only from the
framework's numerics.
"""

import json
import time

import numpy as np
import torch
import torch.nn as tnn

from config import (BATCH_SIZE, EPOCHS, LAYER_SIZES, LEARNING_RATE, MOMENTUM,
                    RESULTS_DIR, SEED)
from data import get_splits
from nn.utils import minibatch_indices
from plotting import plot_training_curves


def build_model():
    layers = []
    for i, (n_in, n_out) in enumerate(zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])):
        layers.append(tnn.Linear(n_in, n_out))
        if i < len(LAYER_SIZES) - 2:
            layers.append(tnn.ReLU())
    return tnn.Sequential(*layers)


def load_numpy_init(model, path):
    """Copy the NumPy model's initial weights (W is (in,out) -> transpose)."""
    data = np.load(path)
    linears = [m for m in model if isinstance(m, tnn.Linear)]
    with torch.no_grad():
        for i, lin in enumerate(linears):
            lin.weight.copy_(torch.from_numpy(data[f"W{i}"].T.copy()))
            lin.bias.copy_(torch.from_numpy(data[f"b{i}"].copy()))


@torch.no_grad()
def evaluate(model, x, y, loss_fn, batch_size=5000):
    model.eval()
    total, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        out = model(x[i:i + batch_size])
        total += loss_fn(out, y[i:i + batch_size]).item() * len(out)
        correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    return total / len(x), correct / len(x)


def main():
    init_path = RESULTS_DIR / "init_weights.npz"
    if not init_path.exists():
        raise FileNotFoundError("Run train_numpy.py first (it saves the shared initial weights).")

    torch.manual_seed(SEED)
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = get_splits()
    x_tr, x_val, x_te = map(torch.from_numpy, (x_tr, x_val, x_te))
    y_tr, y_val, y_te = (torch.from_numpy(y.astype(np.int64)) for y in (y_tr, y_val, y_te))

    model = build_model()
    load_numpy_init(model, init_path)
    loss_fn = tnn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    n = len(x_tr)
    t_start = time.perf_counter()

    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        model.train()
        running_loss, correct = 0.0, 0
        for idx in minibatch_indices(n, BATCH_SIZE, SEED + epoch):
            idx = torch.from_numpy(idx)
            xb, yb = x_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()

        val_loss, val_acc = evaluate(model, x_val, y_val, loss_fn)
        history["train_loss"].append(running_loss / n)
        history["train_acc"].append(correct / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(time.perf_counter() - t0)
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | train loss {history['train_loss'][-1]:.4f} "
              f"acc {history['train_acc'][-1]:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} "
              f"| {history['epoch_time'][-1]:.1f}s")

    history["train_time"] = time.perf_counter() - t_start
    history["test_loss"], history["test_acc"] = evaluate(model, x_te, y_te, loss_fn)
    print(f"\nTest loss {history['test_loss']:.4f} | Test accuracy {history['test_acc']:.4f} "
          f"| Training time {history['train_time']:.1f}s")

    (RESULTS_DIR / "pytorch_history.json").write_text(json.dumps(history, indent=2))
    plot_training_curves(history, RESULTS_DIR / "pytorch_training_curves.png",
                         title="PyTorch Equivalent (784-128-64-10, SGD + momentum)")


if __name__ == "__main__":
    main()
