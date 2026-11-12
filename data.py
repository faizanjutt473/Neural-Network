"""MNIST loading and splitting (data/mnist.npz created by download_data.py)."""

import numpy as np

from config import DATA_PATH, NUM_CLASSES, SEED, VAL_FRACTION
from nn.utils import one_hot


def load_mnist(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python download_data.py` first.")
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


def train_val_split(x, y, val_fraction=VAL_FRACTION, seed=SEED):
    perm = np.random.default_rng(seed).permutation(len(x))
    n_val = int(len(x) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def get_splits():
    """Return float32 train/val/test sets: (X, y_int) for each."""
    x_train, y_train, x_test, y_test = load_mnist()
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train)
    return (x_tr.astype(np.float32), y_tr), (x_val.astype(np.float32), y_val), \
           (x_test.astype(np.float32), y_test)


def get_onehot_splits():
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = get_splits()
    return (x_tr, one_hot(y_tr, NUM_CLASSES)), (x_val, one_hot(y_val, NUM_CLASSES)), \
           (x_te, one_hot(y_te, NUM_CLASSES))
