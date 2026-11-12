"""Small NumPy-only helpers shared by training scripts."""

import numpy as np


def one_hot(labels, num_classes, dtype=np.float32):
    out = np.zeros((labels.shape[0], num_classes), dtype=dtype)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def minibatch_indices(n_samples, batch_size, seed):
    """Yield shuffled index arrays. Same seed -> same order (used for fair
    comparison with the PyTorch model)."""
    order = np.random.default_rng(seed).permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield order[start:start + batch_size]
