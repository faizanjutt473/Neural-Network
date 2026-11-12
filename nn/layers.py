"""Fully connected (dense) layer."""

import numpy as np


class Dense:
    """Affine layer:  Z = X @ W + b.

    Shapes: X (N, n_in), W (n_in, n_out), b (n_out,), Z (N, n_out).

    Backward (given dL/dZ):
        dW = X^T @ dL/dZ
        db = sum over batch of dL/dZ
        dX = dL/dZ @ W^T
    """

    def __init__(self, n_in, n_out, init="he", rng=None, dtype=np.float32):
        rng = rng if rng is not None else np.random.default_rng()
        if init == "he":          # variance 2 / n_in  -> suited to ReLU
            std = np.sqrt(2.0 / n_in)
        elif init == "xavier":    # variance 2 / (n_in + n_out) -> suited to tanh/sigmoid/softmax
            std = np.sqrt(2.0 / (n_in + n_out))
        else:
            raise ValueError("init must be 'he' or 'xavier'")

        self.W = (rng.standard_normal((n_in, n_out)) * std).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x  # cache input for backward
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def parameters(self):
        """Return [(parameter, gradient), ...]; optimizers update them in place."""
        return [(self.W, self.dW), (self.b, self.db)]
