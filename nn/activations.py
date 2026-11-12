"""Activation functions with explicit forward and backward passes.

Every activation caches whatever it needs during ``forward`` so that
``backward`` can apply the chain rule:  dL/dz = dL/da * da/dz.
"""

import numpy as np


def softmax(z):
    """Numerically stable row-wise softmax."""
    shifted = z - z.max(axis=1, keepdims=True)  # avoids overflow in exp
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


class Activation:
    """Base class: forward(z) -> a, backward(dL/da) -> dL/dz."""

    def forward(self, z):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    def __call__(self, z):
        return self.forward(z)


class ReLU(Activation):
    """a = max(0, z),  da/dz = 1 if z > 0 else 0."""

    def forward(self, z):
        self.mask = z > 0
        return z * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class Sigmoid(Activation):
    """a = 1 / (1 + exp(-z)),  da/dz = a * (1 - a)."""

    def forward(self, z):
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))   # stable for large positive z
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)                # stable for large negative z
        self.out = out
        return out

    def backward(self, grad_out):
        return grad_out * self.out * (1.0 - self.out)


class Tanh(Activation):
    """a = tanh(z),  da/dz = 1 - a^2."""

    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad_out):
        return grad_out * (1.0 - self.out ** 2)


class Softmax(Activation):
    """Row-wise softmax with the full Jacobian-vector product.

    For s = softmax(z):  dL/dz_i = s_i * (g_i - sum_j g_j * s_j)  where g = dL/ds.
    During training the network uses ``SoftmaxCrossEntropy`` (which fuses
    softmax + loss for a simpler and more stable gradient), but this class is
    provided so softmax is a complete, differentiable activation on its own.
    """

    def forward(self, z):
        self.out = softmax(z)
        return self.out

    def backward(self, grad_out):
        dot = np.sum(grad_out * self.out, axis=1, keepdims=True)
        return self.out * (grad_out - dot)


_ACTIVATIONS = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh, "softmax": Softmax}


def get_activation(name):
    """Create a fresh activation instance from its name."""
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
