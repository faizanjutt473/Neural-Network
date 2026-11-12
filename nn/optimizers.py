"""Optimizers."""

import numpy as np


class SGD:
    """Stochastic gradient descent with (classical) momentum.

        v <- beta * v - lr * grad
        w <- w + v

    ``momentum=0`` gives plain SGD. This is mathematically equivalent to
    PyTorch's ``torch.optim.SGD(lr, momentum=beta)`` (which stores v' = -v/lr).
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = None

    def step(self, params_and_grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p, _ in params_and_grads]
        for v, (p, g) in zip(self.velocities, params_and_grads):
            v *= self.momentum
            v -= self.lr * g
            p += v  # in-place update so layers see the new weights
