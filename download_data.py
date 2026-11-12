"""Download MNIST once and store it as data/mnist.npz (pixels scaled to 0-1)."""

from pathlib import Path

import numpy as np
from torchvision import datasets

Path("data").mkdir(exist_ok=True)
train = datasets.MNIST("data", train=True, download=True)
test = datasets.MNIST("data", train=False, download=True)

np.savez_compressed(
    "data/mnist.npz",
    X_train=train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0,
    y_train=train.targets.numpy(),
    X_test=test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0,
    y_test=test.targets.numpy(),
)
print("Saved data/mnist.npz")
