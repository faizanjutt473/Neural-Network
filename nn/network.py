"""Feedforward neural network: model container + mini-batch training loop."""

import time
import numpy as np

from .activations import get_activation, softmax
from .layers import Dense
from .utils import minibatch_indices


class NeuralNetwork:
    """Stack of Dense layers with a chosen hidden activation.

    ``layer_sizes=[784, 128, 64, 10]`` builds
        Dense(784,128) -> act -> Dense(128,64) -> act -> Dense(64,10)
    The final layer outputs raw logits; softmax is applied inside the loss
    (training) or ``predict_proba`` (inference).
    """

    def __init__(self, layer_sizes, activation="relu", seed=42, dtype=np.float32):
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        rng = np.random.default_rng(seed)

        self.layers = []
        n_layers = len(layer_sizes) - 1
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            is_last = i == n_layers - 1
            init = "he" if (activation == "relu" and not is_last) else "xavier"
            self.layers.append(Dense(n_in, n_out, init=init, rng=rng, dtype=dtype))
            if not is_last:
                self.layers.append(get_activation(activation))

    # ----- core passes ---------------------------------------------------
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # ----- parameters ----------------------------------------------------
    @property
    def dense_layers(self):
        return [l for l in self.layers if isinstance(l, Dense)]

    def parameters(self):
        return [pg for layer in self.dense_layers for pg in layer.parameters()]

    def save_weights(self, path):
        arrays = {}
        for i, layer in enumerate(self.dense_layers):
            arrays[f"W{i}"], arrays[f"b{i}"] = layer.W, layer.b
        np.savez(path, **arrays)

    def load_weights(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.dense_layers):
            layer.W[...] = data[f"W{i}"]
            layer.b[...] = data[f"b{i}"]

    # ----- inference -----------------------------------------------------
    def predict_proba(self, x, batch_size=5000):
        return np.vstack([softmax(self.forward(x[i:i + batch_size]))
                          for i in range(0, len(x), batch_size)])

    def predict(self, x, batch_size=5000):
        return self.predict_proba(x, batch_size).argmax(axis=1)

    def evaluate(self, x, y_onehot, loss_fn, batch_size=5000):
        """Return (mean loss, accuracy) without updating any weights."""
        total_loss, correct = 0.0, 0
        for i in range(0, len(x), batch_size):
            xb, yb = x[i:i + batch_size], y_onehot[i:i + batch_size]
            logits = self.forward(xb)
            total_loss += loss_fn.forward(logits, yb) * len(xb)
            correct += int((logits.argmax(1) == yb.argmax(1)).sum())
        return total_loss / len(x), correct / len(x)

    # ----- training ------------------------------------------------------
    def fit(self, x_train, y_train, x_val, y_val, optimizer, loss_fn,
            epochs=20, batch_size=64, seed=42, verbose=True):
        """Mini-batch gradient descent. Labels must be one-hot.

        Per-epoch train loss/accuracy are running averages over the batches
        (the same definition is used in the PyTorch script).
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [],
                   "val_acc": [], "epoch_time": []}
        n = len(x_train)

        for epoch in range(epochs):
            t0 = time.perf_counter()
            running_loss, correct = 0.0, 0

            for idx in minibatch_indices(n, batch_size, seed + epoch):
                xb, yb = x_train[idx], y_train[idx]
                logits = self.forward(xb)                 # 1. forward
                loss = loss_fn.forward(logits, yb)        # 2. loss
                self.backward(loss_fn.backward())         # 3. backprop
                optimizer.step(self.parameters())         # 4. update

                running_loss += loss * len(idx)
                correct += int((logits.argmax(1) == yb.argmax(1)).sum())

            val_loss, val_acc = self.evaluate(x_val, y_val, loss_fn)
            history["train_loss"].append(running_loss / n)
            history["train_acc"].append(correct / n)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epoch_time"].append(time.perf_counter() - t0)

            if verbose:
                print(f"Epoch {epoch + 1:02d}/{epochs} | "
                      f"train loss {history['train_loss'][-1]:.4f} acc {history['train_acc'][-1]:.4f} | "
                      f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                      f"{history['epoch_time'][-1]:.1f}s")
        return history
