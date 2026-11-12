"""Train the NumPy neural network on MNIST."""

import json
import time

from config import (ACTIVATION, BATCH_SIZE, EPOCHS, LAYER_SIZES, LEARNING_RATE,
                    MOMENTUM, RESULTS_DIR, SEED)
from data import get_onehot_splits
from nn import NeuralNetwork, SGD, SoftmaxCrossEntropy
from plotting import plot_training_curves


def main():
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = get_onehot_splits()
    print(f"Train {x_tr.shape} | Val {x_val.shape} | Test {x_te.shape}")

    net = NeuralNetwork(LAYER_SIZES, activation=ACTIVATION, seed=SEED)
    net.save_weights(RESULTS_DIR / "init_weights.npz")   # PyTorch reuses these
    optimizer = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_fn = SoftmaxCrossEntropy()

    t0 = time.perf_counter()
    history = net.fit(x_tr, y_tr, x_val, y_val, optimizer, loss_fn,
                      epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED)
    history["train_time"] = time.perf_counter() - t0

    history["test_loss"], history["test_acc"] = net.evaluate(x_te, y_te, loss_fn)
    print(f"\nTest loss {history['test_loss']:.4f} | Test accuracy {history['test_acc']:.4f} "
          f"| Training time {history['train_time']:.1f}s")

    (RESULTS_DIR / "numpy_history.json").write_text(json.dumps(history, indent=2))
    net.save_weights(RESULTS_DIR / "numpy_trained_weights.npz")
    plot_training_curves(history, RESULTS_DIR / "numpy_training_curves.png",
                         title="NumPy Neural Network (784-128-64-10, SGD + momentum)")
    print("Saved history, weights and curves to results/")


if __name__ == "__main__":
    main()
