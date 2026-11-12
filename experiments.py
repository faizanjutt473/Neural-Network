"""Optional ablations: momentum vs plain SGD, and ReLU vs Tanh vs Sigmoid."""

import json

from config import BATCH_SIZE, LAYER_SIZES, LEARNING_RATE, MOMENTUM, RESULTS_DIR, SEED
from data import get_onehot_splits
from nn import NeuralNetwork, SGD, SoftmaxCrossEntropy
from plotting import plot_comparison

EPOCHS = 10
CONFIGS = {
    "ReLU + momentum 0.9": ("relu", MOMENTUM),
    "ReLU + plain SGD": ("relu", 0.0),
    "Tanh + momentum 0.9": ("tanh", MOMENTUM),
    "Sigmoid + momentum 0.9": ("sigmoid", MOMENTUM),
}


def main():
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = get_onehot_splits()
    histories, lines = {}, ["| Config | Val acc | Test acc | Test loss |", "|---|---|---|---|"]

    for label, (act, beta) in CONFIGS.items():
        print(f"\n=== {label} ===")
        net = NeuralNetwork(LAYER_SIZES, activation=act, seed=SEED)
        loss_fn = SoftmaxCrossEntropy()
        h = net.fit(x_tr, y_tr, x_val, y_val, SGD(LEARNING_RATE, beta), loss_fn,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED)
        test_loss, test_acc = net.evaluate(x_te, y_te, loss_fn)
        histories[label] = h
        lines.append(f"| {label} | {h['val_acc'][-1]:.4f} | {test_acc:.4f} | {test_loss:.4f} |")

    plot_comparison(histories, RESULTS_DIR / "experiments_curves.png",
                    title=f"Ablations ({EPOCHS} epochs)")
    (RESULTS_DIR / "experiments.md").write_text("\n".join(lines) + "\n")
    (RESULTS_DIR / "experiments_history.json").write_text(json.dumps(histories, indent=2))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
