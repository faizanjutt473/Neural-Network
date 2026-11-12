"""Compare NumPy vs PyTorch results: overlay plots + markdown table."""

import json

import numpy as np

from config import RESULTS_DIR
from plotting import plot_comparison


def main():
    np_h = json.loads((RESULTS_DIR / "numpy_history.json").read_text())
    pt_h = json.loads((RESULTS_DIR / "pytorch_history.json").read_text())

    plot_comparison({"NumPy (from scratch)": np_h, "PyTorch": pt_h},
                    RESULTS_DIR / "comparison_curves.png",
                    title="NumPy vs PyTorch (same init, same batches, same hyperparameters)")

    def row(label, key, fmt):
        a, b = np_h[key], pt_h[key]
        return f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {fmt.format(abs(a - b))} |"

    max_val_loss_gap = float(np.max(np.abs(np.array(np_h["val_loss"]) - np.array(pt_h["val_loss"]))))
    table = "\n".join([
        "| Metric | NumPy | PyTorch | Abs. difference |",
        "|---|---|---|---|",
        f"| Final train loss | {np_h['train_loss'][-1]:.4f} | {pt_h['train_loss'][-1]:.4f} "
        f"| {abs(np_h['train_loss'][-1] - pt_h['train_loss'][-1]):.4f} |",
        f"| Final val loss | {np_h['val_loss'][-1]:.4f} | {pt_h['val_loss'][-1]:.4f} "
        f"| {abs(np_h['val_loss'][-1] - pt_h['val_loss'][-1]):.4f} |",
        f"| Final val accuracy | {np_h['val_acc'][-1]:.4f} | {pt_h['val_acc'][-1]:.4f} "
        f"| {abs(np_h['val_acc'][-1] - pt_h['val_acc'][-1]):.4f} |",
        row("Test loss", "test_loss", "{:.4f}"),
        row("Test accuracy", "test_acc", "{:.4f}"),
        row("Training time (s)", "train_time", "{:.1f}"),
    ])
    summary = (f"{table}\n\nMax |val-loss difference| over all epochs: {max_val_loss_gap:.2e}\n")
    print(summary)
    (RESULTS_DIR / "comparison.md").write_text(summary)


if __name__ == "__main__":
    main()
