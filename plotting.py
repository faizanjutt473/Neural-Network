"""Matplotlib helpers (kept outside nn/ so the network stays NumPy-only)."""

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt


def plot_training_curves(history, path, title="NumPy Neural Network"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(epochs, history["train_loss"], label="Train", marker="o", ms=3)
    ax[0].plot(epochs, history["val_loss"], label="Validation", marker="s", ms=3)
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")

    ax[1].plot(epochs, history["train_acc"], label="Train", marker="o", ms=3)
    ax[1].plot(epochs, history["val_acc"], label="Validation", marker="s", ms=3)
    ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")

    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(histories, path, title="Comparison"):
    """histories: {label: history_dict}. Overlays validation/train curves."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    styles = ["-", "--", "-.", ":"]
    for (label, h), ls in zip(histories.items(), styles):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax[0].plot(epochs, h["train_loss"], ls, label=label)
        ax[1].plot(epochs, h["val_loss"], ls, label=label)
        ax[2].plot(epochs, h["val_acc"], ls, label=label)
    ax[0].set(title="Train loss", xlabel="Epoch", ylabel="Loss")
    ax[1].set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
    ax[2].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
