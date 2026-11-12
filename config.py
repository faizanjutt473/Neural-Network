"""Shared hyperparameters so the NumPy and PyTorch runs are identical."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "mnist.npz"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
LAYER_SIZES = [784, 128, 64, 10]
ACTIVATION = "relu"
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 64
EPOCHS = 20
VAL_FRACTION = 0.1
NUM_CLASSES = 10
