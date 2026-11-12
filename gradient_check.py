"""Numerical gradient checking of the hand-written backpropagation.

For every parameter theta_i we compare the analytic gradient with the
centered finite difference  (L(theta_i + eps) - L(theta_i - eps)) / (2 eps).

Relative error = ||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric||)
A value below 1e-7 means backprop is correct. Uses float64 and a tiny network
because checking is O(#parameters) forward passes.
"""

import sys

import numpy as np

from config import RESULTS_DIR
from nn import NeuralNetwork, SoftmaxCrossEntropy, one_hot

EPS = 1e-5
THRESHOLD = 1e-7


def numerical_gradient(net, loss_fn, x, y, param):
    grad = np.zeros_like(param)
    for idx in np.ndindex(*param.shape):
        original = param[idx]
        param[idx] = original + EPS
        loss_plus = loss_fn.forward(net.forward(x), y)
        param[idx] = original - EPS
        loss_minus = loss_fn.forward(net.forward(x), y)
        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * EPS)
    return grad


def relative_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-15)


def check(activation, sizes=(5, 4, 4, 3), batch=8, seed=0):
    rng = np.random.default_rng(seed)
    net = NeuralNetwork(list(sizes), activation=activation, seed=seed, dtype=np.float64)
    loss_fn = SoftmaxCrossEntropy()
    # Non-zero biases: with b = 0 a ReLU pre-activation can be exactly 0 (the
    # non-differentiable kink) and finite differences would disagree with backprop.
    for layer in net.dense_layers:
        layer.b[...] = rng.standard_normal(layer.b.shape) * 0.1
    x = rng.standard_normal((batch, sizes[0]))
    y = one_hot(rng.integers(0, sizes[-1], batch), sizes[-1], dtype=np.float64)

    loss_fn.forward(net.forward(x), y)
    net.backward(loss_fn.backward())

    rows = []
    for li, layer in enumerate(net.dense_layers):
        for name, (param, grad) in zip(("W", "b"), layer.parameters()):
            analytic = grad.copy()
            numeric = numerical_gradient(net, loss_fn, x, y, param)
            rows.append((f"layer{li}.{name}", relative_error(analytic, numeric)))
    return rows


def main():
    lines, ok = [], True
    for act in ("relu", "tanh", "sigmoid"):
        lines.append(f"Activation: {act}  (network 5-4-4-3, eps={EPS}, float64)")
        for name, err in check(act):
            passed = err < THRESHOLD
            ok &= passed
            lines.append(f"  {name:<10} relative error = {err:.3e}  {'PASS' if passed else 'FAIL'}")
        lines.append("")
    lines.append("RESULT: " + ("ALL GRADIENTS CORRECT" if ok else "GRADIENT CHECK FAILED"))

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "gradient_check.txt").write_text(report + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
