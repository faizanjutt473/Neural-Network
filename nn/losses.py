"""Loss functions."""

import numpy as np


class SoftmaxCrossEntropy:
    """Softmax + categorical cross-entropy, fused.

    Loss  = -(1/N) * sum_n sum_k y_nk * log(softmax(z_n)_k)
    Grad  = (softmax(z) - y) / N          (w.r.t. the logits z)

    Fusing the two gives a very simple gradient and uses the log-sum-exp
    trick, so there is no log(0) or exp overflow.
    """

    def forward(self, logits, y_onehot):
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp
        self.probs = np.exp(log_probs)
        self.y = y_onehot
        return float(-(y_onehot * log_probs).sum() / logits.shape[0])

    def backward(self):
        return (self.probs - self.y) / self.y.shape[0]
