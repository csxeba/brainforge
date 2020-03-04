import numpy as np

from .backpropagation import Backpropagation
from ..metrics import mse


class LocalCorrelationAligment(Backpropagation):

    def backpropagate(self, error):
        m = len(error)
        self.layers[-1].backpropagate(error)
        all_deltas = error @ self.backwards_weights  # [m x net_out] [net_out x [layer_outs]] = [m x [layer_outs]]
        start = 0
        for layer in self.trainable_layers[1:-1]:
            num_deltas = np.prod(layer.outshape)
            end = start + num_deltas
            delta = all_deltas[:, num_deltas]
            layer.backpropagate(delta.reshape((m,) + layer.outshape))
            start = end

    def learn_batch(self, X, Y, w=None, metrics=(), update=True):
        m = len(X)
        Y_correlations = np.corrcoef(Y.reshape((m, -1)))

        for layer in self.trainable_layers:
            h_correlations = np.corrcoef(layer.output.reshape(m, -1))
            local_error = mse(h_correlations.flat, Y_correlations.flat)