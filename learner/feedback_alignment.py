import numpy as np

from .backpropagation import BackpropNetwork
from ..util.typing import white


class DirectFeedbackAlignment(BackpropNetwork):

    def __init__(self, layerstack, cost, optimizer, name="", **kw):
        super().__init__(layerstack, cost, optimizer, name, **kw)
        self.backwards_weights = np.concatenate(
            [white(self.outshape[0], np.prod(layer.outshape))
             for layer in self.trainable_layers[:-1]]
        )

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
