import numpy as np

from .backpropagation import BackpropNetwork
from ..util.typing import white


class DirectFeedbackAlignment(BackpropNetwork):

    def __init__(self, layerstack, cost, optimizer, name="", **kw):
        super().__init__(layerstack, cost, optimizer, name, **kw)
        self.backwards_weights = [white(self.outshape[0], np.prod(layer.outshape))
                                  for layer in self.trainable_layers[:-1]]

    def backpropagate(self, error):
        m = len(error)
        self.layers[-1].backpropagate(error)
        for layer, weight in zip(list(self.trainable_layers)[:-1], self.backwards_weights):
            delta = error @ weight
            layer.backpropagate(delta.reshape((m,) + layer.outshape))
