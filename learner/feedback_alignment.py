import numpy as np

from .backpropagation import BackpropNetwork
from ..util.typing import white


class DirectFeedbackAlignment(BackpropNetwork):

    def __init__(self, layerstack, cost, optimizer, name="", **kw):
        super().__init__(layerstack, cost, optimizer, name, **kw)
        self.backwards_weights = [white(self.outshape[0], np.prod(layer.outshape))
                                  for layer in self.trainable_layers]

    def backpropagate(self, error):
        for layer, weight in zip(self.trainable_layers, self.backwards_weights):
            delta = error @ weight
            layer.backpropagate(delta)
