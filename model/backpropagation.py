import numpy as np

from .abstract_learner import Learner
from ..optimization import optimizers, GradientDescent


class BackpropNetwork(Learner):

    def __init__(self, layerstack, cost="mse", optimizer="sgd", name="", **kw):
        super().__init__(layerstack, cost, name, **kw)
        self.optimizer = (
            optimizer if isinstance(optimizer, GradientDescent) else optimizers[optimizer]()
        )
        self.optimizer.initialize(nparams=self.layers.nparams)

    def learn_batch(self, X, Y, w=None):
        m = len(X)
        preds = self.predict(X)
        delta = self.cost.derivative(preds, Y)
        if w is not None:
            delta *= w[:, None]
        self.layers.set_weights(
            self.optimizer.optimize(
                self.layers.get_weights(unfold=True),
                self.backpropagate(delta), m
            )
        )
        return self.cost(self.output, Y) / m

    def backpropagate(self, error):
        # TODO: optimize this, skip untrainable layers at the beginning
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)
        return self.get_gradients(unfold=True)

    def get_gradients(self, unfold=True):
        grads = [l.gradients for l in self.layers if l.trainable]
        if unfold:
            grads = np.concatenate(grads)
        return grads
