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
        self.backpropagate(delta)
        self.update(m)
        return self.cost(self.output, Y) / m

    def update(self, m):
        W = self.layers.get_weights(unfold=True)
        gW = self.get_gradients(unfold=True)
        self.layers.set_weights(self.optimizer.optimize(W, gW, m))

    def backpropagate(self, error):
        # TODO: optimize this, skip untrainable layers at the beginning
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)
            if error is None:
                break
        return self.get_gradients(unfold=True)

    def get_gradients(self, unfold=True):
        grads = [l.gradients for l in self.layers if l.trainable]
        if unfold:
            grads = np.concatenate(grads)
        return grads
