import numpy as np

from .abstract_learner import Learner
from brainforge.optimization import optimizers, GradientDescent


class BackpropNetwork(Learner):

    def __init__(self, layerstack, cost="mse", optimizer="sgd", name="", **kw):
        super().__init__(layerstack, cost, name, **kw)
        self.optimizer = (
            optimizer if isinstance(optimizer, GradientDescent) else optimizers[optimizer]()
        )
        self.optimizer.initialize(nparams=self.layers.num_params)

    def learn_batch(self, X, Y, w=None):
        m = len(X)
        preds = self.predict(X)
        delta = self.cost.derivative(preds, Y)
        if w is not None:
            delta *= w[:, None]
        self.backpropagate(delta)
        self.update(m)
        return self.cost(self.output, Y) / m

    def backpropagate(self, error):
        # TODO: optimize this, skip untrainable layers at the beginning
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)
        return error

    def update(self, m):
        W = self.layers.get_weights(unfold=True)
        gW = self.get_gradients(unfold=True)
        oW = self.optimizer.optimize(W, gW, m)
        self.layers.set_weights(oW, fold=True)

    def get_weights(self, unfold=True):
        self.layers.get_weights(unfold=unfold)

    def set_weigts(self, ws, fold=True):
        self.layers.set_weights(ws=ws, fold=fold)

    def get_gradients(self, unfold=True):
        grads = [l.gradients for l in self.layers if l.trainable]
        if unfold:
            grads = np.concatenate(grads)
        return grads

    @property
    def nparams(self):
        return self.layers.num_params
