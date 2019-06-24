import numpy as np

from .abstract_learner import Learner
from ..optimizers import optimizers, GradientDescent


class BackpropNetwork(Learner):

    def __init__(self, layerstack, cost="mse", optimizer="sgd", name="", **kw):
        super().__init__(layerstack, cost, name, **kw)
        self.optimizer = (
            optimizer if isinstance(optimizer, GradientDescent) else optimizers[optimizer]()
        )
        self.optimizer.initialize(nparams=self.layers.num_params)

    def learn_batch(self, X, Y, w=None, metrics=()):
        m = len(X)
        preds = self.predict(X)
        delta = self.cost.derivative(preds, Y)
        if w is not None:
            delta *= w[:, None]
        self.backpropagate(delta)
        self.update(m)
        train_metrics = {"cost": self.cost(self.output, Y) / m}
        if metrics:
            for metric in metrics:
                train_metrics[str(metric).lower()] = metric(preds, Y) / m
        return train_metrics

    def backpropagate(self, error):
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
    def num_params(self):
        return self.layers.num_params
