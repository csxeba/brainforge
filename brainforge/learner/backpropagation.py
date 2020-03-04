import numpy as np

from .abstract_learner import Learner
from ..optimizers import optimizers, GradientDescent


class Backpropagation(Learner):

    def __init__(self, layerstack, cost="mse", optimizer="sgd", name="", **kw):
        super().__init__(layerstack, cost, name, **kw)
        self.optimizer = (
            optimizer if isinstance(optimizer, GradientDescent) else optimizers[optimizer]()
        )
        self.optimizer.initialize(nparams=self.layers.num_params)

    def learn_batch(self, X, Y, w=None, metrics=(), update=True):
        m = len(X)
        preds = self.predict(X)
        delta = self.cost.derivative(preds, Y)
        if w is not None:
            delta *= w[:, None]
        self.backpropagate(delta)
        if update:
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
        self.zero_gradients()

    def get_weights(self, unfold=True):
        self.layers.get_weights(unfold=unfold)

    def set_weigts(self, ws, fold=True):
        self.layers.set_weights(ws=ws, fold=fold)

    def get_gradients(self, unfold=True):
        grads = [l.gradients for l in self.layers if l.trainable]
        if unfold:
            grads = np.concatenate(grads)
        return grads

    def set_gradients(self, gradients, fold=True):
        trl = (l for l in self.layers if l.trainable)
        if fold:
            start = 0
            for layer in trl:
                end = start + layer.num_params
                layer.set_weights(gradients[start:end])
                start = end
        else:
            for w, layer in zip(gradients, trl):
                layer.set_weights(w)

    def zero_gradients(self):
        for layer in self.layers:
            if not layer.trainable:
                continue
            layer.nabla_w *= 0
            layer.nabla_b *= 0

    @property
    def num_params(self):
        return self.layers.num_params

    @property
    def output_shape(self):
        return self.layers.output_shape

    @property
    def input_shape(self):
        return self.layers.input_shape
