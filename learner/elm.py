import numpy as np
from .abstract_learner import Learner


class ExtremeLearningMachine(Learner):

    def __init__(self, layers, cost="mse", name="", **kw):
        super().__init__(layers, cost, name, **kw)
        for layer in layers[:-1]:
            layer.trainable = False

    def learn_batch(self, X, Y, **kwarg):
        Z = X.copy()
        for layer in self.layers[:-1]:
            Z = layer.feedforward(Z)
        Zpi = np.linalg.pinv(Z)
        Wo = Zpi @ Y
        self.layers[-1].set_weights([Wo, np.array([0] * self.layers[-1].neurons)], fold=False)
