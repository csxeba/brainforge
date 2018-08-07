import numpy as np
from .abstract_learner import Learner


class ExtremeLearningMachine(Learner):

    def __init__(self, layers, cost="mse", name="", solve_mode="pseudoinverse", **kw):
        super().__init__(layers, cost, name, **kw)
        self.solve = {
            "pseudoinverse": self.solve_with_pseudo_inverse,
            "covariance": self.solve_with_covariance_matrices,
            "correlation": self.solve_with_covariance_matrices
        }[solve_mode]
        for layer in layers[:-1]:
            layer.trainable = False

    def solve_with_pseudo_inverse(self, Z, Y):
        A = np.linalg.pinv(Z)
        Wo = A @ Y
        self.layers[-1].set_weights([Wo, np.array([0] * self.layers[-1].neurons)], fold=False)

    def solve_with_covariance_matrices(self, Z, Y):
        A = np.cov(Z.T)
        B = np.cov(Z.T, Y.T)
        W = np.invert(A) @ B
        self.layers[-1].set_weights([W, np.array([0] * self.layers[-1].neurons)], fold=False)

    def learn_batch(self, X, Y, **kwarg):
        H = X.copy()
        for layer in self.layers[:-1]:
            H = layer.feedforward(H)
        self.solve(H, Y)
