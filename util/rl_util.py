import numpy as np


class Experience:

    def __init__(self, limit=40000):
        self.limit = limit
        self.X = []
        self.Y = []

    @property
    def N(self):
        return len(self.X)

    def initialize(self, X, Y):
        self.X = X
        self.Y = Y

    def incorporate(self, X, Y):
        m = len(X)
        N = self.N
        if N == self.limit:
            narg = np.arange(N)
            np.random.shuffle(narg)
            marg = narg[:m]
            self.X[marg] = X
            self.Y[marg] = Y
            return
        self.X = np.concatenate((self.X, X))
        self.Y = np.concatenate((self.Y, Y))
        if N > self.limit:
            narg = np.arange(N)
            np.random.shuffle(narg)
            self.X = self.X[narg]
            self.Y = self.Y[narg]

    def remember(self, X, Y):
        assert len(X) == len(Y)
        if len(self.X) < 1:
            self.initialize(X, Y)
        else:
            self.incorporate(X, Y)

        if self.N > self.limit:
            arg = np.arange(self.N)
            np.random.shuffle(arg)

    def replay(self, batch_size):
        narg = np.arange(self.N)
        np.random.shuffle(narg)
        batch_args = narg[:batch_size]
        if len(batch_args) == 0:
            return [], []
        return self.X[batch_args], self.Y[batch_args]


def discount_rewards(rwd, gamma=0.99):
    """
    Compute the discounted reward backwards in time
    """
    discounted_r = np.zeros_like(rwd)
    running_add = rwd[-1]
    for t in range(len(rwd)-2, -1, -1):
        running_add = running_add * gamma + rwd[t]
        discounted_r[t] = running_add
    discounted_r[-1] = rwd[-1]
    return discounted_r
