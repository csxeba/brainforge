import numpy as np


class Experience:

    def __init__(self, limit=40000, mode="drop"):
        self.limit = limit
        self.X = []
        self.Y = []
        self._adder = {"drop": self._add_drop, "mix in": self._mix_in}[mode]

    @property
    def N(self):
        return len(self.X)

    def initialize(self, X, Y):
        self.X = X
        self.Y = Y

    def _mix_in(self, X, Y):
        m = len(X)
        N = self.N
        narg = np.arange(N)
        np.random.shuffle(narg)
        marg = narg[:m]
        self.X[marg] = X
        self.Y[marg] = Y

    def _add_drop(self, X, Y):
        m = len(X)
        self.X = np.concatenate((self.X[m:], X))
        self.Y = np.concatenate((self.Y[m:], Y))

    def _add(self, X, Y):
        N = self.N
        self.X = np.concatenate((self.X, X))
        self.Y = np.concatenate((self.Y, Y))
        self.X = self.X[-N:]
        self.Y = self.Y[-N:]

    def _incorporate(self, X, Y):
        if self.N < self.limit:
            self._add(X, Y)
        self._adder(X, Y)

    def remember(self, X, Y):
        assert len(X) == len(Y)
        if len(self.X) < 1:
            self.initialize(X, Y)
        else:
            self._incorporate(X, Y)

    def replay(self, batch_size):
        narg = np.arange(self.N)
        np.random.shuffle(narg)
        batch_args = narg[:batch_size]
        if len(batch_args) == 0:
            return [], []
        return self.X[batch_args], self.Y[batch_args]


class LameXP:

    def __init__(self, limit=3000):
        from collections import deque
        self.xp = deque(maxlen=limit)

    def remember(self, s, a, r, s_):
        if s is None:
            return
        self.xp.append((s, a, r, s_))

    def replay_stream(self, batch_size=None):
        if not batch_size:
            batch_size = len(self.xp)
        arg = np.arange(len(self.xp))
        np.random.shuffle(arg)
        for i in arg[:batch_size]:
            yield self.xp[i]


def discount_rewards(rwd, gamma=0.99):
    """
    Compute the discounted reward backwards in time
    """
    discounted_r = np.zeros_like(rwd)
    running_add = rwd[-1]
    for t, r in enumerate(rwd[::-1]):
        running_add += gamma * r
        discounted_r[t] = running_add
    discounted_r[0] = rwd[-1]
    return discounted_r[::-1]
