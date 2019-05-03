import numpy as np


class Experience:

    def __init__(self, limit=40000, mode="drop", downsample=0):
        self.limit = limit
        self.X = []
        self.Y = []
        self._adder = {"drop": self._add_drop, "mix in": self._mix_in}[mode]
        self.downsample = downsample

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
        self.X = np.concatenate((self.X, X))
        self.Y = np.concatenate((self.Y, Y))
        self.X = self.X[-self.limit:]
        self.Y = self.Y[-self.limit:]

    def _incorporate(self, X, Y):
        if self.N < self.limit:
            self._add(X, Y)
        self._adder(X, Y)

    def remember(self, X, Y):
        if self.downsample > 1:
            X, Y = X[::self.downsample], Y[::self.downsample]
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


class TimeExperience(Experience):

    def __init__(self, time, limit=1600, downsample=0):
        super().__init__(limit, mode="drop", downsample=downsample)
        self.time = time

    def replay(self, batch_size):
        narg = np.arange(self.N)
        np.random.shuffle(narg)
        batch_starts = narg[:batch_size]
        X, Y = [], []
        for start in batch_starts:
            X.append(self.X[start:start+self.time])
            Y.append(self.Y[start:start+self.time])
        return np.stack(X, axis=0), np.stack(Y, axis=0)


def replay_memory_factory(limit, mode, time, downsample=0):
    if time > 1:
        return TimeExperience(time, limit, downsample)
    else:
        return Experience(limit, mode, downsample)
