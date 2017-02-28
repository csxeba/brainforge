import abc

import numpy as np

from .core import FFBase
from ..ops import Sigmoid
from ..util import white, white_like


sigmoid = Sigmoid()


class _Recurrent(FFBase):

    def __init__(self, neurons, activation, return_seq=False):
        FFBase.__init__(self, neurons, activation)
        self.Z = 0
        self.Zs = []
        self.cache = []
        self.gates = []

        self.time = 0
        self.return_seq = return_seq

        self.cache = None

    @abc.abstractmethod
    def feedforward(self, stimuli):
        self.inputs = stimuli.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.Zs, self.gates, self.cache = [], [], []
        return np.zeros((self.brain.m, self.neurons))

    @abc.abstractmethod
    def backpropagate(self, error):
        self.nabla_w = np.zeros_like(self.weights)
        self.nabla_b = np.zeros_like(self.biases)
        if self.return_seq:
            return error.transpose(1, 0, 2)
        else:
            error_tensor = np.zeros((self.time, self.brain.m, self.neurons))
            error_tensor[-1] = error
            return error_tensor

    def capsule(self):
        return FFBase.capsule(self) + [self.neurons, self.activation, self.return_seq,
                                       self.get_weights(unfold=False)]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(neurons=capsule[2], activation=capsule[3], return_seq=capsule[4])

    @property
    def outshape(self):
        if self.return_seq:
            return self.time, self.neurons
        else:
            return self.neurons,


class RLayer(_Recurrent):

    def connect(self, to, inshape):
        self.Z = inshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons)
        self.biases = np.zeros((self.neurons,))

        _Recurrent.connect(self, to, inshape)

    def feedforward(self, questions: np.ndarray):

        output = _Recurrent.feedforward(self, questions)

        for t in range(self.time):
            Z = np.concatenate((self.inputs[t], output), axis=-1)
            output = self.activation(Z.dot(self.weights) + self.biases)

            self.Zs.append(Z)
            self.cache.append(output)

        if self.return_seq:
            self.output = np.stack(self.cache, axis=1)
        else:
            self.output = self.cache[-1]

        return self.output

    def backpropagate(self, error):
        """
        Backpropagation through time (BPTT)

        :param error: the deltas flowing from the next layer
        """

        error = _Recurrent.backpropagate(self, error)

        # the gradient flowing backwards in time
        dh = np.zeros_like(error[-1])
        # the gradient wrt the whole input tensor: dC/dX = dC/dY_{l-1}
        dX = np.zeros_like(self.inputs)

        for t in range(self.time-1, -1, -1):
            output = self.cache[t]
            Z = self.Zs[t]

            dh += error[t]
            dh *= self.activation.derivative(output)

            self.nabla_w += Z.T @ dh
            self.nabla_b += dh.sum(axis=0)

            deltaZ = dh @ self.weights.T
            dX[t] = deltaZ[:, :-self.neurons]
            dh = deltaZ[:, -self.neurons:]

        return dX.transpose(1, 0, 2)

    def __str__(self):
        return "RLayer-{}-{}".format(self.neurons, str(self.activation))


class LSTM(_Recurrent):

    def __init__(self, neurons, activation, return_seq=False):
        _Recurrent.__init__(self, neurons, activation, return_seq)
        self.G = neurons * 3
        self.Zs = []
        self.gates = []

    def connect(self, to, inshape):
        _Recurrent.connect(self, to, inshape)
        self.Z = inshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons * 4)
        self.biases = np.zeros((self.neurons * 4,))

    def feedforward(self, X: np.ndarray):

        output = _Recurrent.feedforward(self, X)
        state = np.zeros_like(output)

        for t in range(self.time):
            Z = np.concatenate((self.inputs[t], output), axis=1)

            preact = Z @ self.weights + self.biases
            preact[:, :self.G] = sigmoid(preact[:, :self.G])
            preact[:, self.G:] = self.activation(preact[:, self.G:])

            f, i, o, cand = np.split(preact, 4, axis=-1)

            state = state * f + i * cand
            state_a = self.activation(state)
            output = state_a * o

            self.Zs.append(Z)
            self.gates.append(preact)
            self.cache.append([output, state_a, state, preact])

        if self.return_seq:
            self.output = np.stack([cache[0] for cache in self.cache], axis=1)
        else:
            self.output = self.cache[-1][0]
        return self.output

    def backpropagate(self, error):

        error = _Recurrent.backpropagate(self, error)

        actprime = self.activation.derivative
        sigprime = sigmoid.derivative

        dC = np.zeros_like(error[-1])
        dX = np.zeros_like(self.inputs)
        dZ = np.zeros_like(self.Zs[0])

        for t in range(-1, -(self.time+1), -1):
            output, Ca, state, preact = self.cache[t]
            f, i, o, cand = np.split(self.gates[t], 4, axis=-1)

            # Add recurrent delta to output delta
            error[t] += dZ[:, -self.neurons:]

            # Backprop into state
            dC += error[t] * o * actprime(Ca)

            state_yesterday = 0. if t == -self.time else self.cache[t-1][2]
            # Calculate the gate derivatives
            df = state_yesterday * dC
            di = cand * dC
            do = Ca * error[t]
            dcand = i * dC * actprime(cand)  # Backprop nonlinearity
            dgates = np.concatenate((df, di, do, dcand), axis=-1)
            dgates[:, :self.G] *= sigprime(self.gates[t][:, :self.G])  # Backprop nonlinearity

            dC *= f

            self.nabla_b += dgates.sum(axis=0)
            self.nabla_w += self.Zs[t].T @ dgates

            dZ = dgates @ self.weights.T

            dX[t] = dZ[:, :-self.neurons]

        return dX.transpose(1, 0, 2)

    def __str__(self):
        return "LSTM-{}-{}".format(self.neurons, str(self.activation)[:4])


class GRU(_Recurrent):

    def __init__(self, neurons, activation, return_seq=False):
        super().__init__(neurons, activation, return_seq)

    def connect(self, to, inshape):
        super().connect(to, inshape)
        self.weights = white(inshape[-1] + self.neurons, self.neurons * 3)
        self.biases = np.zeros(self.neurons * 3)

    def feedforward(self, X):
        output = super().feedforward(X)
        neu = self.neurons

        self.inputs = X.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.cache = []

        Wur, Wo = self.weights[:, :-neu], self.weights[:, -neu:]
        bur, bo = self.biases[:-neu], self.biases[-neu:]

        ctx1 = lambda *ar: np.concatenate(ar, axis=1)

        for t in range(self.time):
            Z = ctx1(self.inputs[t], output)
            U, R = np.split(sigmoid(Z @ Wur + bur), 2, axis=1)
            K = R * output
            Zk = ctx1(self.inputs[t], K)
            O = self.activation(Zk @ Wo + bo)
            output = U * output + (1. - U) * O

            self.cache.append([output, K])
            self.gates.append([U, R, O])
            self.Zs.append([Z, Zk])

        if self.return_seq:
            self.output = np.stack([cache[0] for cache in self.cache], axis=1)
        else:
            self.output = self.cache[-1][0]

        return self.output

    def backpropagate(self, error):
        error = super().backpropagate(error)

        actprime = self.activation.derivative
        sigprime = sigmoid.derivative

        dh = np.zeros_like(error[-1])
        dX = np.zeros_like(self.inputs)

        Wu, Wr, Wo = np.split(self.weights, 3, axis=1)
        neu = self.neurons
        ct = lambda *ar, ax=1: np.concatenate(ar, axis=ax)

        for t in range(-1, -(self.time+1), -1):
            (output, K), (U, R, O), (Z, Zk) = self.cache[t], self.gates[t], self.Zs[t]
            dh += error[t]
            dU = (Z[:, -neu:] - O) * sigprime(U) * dh
            dO = (1. - U) * actprime(O) * dh  # type: np.ndarray
            dZk = dO @ Wo.T
            dK = dZk[:, -neu:]
            dR = output * sigprime(R) * dK
            dZ = ct(dU, dR) @ ct(Wu, Wr).T

            nWur = Z.T @ ct(dU, dR)
            nWo = Zk.T @ dO

            dh = dZ[:, -neu:] + R * dK + U * dh
            dX[t] = dZ[:, :-neu] + dZk[:, :-neu]

            self.nabla_w += ct(nWur, nWo)
            self.nabla_b += ct(dU, dR, dO).sum(axis=0)

        return dX.transpose(1, 0, 2)

    def __str__(self):
        return "GRU-{}-{}".format(self.neurons, str(self.activation)[:4])


class ClockworkLayer(_Recurrent):
    def __init__(self, neurons, clocktimes, activaton, return_seq=False):
        super().__init__(neurons, activaton, return_seq)

        self.time_mask = None

    def connect(self, to, inshape):
        pass

    def feedforward(self, stimuli):
        pass

    def backpropagate(self, error):
        pass

    def __str__(self):
        pass


class Reservoir(RLayer):

    def __init__(self, neurons, activation, return_seq=False, p=0.1):
        RLayer.__init__(self, neurons, activation, return_seq)
        self.trainable = False
        self.p = p

    def connect(self, to, inshape):
        RLayer.connect(self, to, inshape)
        self.weights = np.random.binomial(1., self.p, size=self.weights.shape).astype(float)
        self.weights *= np.random.randn(*self.weights.shape)
        self.biases = white_like(self.biases)

    def __str__(self):
        return "Echo-{}-{}".format(self.neurons, str(self.activation)[:4])