import abc

import numpy as np

from .core import FFBase
from ..ops import Sigmoid
from ..util import white, white_like, zX, zX_like, ctx1


sigmoid = Sigmoid()


class RecurrentBase(FFBase):

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
        return zX(self.brain.m, self.neurons)

    @abc.abstractmethod
    def backpropagate(self, error):
        self.nabla_w = zX_like(self.weights)
        self.nabla_b = zX_like(self.biases)
        if self.return_seq:
            return error.transpose(1, 0, 2)
        else:
            error_tensor = zX(self.time, self.brain.m, self.neurons)
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


class RLayer(RecurrentBase):

    def connect(self, to, inshape):
        self.Z = inshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons)
        self.biases = zX(self.neurons,)

        RecurrentBase.connect(self, to, inshape)

    def feedforward(self, questions):

        output = RecurrentBase.feedforward(self, questions)

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

        error = RecurrentBase.backpropagate(self, error)

        # the gradient flowing backwards in time
        dh = zX_like(error[-1])
        # the gradient wrt the whole input tensor: dC/dX = dC/dY_{l-1}
        dX = zX_like(self.inputs)

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


class LSTM(RecurrentBase):

    def __init__(self, neurons, activation, bias_init_factor=7., return_seq=False):
        RecurrentBase.__init__(self, neurons, activation, return_seq)
        self.G = neurons * 3
        self.Zs = []
        self.gates = []
        self.bias_init_factor = bias_init_factor

    def connect(self, to, inshape):
        self.Z = inshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons * 4)
        self.biases = zX(self.neurons * 4,) + self.bias_init_factor
        RecurrentBase.connect(self, to, inshape)

    def feedforward(self, X):

        output = RecurrentBase.feedforward(self, X)
        state = zX_like(output)

        for t in range(self.time):
            Z = np.concatenate((self.inputs[t], output), axis=1)

            preact = Z @ self.weights + self.biases  # type: np.ndarray
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

        error = RecurrentBase.backpropagate(self, error)

        actprime = self.activation.derivative
        sigprime = sigmoid.derivative

        dC = zX_like(error[-1])
        dX = zX_like(self.inputs)
        dZ = zX_like(self.Zs[0])

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


class GRU(RecurrentBase):

    def __init__(self, neurons, activation, return_seq=False):
        super().__init__(neurons, activation, return_seq)

    def connect(self, to, inshape):
        self.weights = white(inshape[-1] + self.neurons, self.neurons * 3)
        self.biases = zX(self.neurons * 3)
        super().connect(to, inshape)

    def feedforward(self, X):
        output = super().feedforward(X)
        neu = self.neurons

        self.inputs = X.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.cache = []

        Wur, Wo = self.weights[:, :-neu], self.weights[:, -neu:]
        bur, bo = self.biases[:-neu], self.biases[-neu:]

        cache = []

        for t in range(self.time):
            Z = ctx1(self.inputs[t], output)
            U, R = np.split(sigmoid(Z @ Wur + bur), 2, axis=1)
            K = R * output

            Zk = ctx1(self.inputs[t], K)
            O = self.activation(Zk @ Wo + bo)
            output = U * output + (1. - U) * O

            cache.append(output)
            self.gates.append([U, R, O])
            self.Zs.append([Z, Zk])

        if self.return_seq:
            self.output = np.stack(cache, axis=1)
        else:
            self.output = cache[-1]

        return self.output

    def backpropagate(self, error):
        # alias these
        ct = np.concatenate
        dact = self.activation.derivative
        dsig = sigmoid.derivative

        error = super().backpropagate(error)

        dh = zX_like(error[-1])
        dX = zX_like(self.inputs)

        Wu, Wr, Wo = np.split(self.weights, 3, axis=1)
        neu = self.neurons

        for t in range(-1, -(self.time+1), -1):
            (U, R, O), (Z, Zk) = self.gates[t], self.Zs[t]
            prevout = Z[:, -neu:]
            dh += error[t]
            dU = (prevout - O) * dsig(U) * dh
            dO = (1. - U) * dact(O) * dh  # type: np.ndarray
            dZk = dO @ Wo.T
            dK = dZk[:, -neu:]
            dR = prevout * dsig(R) * dK
            dZ = ct((dU, dR), axis=1) @ ct((Wu, Wr), axis=1).T

            nWur = Z.T @ ct((dU, dR), axis=1)
            nWo = Zk.T @ dO

            dh = dZ[:, -neu:] + R * dK + U * dh
            dX[t] = dZ[:, :-neu] + dZk[:, :-neu]

            self.nabla_w += ct((nWur, nWo), axis=1)
            self.nabla_b += ct((dU, dR, dO), axis=1).sum(axis=0)

        return dX.transpose(1, 0, 2)

    def __str__(self):
        return "GRU-{}-{}".format(self.neurons, str(self.activation)[:4])


class ClockworkLayer(RLayer):

    def __init__(self, neurons, activaton, blocksizes=None, ticktimes=None, return_seq=False):
        super().__init__(neurons, activaton, return_seq)

        if blocksizes is None:
            block = neurons // 5
            blocksizes = [block] * 5
            blocksizes[0] += (neurons % block)
        else:
            if sum(blocksizes) != self.neurons:
                msg = "Please specify blocksizes so that they sum up to the number "
                msg += "of neurons specified for this layer! ({})".format(neurons)
                raise RuntimeError(msg)
        self.blocksizes = blocksizes

        if ticktimes is None:
            ticktimes = [2 ** i for i in range(len(self.blocksizes))]
        else:
            if min(ticktimes) < 0 or len(ticktimes) != len(self.blocksizes):
                msg = "Ticks specifies the timestep when each block is activated.\n"
                msg += "Please specify the <ticks> parameter so, that its length is "
                msg += "equal to the number of blocks specifid (defaults to 5). "
                msg += "Please also consider that timesteps < 0 are invalid!"
                raise RuntimeError(msg)
        self.ticks = np.array(ticktimes)
        self.tick_array = zX(self.neurons,)
        print("CW blocks:", self.blocksizes)
        print("CW ticks :", self.ticks)

    def connect(self, to, inshape):

        self.Z = inshape[-1] + self.neurons

        W = zX(self.neurons, self.neurons)
        U = white(inshape[-1], self.neurons)

        for i, bls in enumerate(self.blocksizes):
            start = i*bls
            end = start + bls
            W[start:end, start:] = white_like(W[start:end, start:])
            self.tick_array[start:end] = self.ticks[i]

        self.weights = np.concatenate((W, U), axis=0)
        self.biases = zX(self.neurons,)

        RecurrentBase.connect(self, to, inshape)

    def feedforward(self, stimuli):
        output = RecurrentBase.feedforward(self, stimuli)

        for t in range(1, self.time+1):
            time_gate = np.equal(t % self.tick_array, 0.)
            Z = np.concatenate((self.inputs[t-1], output), axis=-1)
            gated_W = self.weights * time_gate[None, :]
            gated_b = self.biases * time_gate
            output = self.activation(Z.dot(gated_W) + gated_b)

            self.Zs.append(Z)
            self.gates.append([time_gate, gated_W])
            self.cache.append(output)

        if self.return_seq:
            self.output = np.stack(self.cache, axis=1)
        else:
            self.output = self.cache[-1]

        return self.output

    def backpropagate(self, error):
        error = RecurrentBase.backpropagate(self, error)

        # the gradient flowing backwards in time
        dh = zX_like(error[-1])
        # the gradient wrt the whole input tensor: dC/dX = dC/dY_{l-1}
        dX = zX_like(self.inputs)

        for t in range(self.time-1, -1, -1):
            output = self.cache[t]
            Z = self.Zs[t]
            time_gate, gated_W = self.gates[t]

            dh += error[t]
            dh *= self.activation.derivative(output)

            self.nabla_w += (Z.T @ dh) * time_gate[None, :]
            self.nabla_b += dh.sum(axis=0) * time_gate

            deltaZ = dh @ gated_W.T
            dX[t] = deltaZ[:, :-self.neurons]
            dh = deltaZ[:, -self.neurons:]

        return dX.transpose(1, 0, 2)

    def __str__(self):
        return "ClockworkLayer-{}-{}".format(self.neurons, self.activation)


class Reservoir(RLayer):

    def __init__(self, neurons, activation, return_seq=False, p=0.1):
        RLayer.__init__(self, neurons, activation, return_seq)
        self.trainable = False
        self.p = p

    def connect(self, to, inshape):
        RLayer.connect(self, to, inshape)
        self.weights = np.random.binomial(1, self.p, size=self.weights.shape).astype(float)
        self.weights *= np.random.randn(*self.weights.shape)
        self.biases = white_like(self.biases)

    def backpropagate(self, error):
        if self.position > 1:
            return super().backpropagate(error)

    def __str__(self):
        return "Reservoir-{}-{}".format(self.neurons, str(self.activation)[:4])
