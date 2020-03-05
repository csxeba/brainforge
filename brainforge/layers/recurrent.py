import abc

import numpy as np

from .abstract_layer import FFBase
from .. import atomic
from ..util import ctx1, zX, zX_like, white, white_like

sigmoid = atomic.Sigmoid()


class RecurrentBase(FFBase):

    def __init__(self, neurons, activation, return_seq=False, **kw):
        super().__init__(neurons, activation, **kw)
        self.Z = 0
        self.Zs = []
        self.cache = None
        self.gates = []

        self.time = 0
        self.return_seq = return_seq

    @abc.abstractmethod
    def feedforward(self, X):
        self.inputs = X.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.Zs, self.gates, self.cache = [], [], []
        return zX(len(X), self.neurons)

    @abc.abstractmethod
    def backpropagate(self, delta):
        self.nabla_w = zX_like(self.weights)
        self.nabla_b = zX_like(self.biases)
        if self.return_seq:
            return delta.transpose(1, 0, 2)
        else:
            error_tensor = zX(self.time, len(delta), self.neurons)
            error_tensor[-1] = delta
            return error_tensor

    @property
    def outshape(self):
        return self.neurons,

    def __str__(self):
        return self.__class__.__name__ + "-{}-{}".format(self.neurons, str(self.activation)[:4])


class RLayer(RecurrentBase):

    def __init__(self, neurons, activation, return_seq=False, **kw):
        super().__init__(neurons, activation, return_seq, **kw)
        if self.compiled:
            from .. import llatomic
            print("Compiling RLayer...")
            self.op = llatomic.RecurrentOp(activation)
        else:
            self.op = atomic.RecurrentOp(activation)

    def connect(self, brain):
        self.Z = brain.outshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons)
        self.biases = zX(self.neurons)
        super().connect(brain)

    def feedforward(self, X):
        super().feedforward(X)
        self.output, self.Z = self.op.forward(self.inputs, self.weights, self.biases)
        return self.output.transpose(1, 0, 2) if self.return_seq else self.output[-1]

    def backpropagate(self, delta):
        delta = super().backpropagate(delta)
        dX, self.nabla_w, self.nabla_b = self.op.backward(
            Z=self.Z, O=self.output, E=delta, W=self.weights
        )
        return dX.transpose(1, 0, 2)


class LSTM(RecurrentBase):

    def __init__(self, neurons, activation, bias_init_factor=7., return_seq=False, **kw):
        super().__init__(neurons, activation, return_seq, **kw)
        self.G = neurons * 3
        self.Zs = []
        self.gates = []
        self.bias_init_factor = bias_init_factor
        if self.compiled:
            from .. import llatomic
            print("Compiling LSTM...")
            self.op = llatomic.LSTMOp(activation)
        else:
            self.op = atomic.LSTMOp(activation)

    def connect(self, brain):
        self.Z = brain.outshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons * 4)
        self.biases = zX(self.neurons * 4) + self.bias_init_factor
        super().connect(brain)

    def feedforward(self, X):
        super().feedforward(X)
        self.output, self.Z, self.cache = self.op.forward(
            X=self.inputs, W=self.weights, b=self.biases
        )
        return self.output.transpose(1, 0, 2) if self.return_seq else self.output[-1]

    def backpropagate(self, delta):
        delta = super().backpropagate(delta)
        dX, self.nabla_w, self.nabla_b = self.op.backward(
            Z=self.Z, O=self.output, E=delta, W=self.weights, cache=self.cache
        )
        return dX.transpose(1, 0, 2)


class GRU(RecurrentBase):

    def connect(self, brain):
        self.weights = white(brain.outshape[-1] + self.neurons, self.neurons * 3)
        self.biases = zX(self.neurons * 3)
        super().connect(brain)

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
            U, R = np.split(sigmoid.forward(Z @ Wur + bur), 2, axis=1)
            K = R * output

            Zk = ctx1(self.inputs[t], K)
            O = self.activation.forward(Zk @ Wo + bo)
            output = U * output + (1. - U) * O

            cache.append(output)
            self.gates.append([U, R, O])
            self.Zs.append([Z, Zk])

        if self.return_seq:
            self.output = np.stack(cache, axis=1)
        else:
            self.output = cache[-1]

        return self.output

    def backpropagate(self, delta):
        # alias these
        ct = np.concatenate
        dact = self.activation.backward
        dsig = sigmoid.backward

        delta = super().backpropagate(delta)

        dh = zX_like(delta[-1])
        dX = zX_like(self.inputs)

        Wu, Wr, Wo = np.split(self.weights, 3, axis=1)
        neu = self.neurons

        for t in range(-1, -(self.time + 1), -1):
            (U, R, O), (Z, Zk) = self.gates[t], self.Zs[t]
            prevout = Z[:, -neu:]
            dh += delta[t]
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


class ClockworkLayer(RecurrentBase):

    def __init__(self, neurons, activation, blocksizes=None, ticktimes=None, return_seq=False):
        super().__init__(neurons, activation, return_seq)

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
        self.tick_array = zX(self.neurons)
        print("CW blocks:", self.blocksizes)
        print("CW ticks :", self.ticks)

    def connect(self, brain):

        self.Z = brain.outshape[-1] + self.neurons

        W = zX(self.neurons, self.neurons)
        U = white(brain.outshape[-1], self.neurons)

        for i, bls in enumerate(self.blocksizes):
            start = i * bls
            end = start + bls
            W[start:end, start:] = white_like(W[start:end, start:])
            self.tick_array[start:end] = self.ticks[i]

        self.weights = np.concatenate((W, U), axis=0)
        self.biases = zX(self.neurons)

        super().connect(brain)

    def feedforward(self, X):
        output = super().feedforward(X)

        for t in range(1, self.time + 1):
            time_gate = np.equal(t % self.tick_array, 0.)
            Z = np.concatenate((self.inputs[t - 1], output), axis=-1)
            gated_W = self.weights * time_gate[None, :]
            gated_b = self.biases * time_gate
            output = self.activation.forward(Z.dot(gated_W) + gated_b)

            self.Zs.append(Z)
            self.gates.append([time_gate, gated_W])
            self.cache.append(output)

        if self.return_seq:
            self.output = np.stack(self.cache, axis=1)
        else:
            self.output = self.cache[-1]

        return self.output

    def backpropagate(self, delta):
        delta = super().backpropagate(delta)

        dh = zX_like(delta[-1])
        dX = zX_like(self.inputs)

        for t in range(self.time - 1, -1, -1):
            output = self.cache[t]
            Z = self.Zs[t]
            time_gate, gated_W = self.gates[t]

            dh += delta[t]
            dh *= self.activation.backward(output)

            self.nabla_w += (Z.T @ dh) * time_gate[None, :]
            self.nabla_b += dh.sum(axis=0) * time_gate

            deltaZ = dh @ gated_W.T
            dX[t] = deltaZ[:, :-self.neurons]
            dh = deltaZ[:, -self.neurons:]

        return dX.transpose(1, 0, 2)


class Reservoir(RLayer):

    trainable = False

    def __init__(self, neurons, activation, return_seq=False, r=0.1):
        RLayer.__init__(self, neurons, activation, return_seq)
        self.r = r

    def connect(self, brain):
        super().connect(brain)
        wx, wy = self.weights.shape
        # Create a sparse weight matrix (biases are included)
        W = np.random.binomial(1, self.r, size=(wx, wy + 1)).astype(float)
        W *= np.random.randn(wx, wy + 1)
        S = np.linalg.svd(W, compute_uv=False)  # compute singular values
        W /= S[0] ** 2  # scale to unit spectral radius
        self.weights = W[:, :-1]
        self.biases = W[:, -1]
