import abc

from brainforge import backend as xp
from brainforge import activations

from .abstract_layer import Parameterized
from .. import initializers


class RecurrentBase(Parameterized):

    def __init__(self, units, activation, return_seq=False):
        super().__init__(units)
        self.Z = 0
        self.Zs = []
        self.cache = []
        self.gates = []
        self.units = units
        self.activation = activation
        if isinstance(self.activation, str):
            self.activation = activations.get(activation)

        self.time = 0
        self.return_seq = return_seq

        self.cache = None

    @abc.abstractmethod
    def forward(self, stimuli):
        self.inputs = stimuli.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.Zs, self.gates, self.cache = [], [], []
        return xp.zeros((len(stimuli), self.units))

    @abc.abstractmethod
    def backward(self, error):
        self.nabla_w = xp.zeros_like(self.weights)
        self.nabla_b = xp.zeros_like(self.biases)
        if self.return_seq:
            return error.transpose(1, 0, 2)
        else:
            error_tensor = xp.zeros((self.time, len(error), self.units))
            error_tensor[-1] = error
            return error_tensor

    @property
    def outshape(self):
        if self.return_seq:
            return self.time, self.units
        else:
            return self.units,


class RLayer(RecurrentBase):

    def connect(self, brain):
        self.Z = brain.outshape[-1] + self.units
        self.weights = initializers.glorot_normal(self.Z, self.units)
        self.biases = xp.zeros(self.units)
        super().connect(brain)

    def forward(self, x):

        output = super().forward(x)

        for t in range(self.time):
            Z = xp.concatenate((self.inputs[t], output), axis=-1)
            output = self.activation(Z.dot(self.weights) + self.biases)

            self.Zs.append(Z)
            self.cache.append(output)

        if self.return_seq:
            self.output = xp.stack(self.cache, axis=1)
        else:
            self.output = self.cache[-1]

        return self.output

    def backward(self, error):
        """
        Backpropagation through time (BPTT)

        :param error: the deltas flowing from the next layer
        """

        error = super().backward(error)

        # the gradient flowing backwards in time
        dh = xp.zeros_like(error[-1])
        # the gradient wrt the whole input tensor: dC/dX = dC/dY_{l-1}
        dX = xp.zeros_like(self.inputs)

        for t in range(self.time - 1, -1, -1):
            output = self.cache[t]
            Z = self.Zs[t]

            dh += error[t]
            dh *= self.activation.derivative(output)

            self.nabla_w += Z.T @ dh
            self.nabla_b += dh.sum(axis=0)

            deltaZ = dh @ self.weights.T
            dX[t] = deltaZ[:, :-self.units]
            dh = deltaZ[:, -self.units:]

        return dX.transpose((1, 0, 2))


class LSTM(RecurrentBase):

    def __init__(self, units, activation, bias_init_factor=7., return_seq=False):
        super().__init__(units, activation, return_seq)
        self.G = units * 3
        self.Zs = []
        self.gates = []
        self.bias_init_factor = bias_init_factor

    def connect(self, brain):
        self.Z = brain.outshape[-1] + self.units
        self.weights = initializers.glorot_normal(self.Z, self.units * 4)
        self.biases = xp.zeros(self.units * 4) + self.bias_init_factor
        super().connect(brain)

    def forward(self, X):

        output = super().forward(X)
        state = xp.zeros_like(output)

        for t in range(self.time):
            Z = xp.concatenate((self.inputs[t], output), axis=1)

            preact = Z @ self.weights + self.biases  # type: xp.ndarray
            preact[:, :self.G] = activations.sigmoid(preact[:, :self.G])
            preact[:, self.G:] = self.activation(preact[:, self.G:])

            f, i, o, cand = xp.split(preact, 4, axis=-1)

            state = state * f + i * cand
            state_a = self.activation(state)
            output = state_a * o

            self.Zs.append(Z)
            self.gates.append(preact)
            self.cache.append([output, state_a, state, preact])

        if self.return_seq:
            self.output = xp.stack([cache[0] for cache in self.cache], axis=1)
        else:
            self.output = self.cache[-1][0]
        return self.output

    def backward(self, error):

        error = super().backward(error)

        actprime = self.activation.derivative
        sigprime = activations.sigmoid.derivative

        dC = xp.zeros_like(error[-1])
        dX = xp.zeros_like(self.inputs)
        dZ = xp.zeros_like(self.Zs[0])

        for t in range(-1, -(self.time + 1), -1):
            output, Ca, state, preact = self.cache[t]
            f, i, o, cand = xp.split(self.gates[t], 4, axis=-1)

            # Add recurrent delta to output delta
            error[t] += dZ[:, -self.units:]

            # Backprop into state
            dC += error[t] * o * actprime(Ca)

            state_yesterday = 0. if t == -self.time else self.cache[t - 1][2]
            # Calculate the gate derivatives
            df = state_yesterday * dC
            di = cand * dC
            do = Ca * error[t]
            dcand = i * dC * actprime(cand)  # Backprop nonlinearity
            dgates = xp.concatenate((df, di, do, dcand), axis=-1)
            dgates[:, :self.G] *= sigprime(self.gates[t][:, :self.G])  # Backprop nonlinearity

            dC *= f

            self.nabla_b += dgates.sum(axis=0)
            self.nabla_w += self.Zs[t].T @ dgates

            dZ = dgates @ self.weights.T

            dX[t] = dZ[:, :-self.units]

        return dX.transpose((1, 0, 2))


class GRU(RecurrentBase):

    def __init__(self, units, activation, return_seq=False):
        super().__init__(units, activation, return_seq)

    def connect(self, brain):
        self.weights = initializers.glorot_normal(brain.outshape[-1] + self.units, self.units * 3)
        self.biases = xp.zeros(self.units * 3)
        super().connect(brain)

    def forward(self, X):
        output = super().forward(X)
        neu = self.units

        self.inputs = X.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.cache = []

        Wur, Wo = self.weights[:, :-neu], self.weights[:, -neu:]
        bur, bo = self.biases[:-neu], self.biases[-neu:]

        cache = []

        for t in range(self.time):
            Z = xp.concatenate([self.inputs[t], output], axis=1)
            U, R = xp.split(activations.sigmoid(Z @ Wur + bur), 2, axis=1)
            K = R * output

            Zk = xp.concatenate([self.inputs[t], K], axis=1)
            O = self.activation(Zk @ Wo + bo)
            output = U * output + (1. - U) * O

            cache.append(output)
            self.gates.append([U, R, O])
            self.Zs.append([Z, Zk])

        if self.return_seq:
            self.output = xp.stack(cache, axis=1)
        else:
            self.output = cache[-1]

        return self.output

    def backward(self, error):
        # alias these
        ct = xp.concatenate
        dact = self.activation.derivative
        dsig = activations.sigmoid.derivative

        error = super().backward(error)

        dh = xp.zeros_like(error[-1])
        dX = xp.zeros_like(self.inputs)

        Wu, Wr, Wo = xp.split(self.weights, 3, axis=1)
        neu = self.units

        for t in range(-1, -(self.time + 1), -1):
            (U, R, O), (Z, Zk) = self.gates[t], self.Zs[t]
            prevout = Z[:, -neu:]
            dh += error[t]
            dU = (prevout - O) * dsig(U) * dh
            dO = (1. - U) * dact(O) * dh  # type: xp.ndarray
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

        return dX.transpose((1, 0, 2))


class ClockworkLayer(RecurrentBase):

    def __init__(self, units, activaton, blocksizes=None, ticktimes=None, return_seq=False):
        super().__init__(units, activaton, return_seq)

        if blocksizes is None:
            block = units // 5
            blocksizes = [block] * 5
            blocksizes[0] += (units % block)
        else:
            if sum(blocksizes) != self.units:
                msg = "Please specify blocksizes so that they sum up to the number "
                msg += "of neurons specified for this layer! ({})".format(units)
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
        self.ticks = xp.array(ticktimes)
        self.tick_array = xp.zeros(self.units)
        print("CW blocks:", self.blocksizes)
        print("CW ticks :", self.ticks)

    def connect(self, brain):

        self.Z = brain.outshape[-1] + self.units

        W = xp.zeros((self.units, self.units))
        U = initializers.glorot_normal(brain.outshape[-1], self.units)

        for i, bls in enumerate(self.blocksizes):
            start = i * bls
            end = start + bls
            W[start:end, start:] = initializers.glorot_normal(W[start:end, start:].shape)
            self.tick_array[start:end] = self.ticks[i]

        self.weights = xp.concatenate((W, U), axis=0)
        self.biases = xp.zeros(self.units)

        super().connect(brain)

    def forward(self, stimuli):
        output = super().forward(stimuli)

        for t in range(1, self.time + 1):
            time_gate = xp.equal(t % self.tick_array, 0.)
            Z = xp.concatenate((self.inputs[t - 1], output), axis=-1)
            gated_W = self.weights * time_gate[None, :]
            gated_b = self.biases * time_gate
            output = self.activation(Z.dot(gated_W) + gated_b)

            self.Zs.append(Z)
            self.gates.append([time_gate, gated_W])
            self.cache.append(output)

        if self.return_seq:
            self.output = xp.stack(self.cache, axis=1)
        else:
            self.output = self.cache[-1]

        return self.output

    def backward(self, error):
        error = super().backward(error)

        dh = xp.zeros_like(error[-1])
        dX = xp.zeros_like(self.inputs)

        for t in range(self.time - 1, -1, -1):
            output = self.cache[t]
            Z = self.Zs[t]
            time_gate, gated_W = self.gates[t]

            dh += error[t]
            dh *= self.activation.derivative(output)

            self.nabla_w += (Z.T @ dh) * time_gate[None, :]
            self.nabla_b += dh.sum(axis=0) * time_gate

            deltaZ = dh @ gated_W.T
            dX[t] = deltaZ[:, :-self.units]
            dh = deltaZ[:, -self.units:]

        return dX.transpose((1, 0, 2))


class Reservoir(RLayer):

    trainable = False

    def __init__(self, units, activation, return_seq=False):
        RLayer.__init__(self, units, activation, return_seq)
        self.trainable = False

    def connect(self, brain):
        super().connect(brain)
        wx, wy = self.weights.shape
        # Create a sparse weight matrix (biases are included)
        W = xp.random.binomial(1, self.r, size=(wx, wy + 1)).astype(float)
        W *= xp.random.randn(wx, wy + 1)
        S = xp.linalg.svd(W, compute_uv=False)  # compute singular values
        W /= S[0] ** 2  # scale to unit spectral radius
        self.weights = W[:, :-1]
        self.biases = W[:, -1]
