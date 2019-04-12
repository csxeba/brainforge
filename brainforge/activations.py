from brainforge import backend as xp


class ActivationFunction:

    def __call__(self, z: xp.ndarray) -> xp.ndarray:
        raise NotImplementedError

    def __str__(self):
        return self.type

    def derivative(self, Z: xp.ndarray) -> xp.ndarray:
        raise NotImplementedError

    @property
    def type(self):
        return self.__class__.__name__.lower()


class Sigmoid(ActivationFunction):

    def __call__(self, Z: xp.ndarray):
        return 1 / (1 + xp.exp(-Z))

    def derivative(self, A: xp.ndarray) -> xp.ndarray:
        return A * (1 - A)


class Tanh(ActivationFunction):

    def __call__(self, Z) -> xp.ndarray:
        return xp.tanh(Z)

    def derivative(self, A: xp.ndarray) -> xp.ndarray:
        return 1 - A**2


class Linear(ActivationFunction):

    def __call__(self, Z) -> xp.ndarray:
        return Z

    def derivative(self, Z):
        return 1


class ReLU(ActivationFunction):

    def __call__(self, Z) -> xp.ndarray:
        return xp.maximum(0, Z)

    def derivative(self, A) -> xp.ndarray:
        return (A > 0).astype(float)


class SoftMax(ActivationFunction):

    def __call__(self, Z) -> xp.ndarray:
        eZ = xp.exp(Z - Z.max(axis=1, keepdims=True))
        return eZ / xp.sum(eZ, axis=1, keepdims=True)

    def derivative(self, A: xp.ndarray):
        I = xp.eye(A.shape[1])
        idx, idy = xp.diag_indices(I)
        return A * (A[..., None] - I[None, ...])[:, idx, idy]


class OnePlus(ActivationFunction):

    type = "oneplus"

    def __call__(self, Z):
        return 1. + xp.log(1. + xp.exp(Z))

    def derivative(self, Z: xp.ndarray):
        eZ = xp.exp(Z)
        return eZ / (eZ + 1.)


linear = Linear()
sigmoid = Sigmoid()
tanh = Tanh()
relu = ReLU()
softmax = SoftMax()
oneplus = OnePlus()


def get(activation: str):
    return {"sigmoid": sigmoid, "tanh": tanh,
            "linear": linear, "relu": relu,
            "softmax": softmax}[activation]
