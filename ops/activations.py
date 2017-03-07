import numpy as np


class ActivationFunction:

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def derivative(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(ActivationFunction):

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return np.divide(1.0, np.add(1, np.exp(-Z)))

    def __str__(self): return "sigmoid"

    def derivative(self, A) -> np.ndarray:
        return A * np.subtract(1.0, A)


class Tanh(ActivationFunction):

    def __call__(self, Z) -> np.ndarray:
        return np.tanh(Z)

    def __str__(self): return "tanh"

    def derivative(self, A) -> np.ndarray:
        return np.subtract(1.0, np.square(A))


class Linear(ActivationFunction):

    def __call__(self, Z) -> np.ndarray:
        return Z

    def __str__(self): return "linear"

    def derivative(self, Z) -> np.ndarray:
        return np.ones_like(Z)


class ReLU(ActivationFunction):

    def __call__(self, Z) -> np.ndarray:
        return np.maximum(0.0, Z)

    def __str__(self): return "relu"

    def derivative(self, A) -> np.ndarray:
        d = np.ones_like(A)
        d[A <= 0.] = 0.
        return d


class SoftMax(ActivationFunction):

    def __call__(self, Z) -> np.ndarray:
        # nZ = Z - np.max(Z)
        eZ = np.exp(Z)
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    def __str__(self): return "softmax"

    def derivative(self, A: np.ndarray) -> np.ndarray:
        """This has to be replaced by a linear backward pass"""
        return 1.


act_fns = {key.lower(): cls for key, cls in locals().items() if "Function" not in key}
