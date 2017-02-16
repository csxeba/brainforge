import numpy as np


class _ActivationFunctionBase:

    def __call__(self, Z: np.ndarray): pass

    def __str__(self): raise NotImplementedError

    def derivative(self, Z: np.ndarray):
        raise NotImplementedError


class Sigmoid(_ActivationFunctionBase):

    def __call__(self, Z: np.ndarray):
        return np.divide(1.0, np.add(1, np.exp(-Z)))

    def __str__(self): return "sigmoid"

    def derivative(self, A):
        return A * np.subtract(1.0, A)


class Tanh(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    def derivative(self, A):
        return np.subtract(1.0, np.square(A))


class Linear(_ActivationFunctionBase):

    def __call__(self, Z):
        return Z

    def __str__(self): return "linear"

    def derivative(self, Z):
        return np.ones_like(Z)


class ReLU(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.maximum(0.0, Z)

    def __str__(self): return "relu"

    def derivative(self, A):
        d = np.greater(A, 0.0).astype("float32")
        return d


class SoftMax(_ActivationFunctionBase):

    def __call__(self, Z):
        # nZ = Z - np.max(Z)
        eZ = np.exp(Z)
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    def __str__(self): return "softmax"

    def derivative(self, A: np.ndarray):
        """This has to be replaced by a linear backward pass"""
        raise NotImplementedError


act_fns = {key.lower(): cls for key, cls in locals().items() if key[0] not in "_"}
