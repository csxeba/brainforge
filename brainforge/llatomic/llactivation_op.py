import abc

from ._llactivation import (
    sigmoid, sigmoid_p,
    tanh, tanh_p,
    relu, relu_p,
    # softmax, softmax_p,
)


class LLActivation(abc.ABC):
    type = ""

    def __init__(self):
        self.llact, self.llactp = {
            "sigmoid": (sigmoid, sigmoid_p),
            "tanh": (tanh, tanh_p),
            "relu": (relu, relu_p)
        }[self.type]

    def forward(self, X):
        return self.llact(X.ravel()).reshape(X.shape)

    def backward(self, A):
        return self.llactp(A.ravel()).reshape(A.shape)


class Sigmoid(LLActivation):
    type = "sigmoid"


class Tanh(LLActivation):
    type = "tanh"


class Sqrt(LLActivation):
    type = "sqrt"


class ReLU(LLActivation):
    type = "relu"


# class SoftMax:
#     type = "softmax"
#     forward = softmax
#     true_backward = softmax_p
#
#     def backward(self, A):
#         return s1


llactivations = {"sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU}
