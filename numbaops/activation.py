from ._llactivation import (
    sigmoid, sigmoid_p,
    tanh, tanh_p,
    sqrt, sqrt_p,
    linear, linear_p,
    relu, relu_p,
    softmax, softmax_p,
    s1
)


class ActivationFunction:

    type = ""

    def __call__(self, Z):
        raise NotImplementedError

    def __str__(self):
        return self.type

    def derivative(self, Z):
        raise NotImplementedError


class Sigmoid(ActivationFunction):

    type = "sigmoid"
    __call__ = sigmoid
    derivative = sigmoid_p


class Tanh(ActivationFunction):

    type = "tanh"
    __call__ = tanh
    derivative = tanh_p


class Sqrt(ActivationFunction):

    type = "sqrt"
    __call__ = sqrt
    derivative = sqrt_p


class Linear(ActivationFunction):

    type = "linear"
    __call__ = linear
    derivative = linear_p


class ReLU(ActivationFunction):

    type = "relu"
    __call__ = relu
    derivative = relu_p


class SoftMax(ActivationFunction):

    type = "softmax"
    __call__ = softmax
    true_derivative = softmax_p

    def derivative(self, A):
        return s1


act_fns = {"sigmoid": Sigmoid, "tanh": Tanh, "sqrt": Sqrt,
           "linear": Linear, "relu": ReLU, "softmax": SoftMax}
