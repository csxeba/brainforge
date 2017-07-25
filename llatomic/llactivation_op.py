from ._llactivation import (
    sigmoid, sigmoid_p,
    tanh, tanh_p,
    sqrt, sqrt_p,
    relu, relu_p,
    # softmax, softmax_p,
    s1
)


class Sigmoid:
    type = "sigmoid"
    forward = staticmethod(sigmoid)
    backward = staticmethod(sigmoid_p)


class Tanh:
    type = "tanh"
    forward = staticmethod(tanh)
    backward = staticmethod(tanh_p)


class Sqrt:
    type = "sqrt"
    forward = staticmethod(sqrt)
    backward = staticmethod(sqrt_p)


class Linear:
    type = "linear"
    forward = staticmethod(lambda A: A)
    backward = staticmethod(lambda E: s1)


class ReLU:
    type = "relu"
    forward = staticmethod(relu)
    backward = staticmethod(relu_p)


# class SoftMax:
#     type = "softmax"
#     forward = softmax
#     true_backward = softmax_p
#
#     def backward(self, A):
#         return s1


act_fns = {"sigmoid": Sigmoid, "tanh": Tanh, "sqrt": Sqrt,
           "linear": Linear, "relu": ReLU}
