from .operations import ReshapeOp
# from .operations import ConvolutionOp, MaxPoolOp
from .numba_ops import ConvolutionOp, MaxPoolOp
from .activations import (Sigmoid, Tanh, ReLU,
                          Linear, SoftMax, act_fns)
