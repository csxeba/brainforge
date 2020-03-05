import unittest

import numpy as np

from brainforge.atomic.activation_op import (
    Sigmoid as NpSigm, Tanh as NpTanh, ReLU as NpReLU
)
from brainforge.llatomic.llactivation_op import (
    Sigmoid as NbSigm, Tanh as NbTanh, ReLU as NbReLU
)


np.random.seed(1337)

ops = {
    "sigmoid": (NpSigm(), NbSigm()),
    "tanh": (NpTanh(), NbTanh()),
    "relu": (NpReLU(), NbReLU())
}


class TestActivationFunctions(unittest.TestCase):

    def _run_function_test(self, func):
        npop, nbop = ops[func]
        npO = npop.forward(self.X)
        nbO = nbop.forward(self.X)

        self.assertTrue(np.allclose(npO, nbO))

    def setUp(self):
        self.X = np.random.randn(100)

    def test_sigmoid(self):
        self._run_function_test("sigmoid")

    def test_tanh(self):
        self._run_function_test("tanh")

    def test_relu(self):
        self._run_function_test("relu")
