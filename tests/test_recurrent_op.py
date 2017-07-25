import unittest

import numpy as np

from brainforge.layers import RLayer
from brainforge.atomic.recurrent_op import RecurrentOp

# batch, time, dim
INDIM = (20, 5, 10)
M, T, D = INDIM
NEURONS = 30


class TestRecurrentOp(unittest.TestCase):

    def test_equality_of_outputs(self):
        X = np.random.randn(*INDIM)
        E = np.random.randn(M, T, NEURONS)
        W = np.random.randn(INDIM[-1] + NEURONS, NEURONS)
        b = np.random.randn(NEURONS)

        lr = RLayer(NEURONS, activation="tanh", return_seq=True)
        lr.weights = W
        lr.biases = b
        op = RecurrentOp("tanh")

        lrO = lr.feedforward(X)
        lrZ = np.stack(lr.Zs)
        opO, opZ = op.forward(X.transpose((1, 0, 2)), W, b)
        self.assertTrue(np.allclose(lrO, opO.transpose(1, 0, 2)))
        self.assertTrue(np.allclose(lrZ, opZ))

        lrE = lr.backpropagate(E)
        opE, nW, nb = op.backward(opZ, opO, E.transpose((1, 0, 2)), W)
        self.assertTrue(np.allclose(lrE, opE.transpose(1, 0, 2)))

