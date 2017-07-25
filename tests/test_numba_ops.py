import unittest

import numpy as np
from matplotlib import pyplot as plt

from brainforge.atomic import (
    ConvolutionOp as NpConv,
    MaxPoolOp as NpPool,
    DenseOp as NpDense
)
from brainforge.llatomic import (
    ConvolutionOp as NbConv,
    MaxPoolOp as NbPool,
    DenseOp as NbDense
)


VISUAL = False


class TestNumbaTensorOps(unittest.TestCase):

    def test_convolution_op(self):
        npop = NpConv()
        nbop = NbConv()

        A = np.random.uniform(size=(1, 1, 12, 12))
        F = np.random.uniform(size=(1, 1, 3, 3))

        npO = npop.apply(A, F, mode="full")
        nbO = nbop.apply(A, F, mode="full")

        self.assertTrue(np.allclose(npO, nbO))

        if VISUAL:
            visualize(A, npO, nbO, supt="Testing Convolutions")

    def test_pooling_op(self):
        npop = NpPool()
        nbop = NbPool()

        A = np.random.uniform(0., 1., (1, 1, 12, 12))

        npO, npF = npop.apply(A, 2)
        nbO, nbF = nbop.apply(A, 2)

        npbF = npop.backward(npO, npF)
        nbbF = nbop.backward(nbO, nbF)

        self.assertTrue(np.allclose(npF, nbF))
        self.assertTrue(np.allclose(npbF, nbbF))
        self.assertTrue(np.allclose(npO, nbO))

        if VISUAL:
            visualize(A, npO, nbO, supt="Testing Pooling")

    def test_dense_op(self):
        npop = NpDense()
        nbop = NbDense()

        A = np.random.uniform(size=(12, 12))
        W = np.random.uniform(size=(12, 12))
        b = np.random.uniform(size=(12,))

        npO = npop.forward(A, W, b)
        nbO = nbop.forward(A, W, b)

        self.assertTrue(np.allclose(npO, nbO))
        if VISUAL:
            visualize(A[None, None, ...], npO[None, None, ...], nbO[None, None, ...], "Testing Dense")


def visualize(A, O1, O2, supt=None):
    d = O1 - O2
    vmax, vmin = max(O1.max(), O2.max()), min(O1.min(), O2.min())
    fig, axarr = plt.subplots(2, 2)
    axarr[0][0].imshow(A[0, 0], vmin=0, vmax=1, cmap="autumn")
    axarr[0][0].set_title("A")
    axarr[0][1].imshow(d[0, 0], cmap="seismic")
    axarr[0][1].set_title("d")
    axarr[1][0].imshow(O1[0, 0], vmin=vmin, vmax=vmax, cmap="hot")
    axarr[1][0].set_title("npO")
    axarr[1][1].imshow(O2[0, 0], vmin=vmin, vmax=vmax, cmap="hot")
    axarr[1][1].set_title("nbO")
    plt.suptitle(supt)
    plt.tight_layout()
    plt.show()
