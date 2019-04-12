from brainforge import backend as xp
from matplotlib import pyplot as plt

from brainforge.atomic import (
    ConvolutionOp as NPConv,
    MaxPoolOp as NpPool,
)
from brainforge.numbaops.lltensor import (
    ConvolutionOp as NBConv,
    MaxPoolOp as NbPool,
)


def visualize(A, d, O1, O2, supt=None):
    print("d.mean() =", d.mean())
    if d.mean() == 0.:
        print("Test passed!")
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


def test_convolutions():
    print("Testing ConvOps")
    npop = NPConv()
    nbop = NBConv()
    A = xp.random.uniform(0., 1., (1, 1, 12, 12))
    F = xp.random.uniform(0., 1., (1, 1, 3, 3))

    npO = npop.apply(A, F, mode="full")
    nbO = nbop.apply(A, F, mode="full")

    dC = xp.abs(npO - nbO)

    visualize(A, dC, npO, nbO, supt="ConvTest")


def test_pooling():
    print("Testing MaxPoolOps")
    npop = NpPool()
    nbop = NbPool()

    A = xp.random.uniform(0., 1., (1, 1, 12, 12))

    npO, npF = npop.apply(A, 2)
    nbO, nbF = nbop.apply(A, 2)

    assert xp.allclose(npF, nbF)

    npbF = npop.backward(npO, npF)
    nbbF = nbop.backward(nbO, nbF)

    assert xp.allclose(npbF, nbbF)

    dP = xp.abs(npO - nbO)

    visualize(A, dP, npO, nbO, supt="PoolTest")


if __name__ == '__main__':
    test_pooling()
