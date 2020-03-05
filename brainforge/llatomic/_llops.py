import numba as nb
import numpy as np

from ._llutil import Xd


@nb.jit("{f2}({f2},{f2},{f1})".format(f1=Xd(1), f2=Xd(2)), nopython=True)
def dense_forward(X, W, b):
    return np.dot(X, W) + b
