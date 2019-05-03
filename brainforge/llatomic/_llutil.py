import numba as nb

from brainforge.config import floatX

nbfloatX = nb.float32 if floatX == "float32" else nb.float64


def Xd(X, t=nbfloatX):
    return "{t}[{0}]".format(",".join([":"]*(X-1) + ["::1"]), t=t)
