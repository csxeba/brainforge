import numba as nb

from ..util import floatX, scalX

nbfloatX = nb.float32 if floatX == "float32" else nb.float64
