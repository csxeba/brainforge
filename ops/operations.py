"""Wrappers for vector-operations and other functions"""
import numpy as np


class Flatten:

    def __init__(self):
        from ..util import rtm
        self.op = rtm

    def __str__(self):
        return "Flatten"

    def __call__(self, A):
        return self.op(A)

    def outshape(self, inshape=None):
        return np.prod(inshape),  # return as tuple!


class Reshape:

    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return "Reshape"

    def __call__(self, A):
        m = A.shape[0]
        return A.reshape(m, *self.shape)

    def outshape(self, inshape=None):
        return self.shape


class Convolution:

    def valid(self, A, F):
        im, ic, iy, ix = A.shape
        nf, fc, fy, fx = F.shape
        recfield_size = fx * fy * fc
        oy, ox = (iy - fy) + 1, (ix - fx) + 1
        rfields = np.zeros((im, oy*ox, recfield_size))

        if fc != ic:
            err = "Supplied filter (F) is incompatible with supplied input! (X)\n"
            err += "input depth: {} != {} :filter depth".format(ic, fc)
            raise ValueError(err)

        for i, pic in enumerate(A):
            for sy in range(oy):
                for sx in range(ox):
                    rfields[i][sy*ox + sx] = pic[:, sy:sy+fy, sx:sx+fx].ravel()

        output = np.matmul(rfields, F.reshape(nf, recfield_size).T)
        output = output.transpose(0, 2, 1).reshape(im, nf, oy, ox)
        return output

    def full(self, A, F):
        nf, fc, fy, fx = F.shape
        py, px = fy - 1, fx - 1
        pA = np.pad(A, pad_width=((0, 0), (0, 0), (py, py), (px, px)),
                    mode="constant", constant_values=0.)
        return self.valid(pA, F)

    def apply(self, A, F, mode="valid"):
        if mode == "valid":
            return self.valid(A, F)
        return self.full(A, F)

    def outshape(self, inshape, fshape, mode="valid"):
        if mode == "valid":
            return tuple(ix - fx + 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        elif mode == "full":
            return tuple(ix + fx - 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        else:
            raise RuntimeError("Unsupported mode:", mode)

    def __str__(self):
        return "Convolution"


class ScipySigConv:

    def __init__(self):
        from scipy.signal import convolve
        self.op = convolve

    def __str__(self):
        return "scipy.signal.convolution"

    def apply(self, A, F, mode="valid"):
        m, ic, iy, ix = A.shape
        nf, fc, fy, fx = F.shape
        ox, oy = self.outshape(A.shape, F.shape, mode)

        assert ic == fc, "Number of channels got messed up"

        if mode == "valid":
            out = np.zeros((m, nf, 1, ox, oy))
            for i, pic in enumerate(A):
                for j, filt in enumerate(F):
                    conved = self.op(pic, filt, mode=mode)
                    out[i, j] = conved
            return out[:, :, 0, :, :]

        out = np.zeros((m, nf, ox, oy))
        for i, pic in enumerate(A):
            for j, filt in enumerate(F):
                for c in range(fc):
                    out[i, j] += self.op(pic[c], filt[c], mode=mode)
        return out

    @staticmethod
    def outshape(inshape, fshape, mode="valid"):
        if mode == "valid":
            return tuple(ix - fx + 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        elif mode == "full":
            return tuple(ix + fx - 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        else:
            raise RuntimeError("Unsupported mode: " + str(mode))


class MaxPool:

    def __init__(self):
        pass

    def __str__(self):
        return "MaxPool"

    def apply(self, A, fdim):
        m, ch, iy, ix = A.shape
        oy, ox = iy // fdim, ix // fdim
        output = np.zeros((m, ch, oy, ox))
        filt = np.zeros((m, ch, iy, ix))
        for i, pic in enumerate(A):
            for c, sheet in enumerate(pic):
                for y, sy in enumerate(range(0, iy, fdim)):
                    for x, sx in enumerate(range(0, ix, fdim)):
                        recfield = sheet[sy:sy+fdim, sx:sx+fdim]
                        value = recfield.max()
                        output[i, c, y, x] = value
                        ffield = np.equal(recfield, value)
                        filt[i, c, sy:sy+fdim, sx:sx+fdim] += ffield
        return output, filt

    def backward(self, E, filt):
        em, ec, ey, ex = E.shape
        fm, fc, fy, fx = filt.shape
        fdim = fy // ey
        for m in range(em):
            for c in range(ec):
                for i, y in enumerate(range(0, fy, fdim)):
                    for j, x in enumerate(range(0, fx, fdim)):
                        filt[m, c, y:y+fdim, x:x+fdim] *= E[m, c, i, j]
        return filt

    def outshape(self, inshape, fdim):
        if len(inshape) == 3:
            m, iy, ix = inshape
            return m, iy // fdim, ix // fdim
        elif len(inshape) == 2:
            iy, ix = inshape
            return iy // fdim, ix // fdim


class _ActivationFunctionBase:

    def __call__(self, Z: np.ndarray): pass

    def __str__(self): raise NotImplementedError

    def derivative(self, Z: np.ndarray):
        raise NotImplementedError

    def backwards(self, A):
        return self.derivative(A)

    @property
    def outshape(self):
        return self._inshape


class _Sigmoid(_ActivationFunctionBase):

    def __call__(self, Z: np.ndarray):
        return np.divide(1.0, np.add(1, np.exp(-Z)))

    def __str__(self): return "sigmoid"

    def derivative(self, A):
        return np.multiply(A, np.subtract(1.0, A))


class _Tanh(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    def derivative(self, A):
        return np.subtract(1.0, np.square(A))


class _Linear(_ActivationFunctionBase):

    def __call__(self, Z):
        return Z

    def __str__(self): return "linear"

    def derivative(self, Z):
        return np.ones_like(Z)


class _ReLU(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.maximum(0.0, Z)

    def __str__(self): return "relu"

    def derivative(self, A):
        d = np.greater(A, 0.0).astype("float32")
        return d


class _SoftMax(_ActivationFunctionBase):

    def __call__(self, Z):
        eZ = np.exp(Z)
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    def __str__(self): return "softmax"

    def derivative(self, A: np.ndarray):
        # This is the negative of the outer product of the last axis with itself
        J = A[..., None] * A[:, None, :]  # given by -a_i*a_j, where i =/= j
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = A * (1. - A)  # given by a_i(1 - a_j), where i = j
        return J.sum(axis=1)  # sum for each sample


sigmoid = _Sigmoid()
tanh = _Tanh()
linear = _Linear()
relu = _ReLU()
softmax = _SoftMax()

act_fns = {key.lower(): fn for key, fn in locals().items() if key[0] not in ("_", "F")}
