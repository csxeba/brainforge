"""Wrappers for vector-operations and other functions"""
import abc

import numpy as np

from ..util import zX, zX_like
from .activation import Sigmoid


sig = Sigmoid()


class Op(abc.ABC):

    @abc.abstractmethod
    def outshape(self, inshape):
        raise NotImplementedError


class DenseOp(Op):

    def __init__(self, neurons):
        self.neurons = neurons

    @staticmethod
    def forward(X, W, b):
        return np.dot(X, W) + b

    @staticmethod
    def backward(X, E, W):
        gW = np.dot(X.T, E)
        # gb = np.sum(E, axis=1)
        gX = np.dot(E, W.T)
        return gW, gX

    def outshape(self, inshape):
        return self.neurons,


class LSTMOp(Op):

    @staticmethod
    def forward(Xrsh, W, b):
        time, im, idim = Xrsh.shape
        neu = b.shape[0]
        output = zX()
        state = np.zeros_like(output)

        for t in range(self.time):
            Z = np.concatenate((self.inputs[t], output), axis=1)

            preact = Z @ self.weights + self.biases
            preact[:, :self.G] = sigmoid(preact[:, :self.G])
            preact[:, self.G:] = self.activation(preact[:, self.G:])

            f, i, o, cand = np.split(preact, 4, axis=-1)

            state = state * f + i * cand
            state_a = self.activation(state)
            output = state_a * o

            self.Zs.append(Z)
            self.gates.append(preact)
            self.cache.append([output, state_a, state, preact])

        if self.return_seq:
            self.output = np.stack([cache[0] for cache in self.cache], axis=1)
        else:
            self.output = self.cache[-1][0]
        return self.output


class FlattenOp(Op):

    def __init__(self):
        from ..util import rtm
        self.op = rtm

    def __str__(self):
        return "Flatten"

    def __call__(self, A):
        return self.op(A)

    def outshape(self, inshape=None):
        return np.prod(inshape),  # return as tuple!


class ReshapeOp(Op):

    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return "Reshape"

    def __call__(self, A):
        m = A.shape[0]
        return A.reshape(m, *self.shape)

    def outshape(self, *args):
        return self.shape


class ConvolutionOp(Op):

    @staticmethod
    def valid(A, F):
        im, ic, iy, ix = A.shape
        nf, fc, fy, fx = F.shape
        # fx, fy, fc, nf = F.shape
        recfield_size = fx * fy * fc
        oy, ox = iy - fy + 1, ix - fx + 1
        rfields = np.zeros((im, oy*ox, recfield_size))
        Frsh = F.reshape(nf, recfield_size)

        if fc != ic:
            err = "Supplied filter (F) is incompatible with supplied input! (X)\n"
            err += "input depth: {} != {} :filter depth".format(ic, fc)
            raise ValueError(err)

        for i, pic in enumerate(A):
            for sy in range(oy):
                for sx in range(ox):
                    rfields[i][sy*ox + sx] = pic[:, sy:sy+fy, sx:sx+fx].ravel()

        output = np.zeros((im, oy*ox, nf))
        for m in range(im):
            output[m] = np.dot(rfields[m], Frsh.T)

        # output = np.matmul(rfields, F.reshape(nf, recfield_size).T)
        output = output.transpose((0, 2, 1)).reshape(im, nf, oy, ox)
        return output

    @staticmethod
    def full(A, F):
        nf, fc, fy, fx = F.shape
        py, px = fy - 1, fx - 1
        pA = np.pad(A, pad_width=((0, 0), (0, 0), (py, py), (px, px)),
                    mode="constant", constant_values=0.)
        return ConvolutionOp.valid(pA, F)

    @staticmethod
    def apply(A, F, mode="valid"):
        if mode == "valid":
            return ConvolutionOp.valid(A, F)
        return ConvolutionOp.full(A, F)

    @staticmethod
    def outshape(inshape, fshape, mode="valid"):
        ic, iy, ix = inshape[-3:]
        fx, fy, fc, nf = fshape
        if mode == "valid":
            return nf, iy - fy + 1, ix - fx + 1
        elif mode == "full":
            return nf, iy + fy - 1, ix + fx - 1
        else:
            raise RuntimeError("Unsupported mode:", mode)

    def __str__(self):
        return "Convolution"


class MaxPoolOp(Op):

    def __str__(self):
        return "MaxPool"

    @staticmethod
    def apply(A, fdim):
        im, ic, iy, ix = A.shape
        oy, ox = iy // fdim, ix // fdim
        output = zX(im, ic, oy, ox)
        filt = zX_like(A)
        for m in range(im):
            for c in range(ic):
                for y, sy in enumerate(range(0, iy, fdim)):
                    for x, sx in enumerate(range(0, ix, fdim)):
                        recfield = A[m, c, sy:sy+fdim, sx:sx+fdim]
                        value = recfield.max()
                        output[m, c, y, x] = value
                        ffield = np.equal(recfield, value)
                        filt[m, c, sy:sy+fdim, sx:sx+fdim] += ffield
        return output, filt

    @staticmethod
    def backward(E, filt):
        em, ec, ey, ex = E.shape
        fm, fc, fy, fx = filt.shape
        fdim = fy // ey
        for m in range(em):
            for c in range(ec):
                for i, y in enumerate(range(0, fy, fdim)):
                    for j, x in enumerate(range(0, fx, fdim)):
                        filt[m, c, y:y+fdim, x:x+fdim] *= E[m, c, i, j]
        return filt

    @staticmethod
    def outshape(inshape, fdim):
        if len(inshape) == 3:
            m, iy, ix = inshape
            return m, iy // fdim, ix // fdim
        elif len(inshape) == 2:
            iy, ix = inshape
            return iy // fdim, ix // fdim
