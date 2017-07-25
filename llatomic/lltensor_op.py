import numpy as np
import numba as nb

from ._llutil import nbfloatX, Xd

intX = nb.typeof(int())


@nb.jit("{f3}({f4},{f4})".format(f3=Xd(3), f4=Xd(4)),
        nopython=True)
def _reshape_receptive_fields(A, F):
    im, ic, iy, ix = A.shape
    # fx, fy, fc, nf = F.shape
    nf, fc, fy, fx = F.shape
    oy, ox = iy - fy + 1, ix - fx + 1
    recfield_size = fx*fy*fc
    rfields = np.zeros((im, oy * ox, recfield_size), dtype=nbfloatX)
    for m in range(im):
        for sy in range(oy):
            for sx in range(ox):
                rfields[m, sy * ox + sx] = A[m, :, sy:sy + fy, sx:sx + fx].ravel()
    return rfields


@nb.jit("{f3}({f4},{f4})".format(f3=Xd(3), f4=Xd(4)),
        nopython=True)
def correlate(A, F):
    im, ic, iy, ix = A.shape
    nf, fc, fy, fx = F.shape
    # fx, fy, fc, nf = F.shape
    oy, ox = iy - fy + 1, ix - fx + 1
    rfields = _reshape_receptive_fields(A, F)
    # output = np.zeros((im, oy*ox, nf), dtype=nbfloatX)
    Frsh = F.reshape(nf, fx * fy * fc)

    output = np.zeros((im, oy*ox, nf))
    for m in range(im):
        output[m] = np.dot(rfields[m], Frsh.T)

    return output


@nb.jit("{f1}({f4},{i})".format(f1=Xd(1), f4=Xd(4), i=intX),
        nopython=True)
def maxpool(A, fdim):
    im, ic, iy, ix = A.shape
    oy, ox = iy // fdim, ix // fdim
    output = np.zeros((im, ic, oy, ox), dtype=nbfloatX)
    filt = np.zeros_like(A, dtype=nbfloatX)
    for m in range(im):
        for c in range(ic):
            for y, sy in enumerate(range(0, iy, fdim)):
                for x, sx in enumerate(range(0, ix, fdim)):
                    recfield = A[m, c, sy:sy + fdim, sx:sx + fdim]
                    value = recfield.max()
                    output[m, c, y, x] = value
                    filterfield = np.equal(recfield, value)
                    filt[m, c, sy:sy + fdim, sx:sx + fdim] += filterfield
    return np.concatenate((output.ravel(), filt.ravel()))


@nb.jit(nopython=True)
def inflate(A, filt):
    em, ec, ey, ex = A.shape
    fm, fc, fy, fx = filt.shape
    fdim = fy // ey
    for m in range(em):
        for c in range(ec):
            for i, y in enumerate(range(0, fy, fdim)):
                for j, x in enumerate(range(0, fx, fdim)):
                    filt[m, c, y:y + fdim, x:x + fdim] *= A[m, c, i, j]
    return filt


class ConvolutionOp:

    @staticmethod
    def valid(A, F):
        return correlate(A, F)

    @staticmethod
    def full(A, F):
        # fx, fy, fc, nf = F.shape
        nf, fc, fy, fx = F.shape
        py, px = fy - 1, fx - 1
        pA = np.pad(A, pad_width=((0, 0), (0, 0), (py, py), (px, px)),
                    mode="constant", constant_values=0.)
        return correlate(pA, F)

    def forward(self, A, F, mode="valid"):
        if not A.flags["C_CONTIGUOUS"]:
            A = A.copy()
        if not F.flags["C_CONTIGUOUS"]:
            F = F.copy()
        Z = self.valid(A, F) if mode == "valid" else self.full(A, F)
        oc, oy, ox = self.outshape(A.shape, F.shape, mode=mode)
        # Z: [im, oy*ox, nf]
        return Z.transpose((0, 2, 1)).reshape(A.shape[0], oc, oy, ox)

    @staticmethod
    def outshape(inshape, fshape, mode="valid"):
        ic, iy, ix = inshape[-3:]
        # fx, fy, fc, nf = fshape
        nf, fc, fy, fx = fshape
        if mode == "valid":
            return nf, iy - fy + 1, ix - fx + 1
        else:
            return nf, iy + fy - 1, ix + fx - 1

    def __str__(self):
        return "Convolution"


class MaxPoolOp:

    def __init__(self):
        self.backward = inflate

    def __str__(self):
        return "MaxPool"

    @staticmethod
    def forward(A, fdim):
        if not A.flags["C_CONTIGUOUS"]:
            A = A.copy()
        im, ic, iy, ix = A.shape
        oy, ox = iy // fdim, ix // fdim
        outarr = maxpool(A, fdim)
        output = outarr[:-A.size].reshape(im, ic, oy, ox)
        filt = outarr[-A.size:].reshape(A.shape)
        return output, filt

    @staticmethod
    def outshape(inshape, fdim):
        if len(inshape) == 3:
            m, iy, ix = inshape
            return m, iy // fdim, ix // fdim
        elif len(inshape) == 2:
            iy, ix = inshape
            return iy // fdim, ix // fdim
