import numpy as np
import numba as nb


floatX = nb.typeof(float())
f4f4_f4 = ("({t}[:, :, :, :],{t}[:, :, :, :])({t}[:, :, :, :])"
           .format(t=floatX))


@nb.jit(nopython=True)
def convvalid(A, F):
    im, ic, iy, ix = A.shape
    nf, fc, fy, fx = F.shape
    recfield_size = fx * fy * fc
    oy, ox = (iy - fy) + 1, (ix - fx) + 1
    rfields = np.zeros((im, oy * ox, recfield_size))

    if fc != ic:
        err = "Supplied filter (F) is incompatible with supplied input! (X)\n" \
              + "input depth: {} != {} :filter depth".format(ic, fc)
        raise ValueError(err)

    for i, pic in enumerate(A):
        for sy in range(oy):
            for sx in range(ox):
                rfields[i][sy * ox + sx] = pic[:, sy:sy + fy, sx:sx + fx].ravel()

    return (np.matmul(rfields, F.reshape(nf, recfield_size).T)
            .transpose(0, 2, 1).reshape(im, nf, oy, ox))


@nb.jit(nopython=True)
def convfull(A, F):
    nf, fc, fy, fx = F.shape
    py, px = fy - 1, fx - 1
    pA = np.pad(A, pad_width=((0, 0), (0, 0), (py, py), (px, px)),
                mode="constant", constant_values=0.)
    return convvalid(pA, F)

spec = "({t},int32)({t})".format(t="{}[:, :, :, :]".format(floatX))
@nb.jit(nopython=True)
def maxpool(A, fdim):
    m, ch, iy, ix = A.shape
    oy, ox = iy // fdim, ix // fdim
    output = np.zeros((m * ch * oy * ox,))
    filt = np.zeros_like(A)
    counter = 0
    for i, pic in enumerate(A):
        for c, sheet in enumerate(pic):
            for y, sy in enumerate(range(0, iy, fdim)):
                for x, sx in enumerate(range(0, ix, fdim)):
                    recfield = sheet[sy:sy + fdim, sx:sx + fdim]
                    value = recfield.max()
                    output[counter] = value
                    ffield = np.equal(recfield, value)
                    filt[i, c, sy:sy + fdim, sx:sx + fdim] += ffield
    return np.concatenate((output, filt.ravel()))


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

    def __init__(self):
        self.valid = convvalid
        self.full = convfull

    def apply(self, A, F, mode="valid"):
        if mode == "valid":
            return self.valid(A, F)
        return self.full(A, F)

    @staticmethod
    def outshape(inshape, fshape, mode="valid"):
        if mode == "valid":
            return tuple(ix - fx + 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        elif mode == "full":
            return tuple(ix + fx - 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        else:
            raise RuntimeError("Unsupported mode:", mode)

    def __str__(self):
        return "Convolution"


class MaxPoolOp:

    def __init__(self):
        self.backward = inflate

    def __str__(self):
        return "MaxPool"

    @staticmethod
    def apply(A, fdim):
        m, ch, iy, ix = A.shape
        oy, ox = iy // fdim, ix // fdim
        outarr = maxpool(A, fdim)
        output = outarr[:-A.size].reshape(m, ch, oy, ox)
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
