import numpy as np
import numba as nb


floatX = nb.typeof(float())
intX = nb.typeof(int())
far_4D = "{type}[:, :, :, :]".format(type=floatX)
f4f4_f4 = ("f4({f4},{f4})"
           .format(f4=far_4D))


@nb.jit(signature_or_function=f4f4_f4, nopython=True)
def convvalid(A, Frsh, O):
    im, ic, iy, ix = A.shape
    nf, recfield_size = Frsh.shape
    oy, ox = O.shape[-2:]
    rfield = np.zeros((oy * ox, recfield_size))
    # rfields = np.zeros((im, oy * ox, recfield_size))
    output = np.zeros((im, nf, oy, ox))

    for m in range(im):
        for sy in range(oy):
            for sx in range(ox):
                rfield[sy * ox + sx] = A[m, :, sy:sy + fy, sx:sx + fx].ravel()
                # rfields[m, sy * ox + sx] = A[m, :, sy:sy + fy, sx:sx + fx].ravel()
        output[m] = np.dot(rfield, Frsh.T)

    # mul = np.matmul(rfields, F.reshape(nf, recfield_size).T).transpose(0, 2, 1)
    # out = mul.reshape(im, nf, oy, ox)
    return output


@nb.jit(signature_or_function=f4f4_f4, nopython=True)
def convfull(A, F):
    nf, fc, fy, fx = F.shape
    py, px = fy - 1, fx - 1
    pA = np.pad(A, pad_width=((0, 0), (0, 0), (py, py), (px, px)),
                mode="constant", constant_values=0.)
    return convvalid(pA, F)


@nb.jit(signature_or_function="{t}[:]({f4}, {ints})"
        .format(f4=far_4D, t=floatX, ints=nb.typeof(int())),
        nopython=True)
def maxpool(A, fdim):
    m, ch, iy, ix = A.shape
    oy, ox = iy // fdim, ix // fdim
    output = np.zeros((m * ch * oy * ox,))
    filt = np.zeros_like(A)
    counter = 0
    for i in range(len(A)):
        pic = A[i]
        for c in range(len(pic)):
            sheet = pic[c]
            for y, sy in enumerate(range(0, iy, fdim)):
                for x, sx in enumerate(range(0, ix, fdim)):
                    recfield = sheet[sy:sy + fdim, sx:sx + fdim]
                    value = recfield.max()
                    output[counter] = value
                    ffield = np.equal(recfield, value)
                    filt[i, c, sy:sy + fdim, sx:sx + fdim] += ffield
    return np.concatenate((output, filt.ravel()))


@nb.jit(signature_or_function=f4f4_f4,
        nopython=True)
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
