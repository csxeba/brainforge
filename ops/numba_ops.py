import numpy as np
import numba as nb


@nb.jitclass
class FlattenOp:

    def apply(self, A):
        return A.reshape(A.shape[0], np.prod(A.shape[1:]))

    def outshape(self, inshape=None):
        return np.prod(inshape),

    def __str__(self):
        return "Flatten"


@nb.jitclass(spec=())
class ConvolutionOp:

    def __init__(self):
        pass

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


@nb.jitclass(spec=())
class MaxPoolOp:

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
