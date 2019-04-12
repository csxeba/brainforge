from brainforge import backend as xp

from .abstract_layer import Layer, NoParamMixin, Parameterized
from ..util import white, zX, zX_like


class MaxPool(NoParamMixin, Layer):

    def __init__(self, filter_size=2):
        super().__init__()
        self.filter_size = filter_size
        self.mask = None

    def connect(self, brain):
        ic, iy, ix = brain.outshape[-3:]
        if any((iy % self.filter_size, ix % self.filter_size)):
            raise RuntimeError(
                "Incompatible shapes: {} % {}".format((ix, iy), self.filter_size)
            )
        super().connect(brain)

    def forward(self, x: xp.ndarray):
        self.inputs = x
        z = xp.stack([x[:, i::self.filter_size, j::self.filter_size]
                      for i in range(self.filter_size) for j in range(self.filter_size)], axis=0)
        output = xp.max(z, axis=0)
        return self.output

    def backward(self, error):
        return self.op.backward(error, self.filter)

    @property
    def outshape(self):
        return self.output.shape[-3:]


class ConvLayer(Parameterized):

    def __init__(self, nfilters, filterx=3, filtery=3, **kw):

        super().__init__(**kw)
        self.nfilters = nfilters
        self.fx = filterx
        self.fy = filtery
        self.depth = 0
        self.stride = 1
        self.inshape = None
        self.op = None

    def connect(self, brain):
        depth, iy, ix = brain.outshape[-3:]
        if any((iy < self.fy, ix < self.fx)):
            raise RuntimeError(
                "Incompatible shapes: iy ({}) < fy ({}) OR ix ({}) < fx ({})"
                .format(iy, self.fy, ix, self.fx)
            )
        self.inshape = brain.outshape
        self.depth = depth
        self.weights = white(self.nfilters, self.depth, self.fy, self.fx)
        self.biases = zX(self.nfilters)
        super().connect(brain)

    def forward(self, X):
        self.inputs = X
        self.output = self.op.apply(X, self.weights, "valid")
        return self.output

    def backward(self, delta):
        # ishp (im, ic, iy, ix)
        # fshp (fx, fy, fc, nf)
        # eshp (im, nf, oy, ox) = oshp
        # er.T (ox, oy, nf, im)
        input_transposed = self.inputs.transpose(1, 0, 2, 3)
        delta_transposed = delta.transpose(1, 0, 2, 3)
        self.nabla_w = self.op.apply(input_transposed, delta_transposed, mode="valid").transpose(1, 0, 2, 3)
        self.nabla_b = delta.sum(axis=(0, 2, 3))  # TODO: why is this commented out???
        rW = self.weights[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
        backpass = self.op.apply(delta, rW, "full")
        return backpass

    @property
    def outshape(self):
        oy, ox = tuple(ix - fx + 1 for ix, fx in
                       zip(self.inshape[-2:], (self.fx, self.fy)))
        return self.nfilters, ox, oy
