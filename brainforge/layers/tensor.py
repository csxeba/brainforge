import numpy as np

from .abstract_layer import LayerBase, NoParamMixin
from ..util import zX, zX_like, white, scalX


class PoolLayer(NoParamMixin, LayerBase):

    def __init__(self, filter_size, compiled=True):
        LayerBase.__init__(self, activation="linear", trainable=False)
        if compiled:
            from ..llatomic.lltensor_op import MaxPoolOp
        else:
            from ..atomic import MaxPoolOp
        self.fdim = filter_size
        self.filter = None
        self.op = MaxPoolOp()

    def connect(self, brain):
        ic, iy, ix = brain.outshape[-3:]
        if any((iy % self.fdim, ix % self.fdim)):
            raise RuntimeError(
                "Incompatible shapes: {} % {}".format((ix, iy), self.fdim)
            )
        LayerBase.connect(self, brain)
        self.output = zX(ic, iy // self.fdim, ix // self.fdim)

    def feedforward(self, questions):
        self.output, self.filter = self.op.forward(questions, self.fdim)
        return self.output

    def backpropagate(self, delta):
        return self.op.backward(delta, self.filter)

    @property
    def outshape(self):
        return self.output.shape[-3:]

    def __str__(self):
        return "Pool-{}x{}".format(self.fdim, self.fdim)


class ConvLayer(LayerBase):

    def __init__(self, nfilters, filterx=3, filtery=3, compiled=True, **kw):
        super().__init__(compiled=compiled, **kw)
        self.nfilters = nfilters
        self.fx = filterx
        self.fy = filtery
        self.depth = 0
        self.stride = 1
        self.inshape = None
        self.op = None

    def connect(self, brain):
        if self.compiled:
            from ..llatomic import ConvolutionOp
        else:
            from ..atomic import ConvolutionOp
        depth, iy, ix = brain.outshape[-3:]
        if any((iy < self.fy, ix < self.fx)):
            raise RuntimeError(
                "Incompatible shapes: iy ({}) < fy ({}) OR ix ({}) < fx ({})"
                .format(iy, self.fy, ix, self.fx)
            )
        super().connect(brain)
        self.op = ConvolutionOp()
        self.inshape = brain.outshape
        self.depth = depth
        self.weights = white(self.nfilters, self.depth, self.fx, self.fy)
        self.biases = zX(self.nfilters)[None, :, None, None]
        self.nabla_b = zX_like(self.biases)
        self.nabla_w = zX_like(self.weights)

    def feedforward(self, X):
        self.inputs = X
        self.output = self.activation.forward(self.op.forward(X, self.weights, "valid"))
        self.output += self.biases
        return self.output

    def backpropagate(self, delta):
        delta *= self.activation.backward(self.output)
        self.nabla_w, self.nabla_b, dX = self.op.backward(X=self.inputs, E=delta, F=self.weights)
        return dX

    @property
    def outshape(self):
        oy, ox = tuple(ix - fx + 1 for ix, fx in zip(self.inshape[-2:], (self.fx, self.fy)))
        return self.nfilters, ox, oy

    def __str__(self):
        return "Conv({}x{}x{})-{}".format(self.nfilters, self.fy, self.fx, str(self.activation)[:4])


class GlobalAveragePooling(NoParamMixin, LayerBase):

    def __init__(self):
        LayerBase.__init__(self)
        NoParamMixin.__init__(self)
        self.repeats = 0

    def feedforward(self, X):
        self.repeats = np.prod(X.shape[2:])
        return X.mean(axis=(2, 3))

    def backpropagate(self, delta):
        m = len(delta)
        delta = np.repeat(delta / scalX(self.repeats), self.repeats)
        delta = delta.reshape((m,) + self.inshape)
        return delta

    @property
    def outshape(self):
        return self.inshape[0],
