from .core import LayerBase, NoParamMixin
from ..util import white, zX, zX_like


class PoolLayer(LayerBase, NoParamMixin):

    def __init__(self, fdim):
        LayerBase.__init__(self, activation="linear", trainable=False)
        from ..numbaops.lltensor import MaxPoolOp
        self.fdim = fdim
        self.filter = None
        self.op = MaxPoolOp()

    def connect(self, to, inshape):
        ic, iy, ix = inshape
        LayerBase.connect(self, to, inshape)
        self.output = zX(ic, iy // self.fdim, ix // self.fdim)

    def feedforward(self, questions):
        """
        Implementation of a max pooling layer.

        :param questions: numpy.ndarray, a batch of outsize from the previous layer
        :return: numpy.ndarray, max pooled batch
        """
        self.output, self.filter = self.op.apply(questions, self.fdim)
        return self.output

    def backpropagate(self, error):
        """
        Calculates the error of the previous layer.
        :param error:
        :return: numpy.ndarray, the errors of the previous layer
        """
        if self.position > 1:
            return self.op.backward(error, self.filter)

    @property
    def outshape(self):
        return self.output.shape[-3:]

    def capsule(self):
        return LayerBase.capsule(self) + [self.fdim]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(fdim=capsule[-1])

    def __str__(self):
        return "Pool-{}x{}".format(self.fdim, self.fdim)


class ConvLayer(LayerBase):

    def __init__(self, nfilters, filterx, filtery, activation="linear", mode="valid", **kw):

        LayerBase.__init__(self, activation=activation, **kw)

        self.nfilters = nfilters
        self.fx = filterx
        self.fy = filtery
        self.depth = 0
        self.stride = 1
        self.mode = mode

        self.inshape = None

        self.op = None

    def connect(self, to, inshape):
        from ..numbaops.lltensor import ConvolutionOp

        LayerBase.connect(self, to, inshape)
        depth, iy, ix = inshape
        self.op = ConvolutionOp()
        self.inshape = inshape
        self.depth = depth
        self.weights = white(self.fx, self.fy, self.depth, self.nfilters)
        self.biases = zX(self.nfilters)
        self.nabla_b = zX_like(self.biases)
        self.nabla_w = zX_like(self.weights)

    def feedforward(self, X):
        self.inputs = X
        self.output = self.activation(self.op.apply(X, self.weights, self.mode))
        return self.output

    def backpropagate(self, error):
        """

        :param error: 4D tensor: (m, filter_number, x, y)
        :return:
        """

        error *= self.activation.derivative(self.output)
        # ishp (im, ic, iy, ix)
        # fshp (fx, fy, fc, nf)
        # eshp (im, nf, oy, ox) = oshp
        # er.T (ox, oy, nf, im)
        iT = self.inputs.transpose(1, 0, 2, 3)
        eT = error.T.transpose(0, 1, 3, 2)
        self.nabla_w = self.op.apply(iT, eT, mode="valid")
        # self.nabla_b = error.sum()  # TODO: why is this commented out???
        rW = self.weights[::-1, ::-1, :, :].transpose(0, 1, 3, 2)
        backpass = self.op.apply(error, rW, "full")
        return backpass

    @property
    def outshape(self):
        oy, ox = tuple(ix - fx + 1 for ix, fx in
                       zip(self.inshape[-2:], (self.fx, self.fy)))
        return self.nfilters, ox, oy

    def capsule(self):
        return LayerBase.capsule(self) + [self.mode, self.activation, self.get_weights(unfold=False)]

    @classmethod
    def from_capsule(cls, capsule):
        nF, depth, fx, fy = capsule[-1][0].shape
        return cls(nF, fx, fy, activation=capsule[-2], mode=capsule[-3], trainable=capsule[1])

    def __str__(self):
        return "Conv({}x{}x{})-{}".format(self.nfilters, self.fx, self.fy, str(self.activation)[:4])
