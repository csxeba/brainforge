from brainforge import backend as xp
from .abstract_layer import Layer, NoParamMixin


class DropOut(NoParamMixin, Layer):

    def __init__(self, dropchance):
        super().__init__()
        self.dropchance = 1. - dropchance
        self.mask = None
        self.inshape = None
        self.training = True

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        if self.brain.learning:
            self.inputs = x
            self.mask = xp.random.uniform(0, 1, self.inshape) < self.dropchance  # type: xp.ndarray
            mask = self.mask
        else:
            mask = self.dropchance

        self.output = x * mask
        return self.output

    def backward(self, error: xp.ndarray) -> xp.ndarray:
        return error * self.mask

    @property
    def outshape(self):
        return self.inshape
