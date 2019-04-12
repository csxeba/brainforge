from brainforge import backend as xp

from brainforge.atomic.activation import OnePlus


oneplus = OnePlus()


class DNC:

    def __init__(self, controller, reads):
        self.ctrl = controller
        self.memory = None
        self.usage = None
        self.link = None
        self.reads = xp.zeros(())

    def forward_step(self, x, reads):
        Z = xp.concatenate((x, reads.flat))

    def feedforward(self, X):
        pass
