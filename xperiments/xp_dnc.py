import numpy as np

from brainforge.atomic.activation import OnePlus


oneplus = OnePlus()


class DNC:

    def __init__(self, controller, reads):
        self.ctrl = controller
        self.memory = None
        self.usage = None
        self.link = None
        self.reads = np.zeros(())

    def forward_step(self, x, reads):
        Z = np.concatenate((x, reads.flat))

    def feedforward(self, X):
        pass
