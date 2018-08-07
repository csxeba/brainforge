import numpy as np

from .abstract_model import Model


class Graph(Model):

    def __init__(self, input_shape, nodes, connectivity):
        super().__init__(input_shape)
        self.nodes = nodes  # some iterable containing Model instances (Layerstacks and/or Graphs)
        self.conn = connectivity  # 0/1 matrix defining connectivity vs timestep

    def feedforward(self, X):
        for mask in self.conn:
            X = np.concatenate(
                [node.feedforward(X) for ix, node in enumerate(self.nodes) if ix in mask]
            )
        return X

    @property
    def outshape(self):
        return self.nodes[-1].outshape

    @property
    def nparams(self):
        return sum(node.num_params for node in self.nodes)

    def get_weights(self, unfold=True):
        return []

    def set_weights(self, fold=True):
        pass
