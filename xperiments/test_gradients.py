import unittest

from numpy.linalg import norm

from csxdata import CData, roots
from csxdata.utilities.parsers import mnist_tolearningtable

from brainforge.util import numerical_gradients, analytical_gradients
from brainforge.model import GradientLearner
from brainforge.architecture import DenseLayer


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.data = CData(mnist_tolearningtable(roots["misc"] + "mnist.pkl.gz", fold=False), headers=None)
        self.data.transformation = "std"
        self.X, self.Y = self.data.table("testing", m=5, shuff=False)

        self.net = GradientLearner(self.data.neurons_required[0], name="NumGradTestNetwork")
        self.net.add(DenseLayer(30, activation="sigmoid"))

    def test_mse_with_sigmoid_output(self):
        self.net.add(DenseLayer(self.data.neurons_required[1], activation="sigmoid"))
        self.net.finalize(cost="mse", optimizer="sgd")
        self._run_numerical_gradient_test()

    def test_xent_with_sigmoid_output(self):
        self.net.add(DenseLayer(self.data.neurons_required[1], activation="sigmoid"))
        self.net.finalize(cost="xent", optimizer="sgd")
        self._run_numerical_gradient_test()

    def test_xent_with_softmax_output(self):
        self.net.add(DenseLayer(self.data.neurons_required[1], activation="softmax"))
        self.net.finalize(cost="xent", optimizer="sgd")
        self._run_numerical_gradient_test()

    def _run_numerical_gradient_test(self):
        self.net.fit(*self.data.table("learning", m=20), batch_size=20, epochs=1, verbose=0)

        numerical = numerical_gradients(self.net, self.X, self.Y)
        analytical = analytical_gradients(self.net, self.X, self.Y)
        diff = analytical - numerical
        error = norm(diff) / max(norm(numerical), norm(analytical))

        dfstr = "{0: .4f}".format(error)

        self.assertLess(error, 1e-2, "FATAL ERROR, {} (relerr) >= 1e-2".format(dfstr))
        self.assertLess(error, 1e-4, "ERROR, 1e-2 > {} (relerr) >= 1e-4".format(dfstr))
        self.assertLess(error, 1e-7, "SUSPICIOUS, 1e-4 > {} (relerr) >= 1e-7".format(dfstr))
