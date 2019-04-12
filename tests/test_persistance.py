import unittest
import pickle

from brainforge.util import etalon, persistance
from brainforge import BackpropNetwork
from brainforge.layers import Linear


class TestPersistance(unittest.TestCase):

    def setUp(self):
        X, Y = etalon
        self.net = BackpropNetwork(input_shape=(4,), layers=[
            Linear(30, activation="sigmoid"),
            Linear(3, activation="softmax")
        ], cost="xent", optimizer="sgd")
        self.net.fit(X, Y, batch_size=len(X)//2, epochs=3, validation=etalon)
        self.cost1, self.acc1 = self.net.evaluate(*etalon)

    def test_dense_with_pickle(self):
        sleepy = pickle.dumps(self.net)
        netcopy = pickle.loads(sleepy)
        self._check_persistence_ok(netcopy)

    def test_dense_with_capsule(self):
        persistance.Capsule.encapsulate(self.net, dumppath="./PersistanceTest.bro")
        netcopy = persistance.load("./PersistanceTest.bro")
        self._check_persistence_ok(netcopy)

    def _check_persistence_ok(self, netcopy):
        cost2, acc2 = netcopy.evaluate(*etalon)

        self.assertAlmostEqual(self.cost1, cost2)
        self.assertAlmostEqual(self.acc1, acc2)
        self.assertFalse(self.net is netcopy)
