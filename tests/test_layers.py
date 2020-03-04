import unittest

import numpy as np

from brainforge import layers
from brainforge.util import testing


class TestGlobalAveragePooling(unittest.TestCase):

    def setUp(self) -> None:
        self.inputs = np.empty([3, 5, 5], dtype="float32")
        for channel in range(3):
            self.inputs[channel] = np.full([5, 5], fill_value=channel+1, dtype="float32")
        self.outputs = np.array([1., 2., 3.], dtype="float32")
        self.brain = testing.NoBrainer(outshape=self.inputs.shape)
        self.layer = layers.GlobalAveragePooling()
        self.layer.connect(self.brain)

    def test_forward_pass_is_correct(self):

        output = self.layer.feedforward(self.inputs[None, ...])[0]
        np.testing.assert_equal(output, self.outputs)

    def test_backwards_pass_is_correct(self):

        self.layer.feedforward(self.inputs[None, ...])
        delta = self.layer.backpropagate(self.outputs[None, ...])[0]

        np.testing.assert_allclose(delta, self.inputs / 25)
