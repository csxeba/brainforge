"""
Neural Network Framework on top of NumPy
Copyright (C) 2017  Csaba Gór

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

import warnings

import numpy as np

from .learner import Learner
from ..optimization import optimizers


class BackpropNetwork(Learner):

    def finalize(self, cost="mse", optimizer="sgd"):
        super().finalize(cost)

        self.optimizer = optimizers[optimizer](self.nparams) \
            if isinstance(optimizer, str) else optimizer
        self._finalized = True
        return self

    def learn_batch(self, X, Y, w=None):
        self.X, self.Y = X, Y
        preds = self.predict(X)
        delta = self.cost.derivative(preds, Y)
        if w is not None:
            delta *= w[:, None]
        self.backpropagate(delta)
        self.set_weights(
            self.optimizer.optimize(
                self.get_weights(unfold=True),
                self.get_gradients(unfold=True),
                self.m
            )
        )
        return self.cost(self.output, Y)

    def backpropagate(self, error):
        # TODO: optimize this, skip untrainable architecture at the beginning
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)

    # ---- Some utilities ----

    def reset(self):
        for layer in (l for l in self.layers if l.trainable):
            layer.reset()

    def describe(self, verbose=0):
        if not self.name:
            name = "BrainForge Artificial Neural Network."
        else:
            name = "{}, the Artificial Neural Network.".format(self.name)
        chain = "----------\n"
        chain += name + "\n"
        chain += "Age: " + str(self.age) + "\n"
        chain += "Architecture: " + "->".join(self.architecture) + "\n"
        chain += "----------"
        if verbose:
            print(chain)
        else:
            return chain

    def get_gradients(self, unfold=True):
        grads = [l.gradients for l in self.layers if l.trainable]
        if unfold:
            grads = np.concatenate(grads)
        return grads

    def gradient_check(self, X, Y, verbose=1, epsilon=1e-5):
        from ..util import gradient_check
        if self.age == 0:
            warnings.warn("Performing gradient check on an untrained Neural Network!",
                          RuntimeWarning)
        return gradient_check(self, X, Y, verbose=verbose, epsilon=epsilon)

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def gradients(self):
        return self.get_gradients(unfold=True)
