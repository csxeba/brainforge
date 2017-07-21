import warnings

import numpy as np


class GradientCheck:

    def __init__(self, network, epsilon, display):
        if network.age <= 1:
            warnings.warn(
                "\nPerforming gradient check on an untrained neural network!",
                RuntimeWarning
            )
        self.net = network
        self.eps = epsilon
        self.dsp = display

    def fold_difference_matrices(self, dvec):
        diffs = []
        start = 0
        for layer in self.net.layers[1:]:
            if not layer.trainable:
                continue
            iweight = start + layer.weights.size
            ibias = iweight + layer.biases.size
            wshape = [sh for sh in layer.weights.shape if sh > 1]
            bshape = [sh for sh in layer.biases.shape if sh > 1]
            diffs.append(dvec[start:iweight].reshape(wshape))
            diffs.append(dvec[iweight:ibias].reshape(bshape))
            start = ibias
        return diffs

    def analyze_difference_matrices(self, dvec):
        dmats = self.fold_difference_matrices(np.abs(dvec))
        for i, d in enumerate(dmats):
            print("Sum of difference matrix no {0}: {1:.4e}".format(i, d.sum()))
            self.display_matrices(d)

    def display_matrices(self, mats):
        from matplotlib import pyplot

        if mats.ndim > 2:
            for mat in mats:
                self.display_matrices(mat)
        else:
            pyplot.imshow(np.atleast_2d(mats), cmap="hot")
            pyplot.show()

    def get_results(self, er, verbose=1):
        if er < 1e-7:
            errcode = 0
        elif er < 1e-5:
            errcode = 1
        elif er < 1e-3:
            errcode = 2
        else:
            errcode = 3

        if verbose:
            print("Result of gradient check:")
            print(["Gradient check passed, error {} < 1e-7",
                   "Suspicious gradients, 1e-7 < error {} < 1e-5",
                   "Gradient check failed, 1e-5 < error {} < 1e-3",
                   "Fatal fail in gradient check, error {} > 1e-3"
                   ][errcode].format("({0:.1e})".format(er)))

        return True if errcode < 2 else False

    def run(self, X, Y):
        norm = np.linalg.norm
        analytic = self.analytical_gradients(self.net, X, Y)
        numeric = self.numerical_gradients(self.net, X, Y)
        diff = analytic - numeric
        relative_error = norm(diff) / max(norm(numeric), norm(analytic))
        passed = self.get_results(relative_error)

        if self.dsp and not passed:
            self.analyze_difference_matrices(diff)

        return passed

    def numerical_gradients(self, X, Y):
        network = self.net
        ws = network.get_weights(unfold=True)
        numgrads = np.zeros_like(ws)
        perturb = np.copy(numgrads)

        nparams = ws.size
        print("Calculating numerical gradients...")
        for i in range(nparams):
            print("\r{0:>{1}} / {2:<}".format(i+1, len(str(nparams)), nparams), end=" ")
            perturb[i] += self.eps

            network.set_weights(ws + perturb, fold=True)
            pred1 = network.feedforward(X)
            cost1 = network.cost(pred1, Y)
            network.set_weights(ws - perturb, fold=True)
            pred2 = network.feedforward(X)
            cost2 = network.cost(pred2, Y)

            numgrads[i] = (cost1 - cost2)
            perturb[i] = 0.

        numgrads /= 2.
        network.set_weights(ws, fold=True)

        print("Done!")

        return numgrads

    def analytical_gradients(self, X, Y):
        network = self.net
        print("Calculating analytical gradients...")
        print("Forward pass:", end=" ")
        preds = network.feedforward(X)
        print("done! Backward pass:", end=" ")
        delta = network.cost.derivative(preds, Y)
        nabla = network.backpropagate(delta)
        print("done!")
        return nabla
