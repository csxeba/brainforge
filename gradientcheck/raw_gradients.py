import numpy as np


def analytical_gradients(gcobj, X, Y):
    network = gcobj.net
    print("Calculating analytical gradients...")
    print("Forward pass:", end=" ")
    preds = network.predict(X)
    print("done! Backward pass:", end=" ")
    delta = network.cost.derivative(preds, Y)
    nabla = network.backpropagate(delta)
    print("done!")
    return nabla


def numerical_gradients(gcobj, X, Y):
    network = gcobj.net
    ws = network.layers.get_weights(unfold=True)
    numgrads = np.zeros_like(ws)
    perturb = np.copy(numgrads)

    nparams = ws.size
    print("Calculating numerical gradients...")
    for i in range(nparams):
        print("\r{0:>{1}} / {2:<}".format(i + 1, len(str(nparams)), nparams), end=" ")
        perturb[i] += gcobj.eps

        network.layers.set_weights(ws + perturb, fold=True)
        pred1 = network.predict(X)
        cost1 = network.cost(pred1, Y)
        network.layers.set_weights(ws - perturb, fold=True)
        pred2 = network.predict(X)
        cost2 = network.cost(pred2, Y)

        numgrads[i] = (cost1 - cost2)
        perturb[i] = 0.

    numgrads /= 2.
    network.layers.set_weights(ws, fold=True)

    print("Done!")

    return numgrads

