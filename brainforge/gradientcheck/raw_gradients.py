from brainforge import backend as xp


def analytical_gradients(gcobj, X, Y):
    network = gcobj.net
    print("Calculating analytical gradients...")
    print("Forward pass:", end=" ")
    preds = network.predict(X)
    print("done! Backward pass:", end=" ")
    delta = network.cost.derivative(preds, Y)
    network.backward(delta)
    nabla = network.get_gradients(unfold=True)
    print("done!")
    return nabla


def numerical_gradients(gcobj, X, Y):
    network = gcobj.net
    ws = network.layers.get_weights(unfold=True)
    numgrads = xp.zeros_like(ws)
    perturb = xp.zeros_like(ws)

    nparams = ws.size
    lstr = len(str(nparams))
    print("Calculating numerical gradients...")
    for i in range(nparams):
        print("\r{0:>{1}} / {2}".format(i + 1, lstr, nparams), end=" ")
        perturb[i] += gcobj.eps

        network.layers.set_weights(ws + perturb, fold=True)
        pred1 = network.predict(X)
        cost1 = network.cost(pred1, Y)
        network.layers.set_weights(ws - perturb, fold=True)
        pred2 = network.predict(X)
        cost2 = network.cost(pred2, Y)

        numgrads[i] = (cost1 - cost2)
        perturb[i] = 0.

    numgrads /= 2. * gcobj.eps
    network.layers.set_weights(ws, fold=True)

    print("Done!")

    return numgrads

