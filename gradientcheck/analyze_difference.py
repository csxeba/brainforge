import numpy as np


def analyze_difference_matrices(gcobj, dvec):
    dmats = fold_difference_matrices(gcobj, np.abs(dvec))
    for i, d in enumerate(dmats):
        print("Sum of difference matrix no {0}: {1:.4e}".format(i, d.sum()))
        display_matrices(d)


def fold_difference_matrices(gcobj, dvec):
    diffs = []
    start = 0
    for layer in gcobj.net.layers[1:]:
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


def display_matrices(mats):
    from matplotlib import pyplot

    if mats.ndim > 2:
        for mat in mats:
            display_matrices(mat)
    else:
        pyplot.imshow(np.atleast_2d(mats), cmap="hot")
        pyplot.show()


def get_results(er, verbose=1):
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
