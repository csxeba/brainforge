from csxdata import CData
from csxdata import Sequence, roots
from csxnet import Network
from layers.core import *
from layers.fancy import HighwayLayer
from layers.recurrent import RLayer, LSTM, EchoLayer
from layers.tensor import PoolLayer, ConvLayer


def pull_petofi_data():
    return Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=5,
                    cross_val=0.01)


def pull_mnist_data(m=1000):
    from csxdata.utilities.parsers import mnist_tolearningtable

    X, Y = mnist_tolearningtable(roots["misc"] + "mnist.pkl.gz")
    if m is None:
        m = X.shape[0]
    X, Y = X[:m], Y[:m]

    return CData((X, Y), standardize=True)


def build_ultimate_recurrent_combo_network(data: Sequence, gradcheck=True):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="UltimateRecurrentComboNetwork")
    net.add(LSTM(20, activation="tanh", return_seq=True))
    net.add(RLayer(10, activation="tanh", return_seq=True))
    net.add(EchoLayer(5, activation="tanh"))
    net.add(DenseLayer(20, activation="tanh"))
    net.add(HighwayLayer(activation="tanh"))
    net.add(DenseLayer(outshape, activation="sigmoid"))
    net.finalize("xent", optimizer="adam")

    net.describe(1)

    if gradcheck:
        net.fit(*data.table("learning", m=20), batch_size=20, epochs=1, verbose=0)
        if not net.gradient_check(*data.table("testing", m=5)):
            raise RuntimeError("Gradient check failed!")
    return net


def build_ultimate_convolutional_combo_network(data: CData, gradcheck=True):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="UltimateConvolutionalComboNetwork")
    net.add(ConvLayer(1, 8, 8))
    net.add(PoolLayer(3))
    net.add(Activation("sigmoid"))
    net.add(HighwayLayer(activation="relu"))
    net.add(Flatten())
    net.add(HighwayLayer(activation="relu"))
    net.add(DenseLayer(20, activation="tanh"))
    net.add(DenseLayer(outshape, activation="sigmoid"))
    net.finalize("xent", optimizer="adam")

    net.describe(1)

    if gradcheck:
        net.fit(*data.table("learning", m=20), batch_size=20, epochs=1, verbose=0)
        if not net.gradient_check(*data.table("testing", m=5)):
            raise RuntimeError("Gradient check failed!")
    return net


def rxperiment():
    petofi = pull_petofi_data()
    model = build_ultimate_recurrent_combo_network(petofi, gradcheck=False)
    model.fit_csxdata(petofi, monitor=["acc"], epochs=1)
    tx, ty = petofi.table("testing", m=50)
    acc_before_sleeping = model.evaluate(tx, ty)
    again = model.evaluate(tx, ty)
    assert acc_before_sleeping == again

    bedroom = roots["tmp"] + model.name + ".bro"
    model.encapsulate(bedroom)
    del model
    model = Network.from_capsule(bedroom)
    acc_after_sleeping = model.evaluate(tx, ty)

    if acc_before_sleeping != acc_after_sleeping:
        err = "Sleeping altered {}!\n".format(model.name)
        err += "Before: {}, After: {}".format(acc_before_sleeping, acc_after_sleeping)
        raise RuntimeError(err)
    print("Sleeping didn't alter", model.name)


def cxperiment():
    mnist = pull_mnist_data(1000)
    model = build_ultimate_convolutional_combo_network(mnist, gradcheck=False)
    model.fit_csxdata(mnist, monitor=["acc"], epochs=1)

    tx, ty = mnist.table("testing", m=50)
    acc_before_sleeping = model.evaluate(tx, ty)

    bedroom = roots["tmp"] + model.name + ".bro"
    model.encapsulate(bedroom)
    del model
    model = Network.from_capsule(bedroom)
    acc_after_sleeping = model.evaluate(tx, ty)

    if acc_before_sleeping != acc_after_sleeping:
        err = "Sleeping altered {}!\n".format(model.name)
        err += "Before: {}, After: {}".format(acc_before_sleeping, acc_after_sleeping)
        raise RuntimeError(err)
    print("Sleeping didn't alter", model.name)


if __name__ == '__main__':
    rxperiment()
    cxperiment()
