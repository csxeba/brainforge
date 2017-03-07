from csxdata import Sequence, roots
from csxdata.utilities.helpers import speak_to_me

from brainforge import Network
from brainforge.layers import (LSTM, GRU, RLayer, ClockworkLayer,
                               DenseLayer)


def pull_petofi_data():
    return Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=33,
                    cross_val=0.01, lower=True, dehungarize=True)


def build_keras_net(data: Sequence):
    from keras.models import Sequential
    from keras.layers import SimpleRNN, Dense

    indim, outdim = data.neurons_required
    return Sequential([
        SimpleRNN(input_dim=indim, output_dim=(500,), activation="tanh"),
        Dense(outdim, activation="sigmoid")
    ]).compile(optimizer="sgd", loss="xent")


def build(data, what):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="TestRNN")
    rl1 = 180
    act = "relu"
    if what.lower() == "lstm":
        net.add(LSTM(rl1, activation=act))
    elif what.lower() == "gru":
        net.add(GRU(rl1, activation=act))
    elif what.lower() == "cwrnn":
        net.add(ClockworkLayer(rl1, activaton=act))
    else:
        net.add(RLayer(rl1, activation=act))

    net.add(DenseLayer(outshape, activation="softmax"))
    net.finalize("xent", optimizer="adagrad")
    return net


def xperiment():
    petofi = pull_petofi_data()
    net = build(petofi, what="CWRNN")
    net.describe(verbose=1)
    print("Initial cost: {} acc: {}".format(*net.evaluate(*petofi.table("testing"))))
    print(speak_to_me(net, petofi))

    X, Y = petofi.table("learning")

    bsize = 180

    for decade in range(1, 10):
        net.fit(X, Y, bsize, 5, monitor=["acc"], validation=petofi.table("testing"))
        print("-" * 12 + "+")
        print("Decade: {0:3<}.5 |".format(decade-1))
        print("-" * 12 + "+")
        print(speak_to_me(net, petofi))
        net.fit(X, Y, bsize, 5, monitor=["acc"], validation=petofi.table("testing"))
        print("-" * 12 + "+")
        print("Decade: {0:3<} |".format(decade))
        print("-" * 12 + "+")
        print(speak_to_me(net, petofi))


if __name__ == '__main__':
    xperiment()
