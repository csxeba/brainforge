from csxdata import Sequence, roots
from csxdata.utilities.helpers import speak_to_me

from brainforge import Network
from brainforge.layers import LSTM, GRU, RLayer, DenseLayer


def pull_petofi_data():
    return Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=5,
                    cross_val=0.01, lower=True, dehungarize=True)


def build_keras_net(data: Sequence):
    from keras.models import Sequential
    from keras.layers import SimpleRNN, Dense

    indim, outdim = data.neurons_required
    return Sequential([
        SimpleRNN(input_dim=indim, output_dim=(500,), activation="tanh"),
        Dense(outdim, activation="sigmoid")
    ]).compile(optimizer="sgd", loss="xent")


def _build(data, what):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="TestRNN")
    rl1 = 30
    rl2 = 10
    act = "relu"
    if what.lower() == "lstm":
        net.add(LSTM(rl1, activation=act, return_seq=True))
        net.add(LSTM(rl2, activation=act))
    elif what.lower() == "gru":
        net.add(GRU(rl1, activation=act, return_seq=True))
        net.add(GRU(rl2, activation=act))
    else:
        net.add(RLayer(rl1, activation=act, return_seq=True))
        net.add(RLayer(rl2, activation=act))

    net.add(DenseLayer(outshape, activation="softmax"))
    net.finalize("xent")
    return net


def build_rnn(data: Sequence):
    return _build(data, "rnn")


def build_LSTM(data: Sequence):
    return _build(data, "lstm")


def build_GRU(data: Sequence):
    return _build(data, "gru")


def xperiment():
    petofi = pull_petofi_data()
    net = build_rnn(petofi)
    net.describe(verbose=1)
    print("Initial cost: {} acc: {}".format(*net.evaluate(*petofi.table("testing"))))
    print(speak_to_me(net, petofi))

    net.fit(*petofi.table("learning", m=10, shuff=True), epochs=1, verbose=0, shuffle=False)
    if not net.gradient_check(*petofi.table("testing", m=10)):
        return

    X, Y = petofi.table("learning")

    for decade in range(1, 10):
        net.fit(X, Y, 20, 5, monitor=["acc"], validation=petofi.table("testing"))
        print("-"*12)
        print("Decade: {0:3<}.5 |".format(decade-1))
        print("-"*12)
        print(speak_to_me(net, petofi))
        net.fit(X, Y, 20, 5, monitor=["acc"], validation=petofi.table("testing"))
        print("-"*12)
        print("Decade: {0:2<} |".format(decade))
        print("-"*12)
        print(speak_to_me(net, petofi))


def smallrun():
    petofi = pull_petofi_data()
    net = build_LSTM(petofi)
    net.fit(*petofi.table("learning", m=120), epochs=10, verbose=0)


if __name__ == '__main__':
    xperiment()
