from csxdata import Sequence, roots
from csxdata.utilities.helpers import speak_to_me

from csxnet import Network
from layers.core import LSTM, DenseLayer
from layers.recurrent import RLayer, LSTM


def pull_petofi_data():
    return Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=5,
                    cross_val=0.01)


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
    if what.lower() == "lstm":
        net.add(LSTM(30, activation="tanh", return_seq=True))
        net.add(LSTM(30, activation="tanh"))
    else:
        net.add(RLayer(30, activation="tanh", return_seq=True))
        net.add(RLayer(30, activation="tanh"))
    net.add(DenseLayer(outshape, activation="sigmoid"))
    net.finalize("mse")
    return net


def build_rnn(data: Sequence):
    return _build(data, "rnn")


def build_LSTM(data: Sequence):
    return _build(data, "lstm")


def xperiment():
    petofi = pull_petofi_data()
    net = build_LSTM(petofi)
    net.describe(verbose=1)
    print("Initial cost: {} acc: {}".format(*net.evaluate(*petofi.table("testing"))))
    print(speak_to_me(net, petofi))

    net.fit(*petofi.table("learning", m=40, shuff=True), epochs=1, verbose=0, shuffle=False)
    if not net.gradient_check(*petofi.table("testing", m=10)):
        return

    X, Y = petofi.table("learning")

    for decade in range(1, 10):
        net.fit(X, Y, 20, 5, monitor=["acc"], validation=petofi.table("testing"))
        print("-"*12)
        print("Decade: {0:2<}.5 |".format(decade-1))
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
