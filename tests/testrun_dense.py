from csxdata import CData, roots, log
from csxdata.utilities.parsers import mnist_tolearningtable

from brainforge import Network
from brainforge.layers import DenseLayer, DropOut, HighwayLayer

mnistpath = roots["misc"] + "mnist.pkl.gz"
logstring = ""


def get_mnist_data(path):
    data = CData(mnist_tolearningtable(path, fold=False), headers=None)
    data.transformation = "std"
    return data


def keras_reference_network(data):
    from keras.models import Sequential
    from keras.layers import Dense
    inshape, outshape = data.neurons_required
    model = Sequential([
        Dense(60, activation="tanh", input_shape=inshape),
        Dense(outshape[0], activation="softmax")
    ])
    model.compile(optimizer="nadam", loss="categorical_crossentropy",
                  metrics=["acc"])
    return model


def get_dense_network(data):
    fanin, fanout = data.neurons_required
    nw = Network(fanin, name="TestDenseNet")
    nw.add(DenseLayer(30, activation="tanh", trainable=True))
    nw.add(DenseLayer(fanout, activation="softmax"))
    nw.finalize("xent", optimizer="sgd")
    return nw


def get_drop_net(data):
    fanin, fanout = data.neurons_required
    nw = Network(fanin, name="TestDropoutNet")
    nw.add(DenseLayer(30, activation="sigmoid"))
    nw.add(DropOut(0.5))
    nw.add(DenseLayer(fanout, activation="sigmoid"))
    nw.finalize("mse", optimizer="sgd")
    return nw


def get_highway_net(data):
    fanin, fanout = data.neurons_required
    nw = Network(fanin, name="TestHighwayNet")
    nw.add(DenseLayer(120, activation="tanh"))
    nw.add(HighwayLayer(activation="tanh"))
    nw.add(HighwayLayer(activation="tanh"))
    nw.add(DenseLayer(fanout, activation="softmax"))
    nw.finalize(cost="xent", optimizer="sgd")
    return nw


def main():

    log(" --- CsxNet Brainforge testrun ---")
    mnist = get_mnist_data(mnistpath)
    mnist.transformation = "std"

    net = get_dense_network(mnist)
    net.gradient_check(*mnist.table("testing", m=50))
    # net = keras_reference_network(mnist)
    net.fit(*mnist.table("learning"), batch_size=20, epochs=30, verbose=1,
            validation_data=mnist.table("testing"))


if __name__ == '__main__':
    main()
