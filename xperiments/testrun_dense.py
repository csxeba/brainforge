from csxdata import CData, roots, log
from csxdata.utilities.parsers import mnist_tolearningtable

from brainforge import BackpropNetwork
from brainforge.architecture import DenseLayer, DropOut, HighwayLayer
from brainforge.optimization import SGD

mnistpath = roots["misc"] + "mnist.pkl.gz"
logstring = ""


def get_mnist_data(path):
    data = CData(mnist_tolearningtable(path, fold=False), headers=None)
    # data.transformation = "std"
    return data


def keras_reference_network(data):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD
    inshape, outshape = data.neurons_required
    model = Sequential([
        Dense(30, activation="sigmoid", input_shape=inshape),
        Dense(outshape[0], activation="sigmoid")
    ])
    model.compile(optimizer=SGD(3.), loss="mse",
                  metrics=["acc"])
    return model


def get_dense_network(data):
    fanin, fanout = data.neurons_required
    nw = BackpropNetwork(fanin, name="TestDenseNet")
    nw.add(DenseLayer(30, activation="sigmoid"))
    nw.add(DenseLayer(fanout, activation="sigmoid"))
    nw.finalize("mse", optimizer=SGD(nw.nparams, 3.))
    return nw


def get_drop_net(data):
    fanin, fanout = data.neurons_required
    nw = BackpropNetwork(fanin, name="TestDropoutNet")
    nw.add(DenseLayer(30, activation="sigmoid"))
    nw.add(DropOut(0.5))
    nw.add(DenseLayer(fanout, activation="sigmoid"))
    nw.finalize("mse", optimizer="sgd")
    return nw


def get_highway_net(data):
    fanin, fanout = data.neurons_required
    nw = BackpropNetwork(fanin, name="TestHighwayNet")
    nw.add(DenseLayer(120, activation="tanh"))
    nw.add(HighwayLayer(activation="tanh"))
    nw.add(HighwayLayer(activation="tanh"))
    nw.add(DenseLayer(fanout, activation="softmax"))
    nw.finalize(cost="xent", optimizer="sgd")
    return nw


def main():

    log(" --- Brainforge testrun ---")
    mnist = get_mnist_data(mnistpath)

    net = get_dense_network(mnist)
    knet = keras_reference_network(mnist)
    # if not net.gradient_check(*mnist.table("testing", m=5)):
    #     raise RuntimeError("Gradient Check failed!")
    net.fit(*mnist.table("learning"), batch_size=20, epochs=3, verbose=1,
            validation=mnist.table("testing"), monitor=["acc"])
    knet.fit(*mnist.table("learning"), batch_size=20, epochs=3, verbose=1,
             validation_data=mnist.table("testing"))


if __name__ == '__main__':
    main()
