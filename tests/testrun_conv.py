from csxdata import CData, roots
from csxnet import Network
from layers.core import ConvLayer, DenseLayer, Flatten, Activation
from layers.tensor import PoolLayer, ConvLayer


def pull_mnist_data():
    mnist = CData(roots["misc"] + "mnist.pkl.gz", cross_val=0.18)
    mnist.transformation = "std"
    return mnist


def build_keras_reference(data: CData):
    from keras.models import Sequential
    from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Activation
    inshape, outshape = data.neurons_required
    net = Sequential([
        Conv2D(nb_filter=1, nb_row=5, nb_col=5, input_shape=inshape),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Activation("tanh"),
        Dense(outshape[0], activation="sigmoid")
    ])
    net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return net


def build_cnn(data: CData):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="TestBrainforgeCNN")
    net.add(ConvLayer(1, 5, 5))
    net.add(PoolLayer(3))
    net.add(Activation("tanh"))
    net.add(Flatten())
    net.add(DenseLayer(outshape, activation="sigmoid", trainable=False))
    net.finalize("xent", optimizer="adam")
    return net


def keras_run():
    mnist = pull_mnist_data()
    net = build_keras_reference(mnist)
    net.fit(*mnist.table("learning"), batch_size=30, nb_epoch=10,
            validation_data=mnist.table("testing"))


def xperiment():
    mnist = pull_mnist_data()
    net = build_cnn(mnist)
    # net.fit(*mnist.table("learning", m=30), batch_size=10, epochs=1, verbose=0)
    net.gradient_check(*mnist.table("testing", m=20))
    net.fit(*mnist.table("learning"), batch_size=30, epochs=10, verbose=1)

if __name__ == '__main__':
    keras_run()
