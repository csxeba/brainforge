import time
from csxdata import CData, roots


def pull_mnist_data():
    mnist = CData(roots["misc"] + "mnist.pkl.gz", cross_val=0.18, floatX="float64")
    mnist.transformation = "std"
    return mnist


def build_keras_reference(data: CData):
    from keras.models import Sequential
    from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation
    inshape, outshape = data.neurons_required
    net = Sequential([
        Conv2D(filters=24, kernel_size=(3, 3), input_shape=inshape, activation="relu"),
        Conv2D(filters=12, kernel_size=(3, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Activation("relu"),
        Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
        Conv2D(filters=10, kernel_size=(3, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Activation("relu"),
        Conv2D(filters=10, kernel_size=(3, 3), activation="tanh"),
        Conv2D(filters=10, kernel_size=(2, 2)),
        Flatten(),
        Activation("softmax"),
    ])
    net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return net


def build_cnn(data: CData):
    from brainforge import Network
    from brainforge.layers import ConvLayer, Flatten, Activation, PoolLayer
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="TestBrainforgeCNN", layers=(
        ConvLayer(24, 3, 3, compiled=True, activation="relu"),
        ConvLayer(12, 3, 3, compiled=True),
        PoolLayer(2, compiled=True),
        Activation("relu"),
        ConvLayer(12, 3, 3, compiled=True, activation="relu"),
        ConvLayer(10, 3, 3, compiled=True),
        PoolLayer(2, compiled=True),
        Activation("relu"),
        ConvLayer(10, 3, 3, compiled=True, activation="tanh"),
        ConvLayer(10, 2, 2, compiled=True),
        Flatten(),
        Activation("softmax"),
    ))
    net.finalize("xent", optimizer="adam")
    return net


def keras_run():
    mnist = pull_mnist_data()
    net = build_keras_reference(mnist)
    print("Initial cost: {}, initial acc: {}"
          .format(*net.evaluate(*mnist.table("testing"))))
    X, Y = mnist.table("learning")
    net.fit(X, Y, batch_size=50, epochs=10,
            validation_data=mnist.table("testing"))
    return net.evaluate(*mnist.table("testing"))


def brainforge_run():
    mnist = pull_mnist_data()
    net = build_cnn(mnist)
    print("Initial cost: {}, initial acc: {}"
          .format(*net.evaluate(*mnist.table("testing"))))
    X, Y = mnist.table("learning")
    net.fit(X, Y, batch_size=50, epochs=10,
            validation=mnist.table("learning"))
    return net.evaluate(*mnist.table("learning"))


def xperiment():
    start = time.time()
    print("Running BRAINFORGE...", end=" ")
    score = brainforge_run()
    print("BRAINFORGE took {} seconds!".format(time.time() - start))
    print("Final cost: {}, acc: {}".format(*score))
    print("\n")
    start = time.time()
    print("Running KERAS...", end=" ")
    score = keras_run()
    print("KERAS took {} seconds!".format(time.time() - start))
    print("Final cost: {}, acc: {}".format(*score))


if __name__ == '__main__':
    xperiment()
