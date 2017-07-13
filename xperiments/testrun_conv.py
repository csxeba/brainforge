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
        Conv2D(filters=15, kernel_size=(3, 3), activation="relu",
               data_format="channels_first", input_shape=inshape),
        Conv2D(filters=15, kernel_size=(3, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Activation("relu"),
        Conv2D(filters=15, kernel_size=(3, 3), activation="relu"),
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


def build_cnn(data: CData, gradcheck=False):
    from brainforge import GradientLearner
    from brainforge.architecture import ConvLayer, Flatten, Activation, PoolLayer
    inshape, outshape = data.neurons_required
    net = GradientLearner(input_shape=inshape, name="TestBrainforgeCNN", layers=(
        ConvLayer(15, 3, 3, compiled=False, activation="relu"),
        ConvLayer(15, 3, 3, compiled=False),
        PoolLayer(2, compiled=True),
        Activation("relu"),
        ConvLayer(15, 3, 3, compiled=False, activation="relu"),
        ConvLayer(10, 3, 3, compiled=False),
        PoolLayer(2, compiled=False),
        Activation("relu"),
        ConvLayer(10, 3, 3, compiled=False, activation="tanh"),
        ConvLayer(10, 2, 2, compiled=True),
        Flatten(),
        Activation("softmax"),
    ))
    net.finalize("xent", optimizer="adam")
    if gradcheck:
        net.gradient_check(*data.table("testing", m=10))
    return net


def keras_run():
    start = time.time()
    print("Running KERAS...")

    mnist = pull_mnist_data()
    net = build_keras_reference(mnist)
    print("Initial cost: {:>7.3f}, initial acc: {:>7.2%}"
          .format(*net.evaluate(*mnist.table("testing"))))
    X, Y = mnist.table("learning")
    net.fit(X, Y, batch_size=50, epochs=1,
            validation_data=mnist.table("testing"))
    score = net.evaluate(*mnist.table("testing"))
    print("Final cost: {}, acc: {}".format(*score))
    print("KERAS took {} seconds!".format(time.time() - start))


def brainforge_run():
    start = time.time()
    print("Running BRAINFORGE...")

    mnist = pull_mnist_data()
    net = build_cnn(mnist, gradcheck=True)
    print("Initial cost: {:>7.3f}, initial acc: {:>7.2%}"
          .format(*net.evaluate(*mnist.table("testing"))))
    X, Y = mnist.table("learning")
    net.fit(X, Y, batch_size=50, epochs=1)
    score = net.evaluate(*mnist.table("testing"))
    print("Final cost: {}, acc: {}".format(*score))
    print("BRAINFORGE took {} seconds!".format(time.time() - start))


def xperiment():
    # keras_run()
    # print("\n")
    brainforge_run()


if __name__ == '__main__':
    xperiment()
