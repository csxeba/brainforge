from csxdata import CData, roots
from brainforge import Network
from brainforge.layers import ConvLayer, DenseLayer, Flatten, Activation, PoolLayer


def pull_mnist_data():
    mnist = CData(roots["misc"] + "mnist.pkl.gz", cross_val=0.18, floatX="float64")
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
    # MNIST: 28x28
    net.add(ConvLayer(24, 3, 3, compiled=True, activation="relu"))  # 26
    net.add(ConvLayer(12, 3, 3, compiled=True))  # 24
    net.add(PoolLayer(2, compiled=True))  # 12
    net.add(Activation("relu"))
    net.add(ConvLayer(12, 3, 3, compiled=True, activation="relu"))  # 10
    net.add(ConvLayer(10, 3, 3, compiled=True))  # 8
    net.add(PoolLayer(2, compiled=True))  # 4
    net.add(Activation("relu"))
    net.add(ConvLayer(10, 3, 3, compiled=True, activation="tanh"))  # 2
    net.add(ConvLayer(10, 2, 2, compiled=True))  # 1
    net.add(Flatten())
    net.add(Activation("softmax"))
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
    net.learn_batch(*mnist.table("learning", m=10))
    net.age += 1
    if not net.gradient_check(*mnist.table("testing", m=10)):
        raise RuntimeError("Gradient Check Failed!")
    X, Y = mnist.table("learning")
    net.fit(X, Y, batch_size=30, epochs=1, verbose=1)

if __name__ == '__main__':
    xperiment()
