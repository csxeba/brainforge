import numpy as np

from brainforge import Network
from brainforge.layers import DenseLayer

from matplotlib import pyplot as plt


def input_stream(m=20):
    Xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Ys = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    while 1:
        arg = np.random.randint(len(Xs), size=m)
        yield Xs[arg], Ys[arg]

net = Network(input_shape=(2,), layers=[
    DenseLayer(4, activation="sigmoid"),
    DenseLayer(2, activation="softmax")
])
net.finalize(cost="xent", optimizer="evolution")

datagen = input_stream(1000)
validation = next(input_stream(100))

costs = net.fit_generator(datagen, 1000, epochs=20, monitor=["acc"],
                          validation=validation, verbose=1)

Xs = np.arange(1, len(costs)+1)
plt.scatter(Xs, costs, "bo")
plt.title("XOR by differential evolution")
plt.show()
