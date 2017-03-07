import numpy as np

from brainforge import Network
from brainforge.layers import DenseLayer

def input_stream(m=20):
    Xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Ys = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    while 1:
        arg = np.random.randint(len(Xs), size=m)
        yield Xs[arg], Ys[arg]

net = Network(input_shape=(2,), layers=[
    DenseLayer(30, activation="sigmoid"),
    DenseLayer(2, activation="softmax")
])
net.finalize(cost="xent", optimizer="sgd")

datagen = input_stream(1000)
validation = next(input_stream(100))

net.fit_generator(datagen, 1000000, epochs=3, monitor=["acc"],
                  validation=validation, verbose=1)
