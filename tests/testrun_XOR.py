import numpy as np

from matplotlib import pyplot as plt

from brainforge import Network
from brainforge.layers import DenseLayer

plt.ion()


def learnme(m=20):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    while 1:
        arg = np.random.randint(len(X), size=m)
        yield X[arg], Y[arg]


def forge_net():
    bob = Network(input_shape=(2,), layers=[
        DenseLayer(12, activation="sigmoid"),
        DenseLayer(2, activation="sigmoid")
    ])
    bob.finalize(cost="xent", optimizer="adam")
    return bob


def forge_ae():
    ae = Network(input_shape=(2,), layers=[
        DenseLayer(12, activation="sigmoid"),
        DenseLayer(2)
    ])
    ae.finalize(cost="mse", optimizer="momentum")
    return ae

if __name__ == '__main__':
    net = forge_net()

    obj = plt.imshow(net.layers[1].weights, interpolation="none",
                     vmin=-2.0, vmax=2.0)

    datagen = learnme(1000)
    epoch = 0
    while epoch < 30:
        print("Epoch", epoch+1)
        X, Y = next(datagen)
        net.epoch(X, Y, batch_size=20, monitor=["acc"], validation=next(learnme(100)), verbose=1)
        obj.set_data = net.layers[1].weights
        plt.pause(1)
        epoch += 1
