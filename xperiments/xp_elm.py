import numpy as np

from keras.datasets import mnist

from brainforge import LayerStack
from brainforge.layers import DenseLayer, ConvLayer, Activation, Flatten
from brainforge.learner.extreme_learning_machine import ExtremeLearningMachine
from brainforge.util import typing


def pull_mnist(split=0.1, flatten=True):
    learning, testing = mnist.load_data()
    X = np.concatenate([learning[0], testing[0]]).astype(typing.floatX)
    Y = np.concatenate([learning[1], testing[1]]).astype("uint8")
    X -= X.mean()
    X /= X.std()
    if flatten:
        X = X.reshape(-1, 784)
    else:
        X = X[:, None, ...]
    Y = np.eye(10)[Y]

    if split:
        arg = np.arange(len(X))
        np.random.shuffle(arg)
        div = int(len(X) * split)
        targ, larg = arg[:div], arg[div:]
        return X[larg], Y[larg], X[targ], Y[targ]

    return X, Y


def build_dense_layerstack():
    return LayerStack(input_shape=(784,), layers=[
        DenseLayer(1024, activation="tanh", trainable=False),
        DenseLayer(10, activation="linear", trainable=True)
    ])


def build_conv_layerstack():
    return LayerStack(input_shape=(1, 28, 28), layers=[
        ConvLayer(16, 7, 7, trainable=False),
        Activation("tanh"),
        Flatten(),
        DenseLayer(10, activation="linear")
    ])


lX, lY, tX, tY = pull_mnist(0.1, flatten=True)
layers = build_dense_layerstack()
elm = ExtremeLearningMachine(layers, cost="mse", solve_mode="pseudoinverse")
elm.learn_batch(tX, tY)
pred = elm.predict(tX)
print("ELM metrics:", elm.cost(pred, tY))
print("ELM acc :", np.mean(tY.argmax(axis=1) == pred.argmax(axis=1)))
pass
