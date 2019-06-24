import numpy as np

from keras.datasets import mnist

from brainforge import LayerStack
from brainforge.layers import DenseLayer
from brainforge.learner.extreme_learning_machine import ExtremeLearningMachine


def pull_mnist(split=0.1):
    learning, testing = mnist.load_data()
    X = np.concatenate([learning[0], testing[0]]).astype("float32")
    Y = np.concatenate([learning[1], testing[1]]).astype("uint8")
    X -= X.mean()
    X /= X.std()
    X = X.reshape(-1, 784)
    Y = np.eye(10)[Y]

    if split:
        arg = np.arange(len(X))
        np.random.shuffle(arg)
        div = int(len(X) * split)
        targ, larg = arg[:div], arg[div:]
        return X[larg], Y[larg], X[targ], Y[targ]

    return X, Y


layers = LayerStack(input_shape=(784,), layers=[
    DenseLayer(60, activation="tanh", trainable=False),
    DenseLayer(10, activation="linear", trainable=True)
])

lX, lY, tX, tY = pull_mnist(0.1)
elm = ExtremeLearningMachine(layers, cost="mse")
elm.learn_batch(tX, tY)
pred = elm.predict(tX)
print("ELM metrics:", elm.cost(pred, tY))
print("ELM acc :", np.mean(tY.argmax(axis=0) == pred.argmax(axis=0)))
pass
