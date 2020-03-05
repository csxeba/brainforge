import numpy as np

from brainforge.model import LayerStack, BackpropNetwork, config
from brainforge.layers import (
    ConvLayer, PoolLayer, Dense, Activation, Flatten
)
from brainforge.optimizers import Momentum
from brainforge.util import batch_stream

config.floatX = "float64"
config.compiled = True

dshape = (200, 5, 10, 10)
opt_lr = 0.001

def fake_data():
    X = np.random.randn(*dshape)
    Y = np.random.uniform(size=dshape[0]) < 0.5
    target = np.zeros((dshape[0], 2))
    target[range(len(X)), Y.astype(int)] = 1.
    return X, target


def build_network():
    C1 = BackpropNetwork(LayerStack(dshape[1:], layers=(
        ConvLayer(6, 3, 3), PoolLayer(2), Activation("relu"), Flatten()
    )), cost="xent", optimizer=Momentum(opt_lr))  # Output: 30, 6x8x8
    C2 = BackpropNetwork(LayerStack(dshape[1:], layers=(
        ConvLayer(6, 5, 5), PoolLayer(2), Activation("relu"), Flatten()
    )), cost="xent", optimizer=Momentum(opt_lr))  # Output: 30, 6x6x6
    C3 = BackpropNetwork(LayerStack(dshape[1:], layers=(
        ConvLayer(6, 7, 7), PoolLayer(2), Activation("relu"), Flatten()
    )), cost="xent", optimizer=Momentum(opt_lr))  # Output: 30, 6x4x4

    conv = [C1, C2, C3]

    FF = LayerStack(input_shape=sum(np.prod(C.layers.outshape) for C in conv),
                    layers=[Dense(60, activation="tanh"),
                            Dense(2, activation="softmax")])
    return conv, BackpropNetwork(FF, cost="xent", optimizer=Momentum(opt_lr))


def train_network(net, X, Y, batch_size):
    N = len(X)
    conv, FF = net
    done = 0
    batches = batch_stream(X, Y, m=batch_size)
    while 1:
        x, y = next(batches)
        CO = np.concatenate([C.predict(x) for C in conv], axis=1)
        FFO = FF.predict(CO)
        cost = float(FF.cost(FFO, y))
        done += len(x)

        print("\r{:>6.2%} Cost: {:.5f}".format(done / N, cost), end="")

        delta = FF.cost.derivative(FFO, y)
        delta = FF.backpropagate(delta)
        sectionsize = [C.layers.outshape[0] for C in conv]
        start = 0
        for C, ss in zip(conv, sectionsize):
            C.backpropagate(delta[:, start:start+ss])
            start = ss
        for node in conv + [FF]:
            W = node.layers.get_weights(unfold=True)
            nabla = node.get_gradients(unfold=True)
            node.optimizer.optimize(W, nabla, len(x))
        if done % N == 0:
            break
    print()


if __name__ == '__main__':
    data = fake_data()
    cnn = build_network()
    epochs = 30
    for epoch in range(1, epochs+1):
        print("Epoch {:>2}/{}".format(epoch, epochs))
        train_network(cnn, *data, batch_size=10)
