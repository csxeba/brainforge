from brainforge.model import LayerStack
from brainforge.learner import DirectFeedbackAlignment
from brainforge.layers import Dense, Flatten

from verres.data import inmemory


def build_layerstack(input_shape, output_dim):
    return LayerStack(input_shape, [
        Flatten(),
        Dense(64, activation="tanh", compiled=True),
        Dense(output_dim, activation="softmax")])


def build_dfa_net(input_shape, output_dim):
    return DirectFeedbackAlignment(build_layerstack(input_shape, output_dim),
                                   cost="cxent", optimizer="adam")


def run_xperiment():
    mnist = inmemory.MNIST()

    lX, lY = mnist.table("train", shuffle=True)
    tX, tY = mnist.table("val", shuffle=False)

    ann = build_dfa_net(lX.shape[1:], lY.shape[1])
    ann.fit(lX, lY, batch_size=32, epochs=10, validation=(tX, tY), metrics=["acc"])


if __name__ == '__main__':
    run_xperiment()
