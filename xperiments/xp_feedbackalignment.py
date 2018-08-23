from brainforge.model import LayerStack
from brainforge.learner import DirectFeedbackAlignment
from brainforge.layers import DenseLayer

from csxdata.utilities.loader import pull_mnist_data


def build_layerstack(input_shape, output_dim):
    return LayerStack(input_shape, [
        DenseLayer(64, activation="tanh", compiled=True),
        DenseLayer(output_dim, activation="softmax")])


def build_dfa_net(input_shape, output_dim):
    return DirectFeedbackAlignment(build_layerstack(input_shape, output_dim),
                                   cost="cxent", optimizer="adam")


def run_xperiment():
    lX, lY, tX, tY = pull_mnist_data()
    ann = build_dfa_net(lX.shape[1:], lY.shape[1])
    ann.fit(lX, lY, batch_size=32, epochs=10, validation=(tX, tY))


if __name__ == '__main__':
    run_xperiment()
