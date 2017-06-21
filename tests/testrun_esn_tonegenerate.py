from sklearn.svm import SVR

from brainforge import Network
from brainforge.layers import Reservoir


def build_model(inputs):
    net = Network(inputs, layers=[
        Reservoir(1000, "relu")
    ])
    net.finalize()
    return net, SVR()


reservoir, svm = build_model(2)

