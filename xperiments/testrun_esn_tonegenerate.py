from sklearn.svm import SVR

from brainforge import BackpropNetwork
from brainforge.architecture import Reservoir


def build_model(inputs):
    net = BackpropNetwork(inputs, layers=[
        Reservoir(1000, "relu")
    ])
    net.finalize()
    return net, SVR()


reservoir, svm = build_model(2)

