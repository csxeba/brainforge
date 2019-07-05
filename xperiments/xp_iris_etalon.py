from brainforge.util import etalon
from brainforge import LayerStack, BackpropNetwork
from brainforge.layers import DenseLayer, DropOut


ls = LayerStack((4,), layers=[
    DenseLayer(120, activation="tanh"),
    # DropOut(0.5),
    DenseLayer(3, activation="softmax")
])

net = BackpropNetwork(ls, cost="cxent", optimizer="momentum")
costs = net.fit(*etalon, epochs=300, validation=etalon, verbose=1)
