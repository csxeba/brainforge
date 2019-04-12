from brainforge.util import etalon
from brainforge import LayerStack, BackpropNetwork
from brainforge.layers import Linear, DropOut


ls = LayerStack((4,), layers=[
    Linear(120, activation="tanh"),
    # DropOut(0.5),
    Linear(3, activation="softmax")
])

net = BackpropNetwork(ls, cost="xent", optimizer="momentum")
costs = net.fit(*etalon, epochs=300, validation=etalon, verbose=1)
