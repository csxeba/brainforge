from brainforge.util import etalon
from brainforge import LayerStack, BackpropNetwork
from brainforge.layers import Dense, DropOut


ls = LayerStack((4,), layers=[
    Dense(120, activation="tanh"),
    # DropOut(0.5),
    Dense(3, activation="softmax")
])

net = BackpropNetwork(ls, cost="cxent", optimizer="momentum")
costs = net.fit(*etalon, epochs=300, validation=etalon, verbose=1)
