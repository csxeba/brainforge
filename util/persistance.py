class Capsule:

    def __init__(self, name=None, cost=None, optimizer=None, architecture=None, layers=None):
        self.vname = name
        self.vcost = cost
        self.voptimizer = optimizer
        self.varchitecture = architecture
        self.vlayers = layers

    def dump(self, path):
        import pickle
        import gzip

        with gzip.open(path, "wb") as handle:
            pickle.dump({k: v for k, v in self.__dict__.items() if k[0] == "v"},
                        handle)

    @classmethod
    def encapsulate(cls, network, dumppath=None):
        capsule = cls(**{
            "name": network.name,
            "cost": network.cost,
            "optimizer": network.optimizer,
            "architecture": network.layers.architecture[:],
            "layers": [layer.capsule() for layer in network.layers.layers]})

        if dumppath is not None:
            capsule.dump(dumppath)
        return capsule

    @classmethod
    def read(cls, path):
        import pickle
        import gzip
        from os.path import exists

        if not exists(path):
            raise RuntimeError("No such capsule:", path)

        new = cls()
        with gzip.open(path) as handle:
            new.__dict__.update(pickle.load(handle))

        return new

    def __getitem__(self, item):
        if item not in self.__dict__:
            raise AttributeError("No such item in capsule:", item)
        return self.__dict__[item]


def load(capsule):
    from ..model import BackpropNetwork
    from ..optimization import optimizers
    from ..util.shame import translate_architecture as trsl

    if not isinstance(capsule, Capsule):
        capsule = Capsule.read(capsule)
    c = capsule

    net = BackpropNetwork(input_shape=c["vlayers"][0][0], name=c["vname"])

    for layer_name, layer_capsule in zip(c["varchitecture"], c["vlayers"]):
        if layer_name[:5] == "Input":
            continue
        layer_cls = trsl(layer_name)
        layer = layer_cls.from_capsule(layer_capsule)
        net.add(layer)

    opti = c["voptimizer"]
    if isinstance(opti, str):
        opti = optimizers[opti]()
    net.finalize(cost=c["vcost"], optimizer=opti)

    for layer, lcaps in zip(net.layers, c["vlayers"]):
        if layer.weights is not None:
            layer.set_weights(lcaps[-1], fold=False)

    return net
