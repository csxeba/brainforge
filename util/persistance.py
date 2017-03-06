class Capsule:
    def __init__(self, dirpath, name=None, cost=None,
                 optimizer=None, architecture=None,
                 layers=None):
        self.name = name
        self.cost = cost
        self.optimizer = optimizer
        self.architecture = architecture
        self.layers = layers
        self.flpath = dirpath + ("/" if dirpath[-1] not in ("/", "\\") else "") + self.name + ".cps"

    def dump(self):
        import pickle
        import gzip

        with gzip.open(self.flpath, "wb") as handle:
            pickle.dump({k: v for k, v in self.__dict__.items() if k[0] != "_"},
                        handle)

    @classmethod
    def read(cls, flpath):
        import pickle
        import gzip
        from os.path import exists

        if not exists(flpath):
            raise RuntimeError("No such capsule:", flpath)

        new = cls()
        with gzip.open(flpath) as handle:
            new.__dict__ = pickle.load(handle)

        return new

    def __getitem__(self, item):
        if item not in self.__dict__:
            raise AttributeError("No such item in capsule:", item)
        return self.__dict__[item]
