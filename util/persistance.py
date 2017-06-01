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
