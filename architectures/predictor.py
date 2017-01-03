def save_model(network):
    if isinstance(network, keras.models.Sequential):
        _save_keras_model(network)
    elif isinstance(network, architecture.ann.Network):
        _save_brainforged_model(network)


def _save_keras_model(network: keras.models.Sequential):
    keras.models.save_model(network, roots["brains"] + "kerasmodel_" + network.name + ".h5")


def _save_brainforged_model(network: csxnet.Network):
    network.save(roots["brain"] + "csxmodel_" + network.name + ".h5")


def load_model(path):
    if "kerasmodel" in path:
        return _load_keras_model(path)
    elif "csxmodel" in path:
        return _load_brainforged_model(path)


def _load_keras_model(path):
    return keras.models.load_model(path)


def _load_brainforged_model(path):
    pass

