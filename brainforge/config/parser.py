import sys

from os.path import expanduser, exists
from configparser import ConfigParser

from brainforge.config import numeric


def _sanitize_configpath():
    cfgpath = "~/.brainforgerc"
    if not exists(cfgpath):
        cfgpath = "~/.brainforgerc.txt"
        if not exists(cfgpath):
            cfgpath = None
    return cfgpath


def _create_configfile(cfg, path):
    cfg.add_section("numeric")
    cfg["numeric"]["floatX"] = "float64"
    cfg["numeric"]["compiled"] = "False"
    with open(expanduser(path) + (".txt" if sys == "win32" else ""), "w") as handle:
        cfg.write(handle)


def set_globals():
    configpath = _sanitize_configpath()
    config = ConfigParser()
    if configpath is None:
        configpath = "~/.brainforgerc"
        _create_configfile(config, configpath)

    config.read([expanduser(configpath)], encoding="utf8")
    numeric.floatX = config["numeric"].get("floatX", "float64").lower()
    numeric.compiled = config["numeric"].get("compiled", "false").lower() == "true"
