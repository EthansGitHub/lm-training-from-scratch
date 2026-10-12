import importlib.metadata

try:
    __version__ = importlib.metadata.version("lm-pretraining")
except importlib.metadata.PackageNotFoundError:
    pass
