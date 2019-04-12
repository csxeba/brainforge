try:
    import cupy as backend
except ImportError:
    import numpy as backend
