"""Model of a version-strict library (pure-Python version gate).
Requires numpy>=2.0 *by version string* at import time, but the actual
computation only uses long-stable numpy APIs (array/mean/matmul)."""
import numpy as np
from numpy.lib import NumpyVersion

if NumpyVersion(np.__version__) < "2.0.0":
    raise ImportError(f"strict_lib requires numpy>=2.0, got {np.__version__}")

def probe():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    m = np.array([[4.0, 1.0], [0.0, 3.0]])
    return f"mean={a.mean()} det_like_trace={m[0,0]*m[1,1]-m[0,1]*m[1,0]}"
