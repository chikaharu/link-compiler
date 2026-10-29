# ImportError by version-string gate (real numpy here is 1.26.4)
import numpy as np
from numpy.lib import NumpyVersion
if NumpyVersion(np.__version__) < "2.0.0":
    raise ImportError(f"this library requires numpy>=2.0, got {np.__version__}")
