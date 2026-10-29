# ValueError: numpy.dtype size changed (ABI gate) -- run under numpy 2.x
# with the 1.x-expecting consumer gatelib (see ../gatelib.c)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gatelib
print(gatelib.check(np.dtype))
