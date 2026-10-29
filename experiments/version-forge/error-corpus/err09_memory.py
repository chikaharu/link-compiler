# numpy MemoryError: impossible allocation (fails fast, allocates nothing)
import numpy as np
a = np.empty(10**12, dtype=np.float64)  # 8 TB
