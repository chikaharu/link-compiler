# bench_python.py -- same computation as examples/bench.link.
# Scalar loop in pure Python (interpreter), vector part in NumPy.
# The Link binary does both natively; the loop is what dominates Python's time.
import math
import numpy as np

total = 0.0
for i in range(10_000_000):
    total += math.sqrt(i + 1.0)
print("sum_sqrt =", total)

v = np.arange(1_000_000, dtype=np.float64)
w = np.cumsum(v)
print("cumsum_last =", w[999_999])
d = np.empty_like(w)
d[0] = w[0]
d[1:] = w[1:] - w[:-1]
print("roundtrip_err =", float(np.linalg.norm(d - v)))
