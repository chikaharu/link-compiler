"""Can the slot-32 side effect be fixed in code?
Fix candidate: TIME-DIVISION. The collision is only fatal if both
readers need the cell at the same time. So: save the genuine 8 bytes
at offset 32, store elsize, call the 1.x consumer, restore the bytes.
numpy's own readers all run outside the wrapped window and only ever
see genuine bytes. Test especially the itemsize=1 dtypes (bool/int8/
uint8) whose permanent store made np.empty emit `get_clear_loop`
internal-error tracebacks in the multiplexing experiment."""
import ctypes, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gatelib

def cell(addr, off, n=8): return ctypes.string_at(addr+off, n)
def put_i32(addr, off, v): ctypes.c_int32.from_address(addr+off).value = v
def put_bytes(addr, off, b): ctypes.memmove(addr+off, b, len(b))

# the type gate stays forced for this demo (gate side, as before)
ctypes.c_int64.from_address(id(np.dtype)+32).value = 96
print("check gate:", gatelib.check(np.dtype))

def wrapped_consume(d):
    saved = cell(id(d), 32)                    # save genuine field
    v = ctypes.c_int64.from_address(id(d)+40).value   # inferred rule: source @40
    put_i32(id(d), 32, int(v))                  # store only during the call
    try:
        return gatelib.consume(d)
    finally:
        put_bytes(id(d), 32, saved)             # restore, always

for name, d in [("bool", np.dtype(bool)), ("int8", np.dtype(np.int8)),
                ("uint8", np.dtype(np.uint8)), ("float64", np.dtype(np.float64)),
                ("structured", np.dtype([('a','i4'),('b','f8')]))]:
    before = cell(id(d), 32)
    got = wrapped_consume(d)
    restored = cell(id(d), 32) == before
    a = np.empty(2, dtype=d)                    # numpy self-use AFTER the call
    print(f"{name:>11}: consume={got} true={d.itemsize} restored={restored} numpy-sum-ok={float(np.arange(2, dtype=np.float64).sum()) if d.kind=='f' else 'n/a'} empty-ok={a.dtype is d}")
print("done -- watch stderr: get_clear_loop tracebacks would appear here if the fix failed")
