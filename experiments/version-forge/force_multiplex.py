"""Instance multiplexing: if forcing is per-instance, pre-create ALL
instances the world will plausibly ask for, force every one of them,
and rely on numpy's dtype cache to hand out the forced instances.
Then probe the boundary: which later-created instances escape?"""
import ctypes, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gatelib

def store(addr, off, val, size=4):
    if size == 8: ctypes.c_int64.from_address(addr+off).value = val
    else: ctypes.c_int32.from_address(addr+off).value = val

print("numpy", np.__version__)
# Phase 1: the type-object gate
store(id(np.dtype), 32, 96, size=8)
print("check(np.dtype) ->", gatelib.check(np.dtype))

TYPES = [bool, np.int8, np.int16, np.int32, np.int64,
         np.uint8, np.uint16, np.uint32, np.uint64,
         np.float16, np.float32, np.float64, np.longdouble,
         np.complex64, np.complex128, np.bytes_, np.str_, np.object_]
uniq = {}
for t in TYPES:
    d = np.dtype(t); uniq.setdefault(id(d), d)
print(f"builtin dtype instances: {len(TYPES)} types -> {len(uniq)} unique instances")
print("identity stability: np.dtype(np.int32) is np.dtype('i4'):", np.dtype(np.int32) is np.dtype('i4'),
      "| zeros().dtype is np.dtype(np.int32):", np.zeros(2, np.int32).dtype is np.dtype(np.int32))

# Phase 2: multiplex -- force EVERY pre-created instance
for d in uniq.values():
    store(id(d), 32, d.itemsize)
print("all instances forced")

# Test: reach each dtype again through fresh numpy operations
ok = ng = 0
for t in TYPES:
    d = np.empty(1, dtype=t).dtype
    got, want = gatelib.consume(d), d.itemsize
    tag = "OK " if got == want else "NG "
    if got == want: ok += 1
    else: ng += 1
    print(f"  {tag} {str(t.__name__):>12}: consume={got} true={want} same-instance={id(d) in uniq}")
print(f"fresh-access results: {ok} pass / {ng} fail")

# Escapes: instances created AFTER the multiplexing
print("--- escapes (created after forcing) ---")
for name, d in [("structured", np.dtype([('a','i4'),('b','f8')])),
                ("newbyteorder >", np.dtype(np.float64).newbyteorder('>')),
                ("datetime64[ns]", np.dtype('datetime64[ns]')),
                ("str variable <U5", np.dtype('<U5'))]:
    print(f"  {name:>18}: consume={gatelib.consume(d)} true={d.itemsize}")
print("numpy self-use still ok:", float(np.arange(4, dtype=np.float64).sum()))
