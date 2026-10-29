"""Dynamic program that wraps errors and infers the required format.

Brute force (force_store.py) treated every failure as waste: enumerate
slots, crash twice, succeed per-instance. This program instead wraps
each error as a *carrier* of format information:

  Stage A (demand): store a distinct marker (= its own offset) into
    every candidate slot of one instance, then call the consumer once.
    The "wrong" value it returns is the marker echo -> it names the
    exact slot the consumer loads from. One trial, no enumeration.
    A second probe with a 64-bit marker measures the read width.
  Stage B (supply): read all slots of many dtype instances and build
    the error matrix E[d, s] = read_s(d) - truth(d). Accumulate the
    outer products G = E^T E (the error Gram over slots). A slot whose
    error column vanishes against itemsize IS the elsize slot; against
    alignment, the alignment slot; against type number, the type_num
    slot. Nonzero errors are not discarded: their support pattern
    (e.g. nonzero only for complex dtypes) identifies the field.
  Stage C (program output): the inferred format is a translation RULE
    (load int64 @src -> store int32 @dst), not a pile of values. Apply
    it to instances that escaped multiplexing and validate.
"""
import ctypes, subprocess, sys, os
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gatelib

def i32(addr, off): return ctypes.c_int32.from_address(addr+off).value
def put_i32(addr, off, v): ctypes.c_int32.from_address(addr+off).value = v
def put_i64(addr, off, v): ctypes.c_int64.from_address(addr+off).value = v

STAGE_A = r'''
import ctypes, sys
sys.path.insert(0, "__HERE__")
import numpy as np, gatelib
dt = np.dtype(np.float64)
mode = sys.argv[1]
if mode == "markers":
    for off in range(16, 100, 4):
        ctypes.c_int32.from_address(id(dt)+off).value = off   # marker = own offset
    print("ECHO", gatelib.consume(dt), flush=True)
else:  # width
    ctypes.c_int64.from_address(id(dt)+32).value = (1 << 32) | 7
    print("WIDTH-ECHO", gatelib.consume(dt), flush=True)
'''

def stage_a():
    env = dict(os.environ, PYTHONPATH="/tmp/np2")
    r1 = subprocess.run([sys.executable, "-c", STAGE_A.replace("__HERE__", HERE), "markers"], capture_output=True, text=True, env=env)
    echo = [l for l in r1.stdout.splitlines() if l.startswith("ECHO")]
    dst = int(echo[0].split()[1]) if echo else None
    r2 = subprocess.run([sys.executable, "-c", STAGE_A.replace("__HERE__", HERE), "width"], capture_output=True, text=True, env=env)
    wecho = [l for l in r2.stdout.splitlines() if l.startswith("WIDTH-ECHO")]
    width = 4 if wecho and int(wecho[0].split()[1]) == 7 else 8
    print(f"[Stage A] marker echo -> consumer loads offset {dst}; 64-bit marker echo -> read width {width} bytes")
    print(f"          (probe subprocess rc: markers={r1.returncode}, width={r2.returncode} -- a post-result crash is wrapped as info: header slots are load-bearing)")
    return dst, width

def stage_b():
    TYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
             np.float16, np.float32, np.float64, np.longdouble, np.complex64, np.complex128, bool, np.object_]
    dts = [np.dtype(t) for t in TYPES]
    slots = list(range(0, 100, 4))
    R = np.array([[i32(id(d), s) for s in slots] for d in dts], dtype=np.int64)
    item = np.array([d.itemsize for d in dts]); ali = np.array([d.alignment for d in dts]); num = np.array([d.num for d in dts])
    E_item = R - item[:, None]; E_ali = R - ali[:, None]; E_num = R - num[:, None]
    G = E_item.T @ E_item   # error Gram: sum of outer products of error columns
    print("[Stage B] error-Gram classification of slots (E = read - truth, over 16 dtypes):")
    src = {}
    for j, s in enumerate(slots):
        if np.all(E_item[:, j] == 0): lbl = "ELSIZE (error vanishes vs itemsize)"; src['elsize'] = s
        elif np.all(E_ali[:, j] == 0): lbl = "alignment (error vanishes vs alignment)"; src['alignment'] = s
        elif np.all(E_num[:, j] == 0): lbl = "type_num (error vanishes vs dtype.num)"; src['num'] = s
        elif np.all(R[:, j] == 0): lbl = "hole (reads 0 for every dtype)"
        elif np.max(np.abs(R[:, j])) > 2**31: lbl = "pointer-like (huge values)"
        else:
            nz = [TYPES[k].__name__ for k in range(len(dts)) if E_item[k, j] != 0]
            lbl = f"other field (itemsize-error support: {nz[:4]}{'...' if len(nz) > 4 else ''})"
        print(f"  slot {s:>3}: ||E_item||^2 = {G[j, j]:>14}  -> {lbl}")
    print(f"          inferred supply layout: elsize int64 @ {src['elsize']}, alignment @ {src['alignment']}, type_num @ {src['num']}")
    return src['elsize']

def stage_c(dst, src):
    def translate(d):  # the inferred RULE: load int64 @src, store int32 @dst
        v = ctypes.c_int64.from_address(id(d)+src).value
        put_i32(id(d), dst, int(v))
        return v
    print(f"[Stage C] inferred translation rule: store int32 @{dst} <- load int64 @{src}")
    cases = [("structured (escaped)", np.dtype([('a','i4'),('b','f8')])),
             ("datetime64[ns] (escaped)", np.dtype('datetime64[ns]')),
             ("<U5 variable (escaped)", np.dtype('<U5')),
             ("<S3 variable (fresh)", np.dtype('<S3')),
             ("newbyteorder >f8", np.dtype(np.float64).newbyteorder('>')),
             ("float64 (control)", np.dtype(np.float64))]
    for name, d in cases:
        translate(d)
        got = gatelib.consume(d)
        print(f"  {'OK ' if got == d.itemsize else 'NG '} {name:>26}: consume={got} true={d.itemsize}")

if __name__ == "__main__":
    dst, width = stage_a()
    src = stage_b()
    stage_c(dst, src)
    print("cost: 2 store-probes + read-only scans, vs brute force 10 trials / 2 segfaults / per-instance values only")
