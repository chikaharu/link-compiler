"""Force-store attack on a verification gate.

The gate (gatelib.check / consume) does not receive values as arguments;
it *loads* them from live objects in memory:
  check   loads tp_basicsize from the np.dtype type object,
  consume loads elsize from offset 32 (1.x layout) of a descr instance.
So instead of patching the checker (binary forgery) or lying about the
version (string forgery), we STORE the expected values into exactly the
slots the checker loads from -- enumerating slots exhaustively, one
trial per fresh subprocess (bad stores crash; the loop survives),
until the gate passes. Stores accumulate across phases.
"""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = r'''
import ctypes, sys
import numpy as np
stores = eval(sys.argv[1])          # list of (object_tag, offset, size, value)
dt = np.dtype(np.float64)
objs = {"type": np.dtype, "inst": dt}
for tag, off, size, val in stores:
    addr = id(objs[tag]) + off
    if size == 8: ctypes.c_int64.from_address(addr).value = val
    else:         ctypes.c_int32.from_address(addr).value = val
sys.path.insert(0, %r)
import gatelib                      # the ORIGINAL, unpatched extension
mode = sys.argv[2]
try:
    if mode == "check":
        print("RESULT check =", gatelib.check(np.dtype))
    elif mode == "consume":
        print("RESULT consume =", gatelib.consume(dt), "true itemsize =", dt.itemsize)
    elif mode == "aftermath":
        print("RESULT check =", gatelib.check(np.dtype))
        print("RESULT consume(f64) =", gatelib.consume(dt), "true =", dt.itemsize)
        dt2 = np.dtype(np.int32)    # a fresh descr instance nobody stored into
        print("RESULT consume(fresh int32) =", gatelib.consume(dt2), "true =", dt2.itemsize)
        a = np.zeros(3, dtype=dt)   # does numpy itself still work on the forced object?
        print("RESULT numpy self-use ok:", float(a.sum()), a.dtype.itemsize)
except Exception as e:
    print("RESULT FAIL:", type(e).__name__, str(e)[:120])
''' % HERE

def run(stores, mode):
    env = dict(os.environ, PYTHONPATH="/tmp/np2")
    r = subprocess.run([sys.executable, "-c", RUNNER, repr(stores), mode],
                       capture_output=True, text=True, timeout=60, env=env)
    out = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    return out if out else [f"<crash rc={r.returncode}>"]

def slots_with_value(tag, obj_desc, value, size, limit):
    """Ask a probe process which slots of the object currently hold `value`."""
    probe = r'''
import ctypes, numpy as np
dt = np.dtype(np.float64)
obj = {"type": np.dtype, "inst": dt}[%r]
vals = []
for off in range(0, %d, %d):
    fmt = ctypes.c_int64 if %d == 8 else ctypes.c_int32
    if fmt.from_address(id(obj)+off).value == %d: vals.append(off)
print("SLOTS", vals)
''' % (tag, limit, size, size, value)
    env = dict(os.environ, PYTHONPATH="/tmp/np2")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith("SLOTS")][0]
    return eval(line[6:])

if __name__ == "__main__":
    stores = []
    print("[Phase 1] gate = check(): it loads tp_basicsize (runtime 88) -- store 96 into the slot it reads")
    trials = 0
    for off in slots_with_value("type", "np.dtype type object", 88, 8, 208):
        trials += 1
        out = run(stores + [("type", off, 8, 96)], "check")
        print(f"  trial {trials}: store(type+{off}, 96) -> {out}")
        if any("check = 96" in l for l in out):
            stores.append(("type", off, 8, 96)); break
    print(f"  gate passed after {trials} trial(s); accumulated stores: {stores}")
    print("[Phase 2] consumer = consume(): loads elsize at 1.x offset from the descr instance -- store 8 slot by slot")
    trials = 0
    for off in range(0, 100, 4):
        trials += 1
        out = run(stores + [("inst", off, 4, 8)], "consume")
        print(f"  trial {trials}: store(inst+{off}, int32 8) -> {out}")
        if any("consume = 8" in l for l in out):
            stores.append(("inst", off, 4, 8)); break
    print(f"  consumer passed after {trials} trial(s)")
    print("[Phase 3] aftermath in the fully forced process:")
    for l in run(stores, "aftermath"):
        print(" ", l)
