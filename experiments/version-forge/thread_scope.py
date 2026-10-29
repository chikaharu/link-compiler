"""Is time-division a thread scope? Stress test.
Variant N (naive): many threads run save->store->call->restore on the
  SAME dtype instance. Prediction: interleavings corrupt (a thread
  saves the forged bytes as 'genuine' and restores them permanently,
  or a restore lands inside another thread's window -> wrong read).
Variant L (lock-scoped): the window is owned by one thread at a time.
Variant T (twin-scoped): each thread owns its own instance (deepcopy);
  the shared singleton is never stored into at all.
Anomalies counted: wrong consume results, leftover forged bytes after
all threads finish, exceptions in a concurrent numpy reader."""
import copy, ctypes, sys, threading
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gatelib

sys.setswitchinterval(1e-6)
ctypes.c_int64.from_address(id(np.dtype)+32).value = 96   # type gate, as before

def cell(d): return ctypes.string_at(id(d)+32, 8)
def restore(d, b): ctypes.memmove(id(d)+32, b, 8)

def make_wrapped(lock=None):
    def wrapped(d):
        if lock: lock.acquire()
        try:
            saved = cell(d)
            v = ctypes.c_int64.from_address(id(d)+40).value
            ctypes.c_int32.from_address(id(d)+32).value = int(v)
            try: return gatelib.consume(d)
            finally: restore(d, saved)
        finally:
            if lock: lock.release()
    return wrapped

def stress(make_instance, wrapped, nthreads=8, iters=3000):
    wrong = [0]*nthreads; exc = []
    stop = threading.Event(); reader_hits = [0]
    def reader():
        while not stop.is_set():
            try:
                a = np.empty(64, dtype=np.float64); _ = a.itemsize
            except Exception as e:
                reader_hits[0] += 1; exc.append(repr(e))
    def worker(i):
        d = make_instance()
        w = 0
        for _ in range(iters):
            if wrapped(d) != 8: w += 1
        wrong[i] = w
    rt = threading.Thread(target=reader); rt.start()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
    for t in ts: t.start()
    for t in ts: t.join()
    stop.set(); rt.join()
    return sum(wrong), reader_hits[0]

shared = np.dtype(np.float64)
genuine = cell(shared)
print("genuine @32 bytes:", genuine.hex())

w, r = stress(lambda: shared, make_wrapped())
print(f"[N naive shared ] wrong consumes={w}  reader exceptions={r}  leftover={cell(shared).hex()} (genuine={genuine.hex()})")
restore(shared, genuine)

lock = threading.Lock()
w, r = stress(lambda: shared, make_wrapped(lock))
print(f"[L lock-scoped  ] wrong consumes={w}  reader exceptions={r}  leftover={cell(shared).hex()}")
restore(shared, genuine)

twins = [copy.deepcopy(shared) for _ in range(8)]
print("twin instances distinct:", len({id(t) for t in twins}) == 8, "| distinct from singleton:", all(id(t) != id(shared) for t in twins))
it = iter(range(8))
w, r = stress(lambda: twins[next(it)], make_wrapped())
print(f"[T twin-scoped  ] wrong consumes={w}  reader exceptions={r}  singleton untouched={cell(shared) == genuine}")
