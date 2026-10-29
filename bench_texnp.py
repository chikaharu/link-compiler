#!/usr/bin/env python3
"""TexNP vs NumPy benchmark.

TexNP side: a .link program compiled by linkc to a native binary; the timed
value is the whole process wall time (startup included), median of N runs.
NumPy side: in-process compute time only (interpreter + import excluded),
median of N runs after one warmup. Outputs are cross-checked for agreement
before timing is reported, so a fast-but-wrong result cannot win.
"""
import os, re, statistics, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

def tokens_nums(text):
    out = []
    for tok in re.split(r'[\s,\[\]=]+', text):
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out

def build(name, src):
    p = os.path.join(OUT, f'bench_{name}.link')
    with open(p, 'w') as f:
        f.write(src)
    binp = os.path.join(OUT, f'bench_{name}')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'linkc.py'), p, '-o', binp],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return binp

def time_binary(binp, reps):
    times, out = [], ''
    for _ in range(reps):
        t = time.perf_counter()
        r = subprocess.run([binp], capture_output=True, text=True)
        times.append(time.perf_counter() - t)
        out = r.stdout
    return statistics.median(times), tokens_nums(out)

def time_numpy(fn, reps):
    res = fn()  # warmup + result
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), list(np.asarray(res, dtype=float).ravel())

# ---------------------------------------------------------------- benches
N = 1_000_000
VEC_PRE = f'(let v (arange {N}.0))'
PSEUDO = f'''(let x (arange {N}.0)) (let y (* x 7919.0))
  (let r (- y (* (floor (/ y 1000003.0)) 1000003.0)))'''
def mat_src(n, tail):
    return f'''(program
      (let A0 (reshape (arange {n * n}.0) {n}.0 {n}.0))
      (let A (+ (* A0 0.001) (* (eye {n}.0) 100.0)))
      {tail})'''
def mat_np(n):
    A0 = np.arange(n * n, dtype=float).reshape(n, n)
    return A0 * 0.001 + np.eye(n) * 100.0

BENCHES = [
    ('sqrt_sum 1e6',
     f'(program {VEC_PRE} (print (sum (sqrt v))))',
     lambda: np.sqrt(np.arange(N, dtype=float)).sum(), 5, 5),
    ('dot 1e6',
     f'(program {VEC_PRE} (print (dot v v)))',
     lambda: (lambda v: v @ v)(np.arange(N, dtype=float)), 5, 5),
    ('cumsum+sum 1e6',
     f'(program {VEC_PRE} (print (sum (cumsum v))))',
     lambda: np.cumsum(np.arange(N, dtype=float)).sum(), 5, 5),
    ('mean/var/std 1e6',
     f'(program {VEC_PRE} (print (mean v) (var v) (std v)))',
     lambda: (lambda v: [v.mean(), v.var(), v.std()])(np.arange(N, dtype=float)), 5, 5),
    ('sort 1e6 (pseudo-random)',
     f'(program {PSEUDO} (let s (sort r)) (print (sum s) (idx s 500000.0)))',
     lambda: (lambda r: [np.sort(r).sum(), np.sort(r)[500000]])(
         (lambda x: x * 7919.0 - np.floor(x * 7919.0 / 1000003.0) * 1000003.0)(
             np.arange(N, dtype=float))), 5, 5),
    ('argsort 1e6 (pseudo-random)',
     f'(program {PSEUDO} (print (sum (argsort r))))',
     lambda: (lambda x: np.argsort(
         x * 7919.0 - np.floor(x * 7919.0 / 1000003.0) * 1000003.0).sum())(
             np.arange(N, dtype=float)), 5, 5),
    ('matmul 200x200',
     mat_src(200, '(let B (matmul A A)) (print (trace B) (sum B))'),
     lambda: (lambda A: [np.trace(A @ A), (A @ A).sum()])(mat_np(200)), 5, 5),
    ('matmul 500x500',
     mat_src(500, '(let B (matmul A A)) (print (trace B) (sum B))'),
     lambda: (lambda A: [np.trace(A @ A), (A @ A).sum()])(mat_np(500)), 5, 5),
    ('inv 300x300',
     mat_src(300, '(print (sum (inv A)))'),
     lambda: np.linalg.inv(mat_np(300)).sum(), 3, 3),
    ('det 300x300',
     ('(program (let A0 (reshape (arange 90000.0) 300.0 300.0)) '
      '(let A (+ (* A0 0.001) (* (eye 300.0) 2.0))) (print (det A)))'),
     lambda: np.linalg.det(np.arange(90000, dtype=float).reshape(300, 300) * 0.001
                           + np.eye(300) * 2.0), 3, 5),
    ('solve 300x300',
     mat_src(300, f'(print (sum (solve A (arange 300.0))))'),
     lambda: np.linalg.solve(mat_np(300), np.arange(300, dtype=float)).sum(), 3, 5),
    ('matvec 500x500',
     mat_src(500, '(print (sum (matvec A (arange 500.0))))'),
     lambda: (mat_np(500) @ np.arange(500, dtype=float)).sum(), 5, 5),
]

def main():
    t = time.perf_counter()
    subprocess.run([sys.executable, '-c', 'import numpy'], capture_output=True)
    np_startup = time.perf_counter() - t
    print(f'environment: numpy {np.__version__}, '
          f'python+numpy startup (process-level, not in numpy column): {np_startup:.3f}s')
    print(f'{"benchmark":28s} {"TexNP(s)":>10s} {"NumPy(s)":>10s} {"ratio":>7s}  check')
    rows = []
    for name, src, fn, rb, rn in BENCHES:
        safe = re.sub(r'[^A-Za-z0-9]+', '_', name)
        binp = build(safe + '_' + str(abs(hash(name)) % 10**8), src)
        tb, got = time_binary(binp, rb)
        tn, exp = time_numpy(fn, rn)
        ok = len(got) == len(exp) and all(
            g == e or abs(g - e) <= 1e-6 * max(1.0, abs(e)) for g, e in zip(got, exp))
        ratio = tn / tb if tb > 0 else float('nan')
        rows.append((name, tb, tn, ratio))
        print(f'{name:28s} {tb:10.4f} {tn:10.4f} {ratio:6.2f}x  '
              f'{"OK" if ok else "MISMATCH got=" + str(got) + " exp=" + str(exp)}')
        assert ok, name
    print()
    print('ratio > 1 means TexNP faster (against in-process NumPy compute time;')
    print('TexNP time additionally includes its own process startup).')

if __name__ == '__main__':
    main()
