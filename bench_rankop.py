#!/usr/bin/env python3
"""rankop optimisation benchmark for TexNP (linkc).

Same mathematical questions, four ways:
  TexNP factored : (outer u v) -- the compiler keeps factors, O(n)/O(n^2)
  TexNP dense    : (+ A (zeros2 n n)) forces the dense path, same values
  NumPy dense    : np.outer materialised, then dense kernels
  NumPy manual   : a human who knows the structure writes it with vectors
Outputs are cross-checked before any timing is reported.
"""
import os, re, statistics, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

def nums(text):
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
    name = re.sub(r'[^A-Za-z0-9]+', '_', name)
    p = os.path.join(OUT, f'bench_ro_{name}.link')
    with open(p, 'w') as f:
        f.write(src)
    binp = os.path.join(OUT, f'bench_ro_{name}')
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
    return statistics.median(times), nums(out)

def time_numpy(fn, reps):
    res = fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), list(np.asarray(res, dtype=float).ravel())

def uvx(n):
    """Deterministic pseudo-random-ish vectors, same in .link and NumPy."""
    link = f'''(let x (arange {n}.0))
    (let u (- x (* (floor (/ x 97.0)) 97.0)))
    (let v (- x (* (floor (/ x 89.0)) 89.0)))'''
    x = np.arange(n, dtype=float)
    u = x - np.floor(x / 97.0) * 97.0
    v = x - np.floor(x / 89.0) * 89.0
    return link, u, v, x

def check(got, exp, name):
    assert len(got) == len(exp), (name, got, exp)
    for g, e in zip(got, exp):
        assert g == e or abs(g - e) <= 1e-6 * max(1.0, abs(e)), (name, g, e)

def run_case(name, n, link_tail, np_dense, np_manual, reps=3):
    link_pre, u, v, x = uvx(n)
    src_f = f'(program {link_pre} (let A (outer u v)) {link_tail})'
    src_d = (f'(program {link_pre} (let A (+ (outer u v) '
             f'(zeros2 {n}.0 {n}.0))) {link_tail})')
    bf, bd = build(f'{name}_f', src_f), build(f'{name}_d', src_d)
    tf, got = time_binary(bf, reps)
    td, gotd = time_binary(bd, reps)
    tn, exp = time_numpy(np_dense, reps)
    tm, expm = time_numpy(np_manual, reps)
    check(got, exp, name); check(gotd, exp, name + ' dense'); check(got, expm, name + ' manual')
    print(f'{name} n={n}: TexNP factored {tf:.4f}s | TexNP dense {td:.4f}s '
          f'(x{td / tf:.1f}) | NumPy dense {tn:.4f}s (x{tn / tf:.1f}) '
          f'| NumPy manual {tm:.4f}s')
    return tf, td, tn, tm

def main():
    print('--- consumers of a rank-1 matrix A = u v^T ---')
    run_case('matvec', 5000,
             '(print (sum (matvec A x)))',
             lambda: (lambda u, v, x: (np.outer(u, v) @ x).sum())(*uvx(5000)[1:]),
             lambda: (lambda u, v, x: u.sum() * (v @ x))(*uvx(5000)[1:]))
    run_case('trace+sum', 5000,
             '(print (trace A) (sum A))',
             lambda: (lambda u, v, x: [np.trace(np.outer(u, v)),
                                       np.outer(u, v).sum()])(*uvx(5000)[1:]),
             lambda: (lambda u, v, x: [u @ v, u.sum() * v.sum()])(*uvx(5000)[1:]))
    run_case('det', 800,
             '(print (det A))',
             lambda: (lambda u, v, x: np.linalg.det(np.outer(u, v)))(*uvx(800)[1:]),
             lambda: 0.0)
    run_case('matmul A@A -> trace+sum', 700,
             '(let C (matmul A A)) (print (trace C) (sum C))',
             lambda: (lambda u, v, x: (lambda A: [np.trace(A @ A), (A @ A).sum()])(
                 np.outer(u, v)))(*uvx(700)[1:]),
             lambda: (lambda u, v, x: [(v @ u) * (u @ v),
                                       (v @ u) * u.sum() * v.sum()])(*uvx(700)[1:]))
    print('--- factored only: sizes where dense materialisation is impossible ---')
    for n in (5000, 20000, 100000):
        link_pre, u, v, x = uvx(n)
        src = (f'(program {link_pre} (let A (outer u v)) '
               f'(print (trace A) (sum (matvec A x)) (det A)))')
        b = build(f'scale_{n}', src)
        t, got = time_binary(b, 3)
        check(got, [u @ v, u.sum() * (v @ x), 0.0], f'scale {n}')
        print(f'n={n}: factored answers in {t:.4f}s '
              f'(dense matrix would be {8.0 * n * n / 1e9:.1f} GB)')

if __name__ == '__main__':
    main()
