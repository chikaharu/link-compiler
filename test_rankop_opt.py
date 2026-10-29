#!/usr/bin/env python3
"""rankop optimisation in linkc: rank-1 factorised matrices must give the
same results as the dense path (and as NumPy), while avoiding O(n^3) work.
The dense path is forced with (+ A (zeros2 n n)), which the factoriser
deliberately does not see through.
"""
import os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')

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

def run(src, name):
    p = os.path.join(OUT, name + '.link')
    with open(p, 'w') as f:
        f.write(src)
    binp = os.path.join(OUT, name)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'linkc.py'), p, '-o', binp],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    o = subprocess.run([binp], capture_output=True, text=True)
    assert o.returncode == 0, o.stderr
    return o.stdout

def close(got, exp, tol=1e-8):
    flat = []
    def rec(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            for e in x: rec(e)
        else:
            flat.append(float(x))
    rec(exp)
    exp = flat
    assert len(got) == len(exp), f'{got} vs {exp}'
    for g, e in zip(got, exp):
        assert abs(g - e) <= tol * max(1.0, abs(e)), f'{g} != {e}'

def main():
    rng = np.random.default_rng(20261029)
    for n in (1, 2, 5, 17):
        u, v, c, d, x = (rng.normal(size=n) for _ in range(5))
        lit = lambda xs: '(vec ' + ' '.join(repr(float(z)) for z in xs) + ')'
        body = f'''
          (let A (outer u v)) (let B (outer c d))
          (print "mv" (matvec A x))
          (print "mm" (matmul A B))
          (print "tr" (trace A) (sum A) (mean A))
          (print "det" (det A))
          (print "tp" (transpose A))
          (print "sc" (matvec (* 3.0 A) x))
          (let C (matmul A B))
          (print "chain" (trace C) (matvec C x) (sum (matmul C A)))
          (let Ad (+ A (zeros2 {n}.0 {n}.0))) (let Bd (+ B (zeros2 {n}.0 {n}.0)))
          (print "dmv" (matvec Ad x))
          (print "dmm" (matmul Ad Bd))
          (print "dtr" (trace Ad) (sum Ad) (mean Ad))
          (print "ddet" (det Ad))
          (print "dtp" (transpose Ad))
          (print "dsc" (matvec (* 3.0 Ad) x))
          (let Cd (matmul Ad Bd))
          (print "dchain" (trace Cd) (matvec Cd x) (sum (matmul Cd Ad)))'''
        src = (f'(program (let u {lit(u)}) (let v {lit(v)}) (let c {lit(c)}) '
               f'(let d {lit(d)}) (let x {lit(x)}) {body})')
        out = run(src, f'rankop_opt_{n}')
        L = {l.split()[0].strip('"'): l for l in out.strip().splitlines()}
        A, B = np.outer(u, v), np.outer(c, d)
        C = A @ B
        exp = {
            'mv': A @ x, 'mm': A @ B,
            'tr': [np.trace(A), A.sum(), A.mean()],
            'det': [0.0 if n > 1 else A[0, 0]],
            'tp': A.T, 'sc': (3 * A) @ x,
            'chain': [np.trace(C), C @ x, (C @ A).sum()],
        }
        dense = {'dmv': 'mv', 'dmm': 'mm', 'dtr': 'tr', 'ddet': 'det',
                 'dtp': 'tp', 'dsc': 'sc', 'dchain': 'chain'}
        for k, e in exp.items():
            close(nums(L[k]), e, tol=1e-7)
        for dk, k in dense.items():
            got, ref = nums(L[dk]), nums(L[k])
            assert len(got) == len(ref)
            for g, r_ in zip(got, ref):
                assert abs(g - r_) <= 1e-7 * max(1.0, abs(r_)), f'{dk}: {g} != {r_}'
        print(f'n={n}: factored == dense == NumPy')
    # mixed factored x dense (both orders)
    n = 6
    u, v, x = (rng.normal(size=n) for _ in range(3))
    M = rng.normal(size=(n, n))
    lit = lambda xs: '(vec ' + ' '.join(repr(float(z)) for z in xs) + ')'
    mat = '(mat ' + ' '.join(lit(r) for r in M) + ')'
    src = f'''(program (let u {lit(u)}) (let v {lit(v)}) (let x {lit(x)})
      (let A (outer u v)) (let M {mat})
      (print "am" (matmul A M)) (print "ma" (matmul M A))
      (print "amv" (matvec (matmul A M) x)))'''
    out = run(src, 'rankop_opt_mixed')
    L = {l.split()[0].strip('"'): l for l in out.strip().splitlines()}
    A = np.outer(u, v)
    close(nums(L['am']), A @ M, tol=1e-7)
    close(nums(L['ma']), M @ A, tol=1e-7)
    close(nums(L['amv']), (A @ M) @ x, tol=1e-7)
    print('mixed factored x dense == NumPy')
    # lazy binding: a 20000x20000 rank-1 matrix is never materialised
    # (dense would need 3.2 GB); only the two factor vectors exist.
    n = 20000
    src = f'''(program
      (let x (arange {n}.0))
      (let u (- x (* (floor (/ x 97.0)) 97.0)))
      (let v (- x (* (floor (/ x 89.0)) 89.0)))
      (let A (outer u v))
      (print "big" (trace A) (sum A) (det A) (sum (matvec A x))))'''
    out = run(src, 'rankop_opt_big')
    L = {l.split()[0].strip('"'): l for l in out.strip().splitlines()}
    xx = np.arange(n, dtype=float)
    uu = xx - np.floor(xx / 97.0) * 97.0
    vv = xx - np.floor(xx / 89.0) * 89.0
    close(nums(L['big']),
          [uu @ vv, uu.sum() * vv.sum(), 0.0, uu.sum() * (vv @ xx)], tol=1e-7)
    print('lazy: 20000x20000 rank-1 answered from factors only')
    print('ALL RANKOP-OPT TESTS PASSED')

if __name__ == '__main__':
    main()
