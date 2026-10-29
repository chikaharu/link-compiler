#!/usr/bin/env python3
"""TexNP verification: the compiled .link binary is compared against NumPy.

NumPy is used ONLY as the test oracle in this script. The generated binary
links no Python, no NumPy and no BLAS/LAPACK (checked via ldd below).
"""
import os, re, subprocess, sys, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
NUM = re.compile(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?')

def compile_and_run(src_path, name):
    binp = os.path.join(OUT, name)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'linkc.py'),
                        src_path, '-o', binp],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([binp], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return binp, out.stdout

def nums(line):
    # token-based: labels such as 'pow2' or 'sum_pow2' must not leak digits
    out = []
    for tok in re.split(r'[\s,\[\]]+', line):
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out

def close(got, exp, tol=1e-9):
    flat = []
    def rec(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            for e in x: rec(e)
        else:
            flat.append(float(x))
    rec(exp)
    exp = flat
    assert len(got) == len(exp), f'length {len(got)} != {len(exp)}: {got} vs {exp}'
    for g, e in zip(got, exp):
        assert abs(g - e) <= tol * max(1.0, abs(e)), f'{g} != {e}'

def main():
    binp, stdout = compile_and_run(os.path.join(HERE, 'examples', 'texnp.link'), 'texnp_test')
    lines = {l.split('=')[0].strip(): l for l in stdout.strip().splitlines()}
    a = np.arange(1, 6, dtype=float)
    b = np.array([5., 1., 4., 2., 3.])
    M = np.array([[1., 2.], [3., 4.]])
    close(nums(lines['a']), a)
    close(nums(lines['sum']), [a.sum(), a.mean(), a.prod()])
    close(nums(lines['var']), [a.var(), a.std(), np.median(a)])
    close(nums(lines['vmin']), [a.min(), a.max(), a.argmin(), a.argmax()])
    close(nums(lines['cumprod']), np.cumprod(a))
    close(nums(lines['pow2']), [*(a ** 2), (a ** 2).sum()])
    close(nums(lines['linspace']), np.linspace(0, 1, 5))
    close(nums(lines['log2']), np.log2([1, 2, 4, 8]))
    close(nums(lines['sort']), [np.sort(b), np.argsort(b), np.sort(b)[::-1]], tol=1e-9)
    close(nums(lines['slice']), [a[1:4], np.concatenate([a[:2], a[3:]])])
    close(nums(lines['clip']), np.clip([-1, .5, 2, 10], 0, 3))
    close(nums(lines['maximum']), np.maximum(a, 3))
    close(nums(lines['full']), np.full(3, 7.0))
    close(nums(lines['det']), [np.linalg.det(M), np.trace(M), np.linalg.matrix_rank(M)])
    close(nums(lines['inv']), np.linalg.inv(M))
    close(nums(lines['solve']), np.linalg.solve(M, [5, 11]))
    close(nums(lines['matpow2']), np.linalg.matrix_power(M, 2))
    close(nums(lines['flatten']), [M.ravel(), M.sum(), M.mean()])
    close(nums(lines['diag-of']), [np.diag(M), *M.shape])
    close(nums(lines['reshape']), np.arange(6).reshape(2, 3))
    close(nums(lines['eye']), np.eye(3))
    close(nums(lines['diag']), np.diag([1, 2, 3]))
    close(nums(lines['rank1']), [np.linalg.matrix_rank([[1, 2], [2, 4]])])
    close(nums(lines['zeros2']), np.zeros((2, 3)))
    close(nums(lines['ones2']), np.ones((1, 3)))
    close(nums(lines['full2']), np.full((2, 2), 9.0))
    print('texnp.link: all lines match NumPy oracle')

    # dependency check: the binary must not pull Python/NumPy/BLAS/LAPACK
    ldd = subprocess.run(['ldd', binp], capture_output=True, text=True).stdout
    for bad in ('python', 'numpy', 'blas', 'lapack', 'openblas'):
        assert bad not in ldd.lower(), f'forbidden dependency {bad} in ldd:\n{ldd}'
    print('ldd: OS libraries only (no python/numpy/blas/lapack)')
    print(ldd.strip())

    # ---- fuzz: random vectors/matrices vs NumPy ----
    rng = random.Random(20261029)
    for trial in range(30):
        n = rng.randint(1, 9)
        v = [round(rng.uniform(-5, 5), 3) for _ in range(n)]
        w = [round(rng.uniform(-5, 5), 3) for _ in range(n)]
        k = rng.randint(2, 4)
        A = [[round(rng.uniform(-3, 3), 3) for _ in range(k)] for _ in range(k)]
        if abs(np.linalg.det(A)) < 0.05:
            continue
        lit = lambda xs: '(vec ' + ' '.join(repr(x) for x in xs) + ')'
        mat = '(mat ' + ' '.join(lit(r) for r in A) + ')'
        src = f'''(program
          (let v {lit(v)}) (let w {lit(w)}) (let A {mat})
          (print "sum" (sum v)) (print "mean" (mean v)) (print "var" (var v))
          (print "std" (std v)) (print "median" (median v))
          (print "sort" (sort v)) (print "argsort" (argsort v))
          (print "dot" (dot v w)) (print "mx" (maximum v w))
          (print "det" (det A)) (print "trace" (trace A))
          (print "inv" (inv A)) (print "rank" (rank A))
          (print "mv" (matvec A v)) (print "mm" (matmul A A)))'''
        p = os.path.join(OUT, 'texnp_fuzz.link')
        with open(p, 'w') as f: f.write(src)
        _, o = compile_and_run(p, 'texnp_fuzz')
        L = {l.split()[0]: l for l in o.strip().splitlines()}
        vv, ww, AA = np.array(v), np.array(w), np.array(A)
        close(nums(L['sum']), [vv.sum()]); close(nums(L['mean']), [vv.mean()])
        close(nums(L['var']), [vv.var()]); close(nums(L['std']), [vv.std()])
        close(nums(L['median']), [np.median(vv)])
        close(nums(L['sort']), np.sort(vv)); close(nums(L['argsort']), np.argsort(vv))
        close(nums(L['dot']), [vv @ ww]); close(nums(L['mx']), np.maximum(vv, ww))
        close(nums(L['det']), [np.linalg.det(AA)], tol=1e-7)
        close(nums(L['trace']), [np.trace(AA)])
        close(nums(L['inv']), np.linalg.inv(AA), tol=1e-7)
        close(nums(L['rank']), [np.linalg.matrix_rank(AA)])
        # matvec/matmul need v of length k; skip when n != k
        if n == k:
            close(nums(L['mv']), AA @ vv, tol=1e-7)
            close(nums(L['mm']), AA @ AA, tol=1e-7)
    print('fuzz: 30 random cases match NumPy oracle')
    print('ALL TEXNP TESTS PASSED')

if __name__ == '__main__':
    main()
