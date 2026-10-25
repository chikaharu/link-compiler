#!/usr/bin/env python3
"""
Tests for linkir's symbolic (non-unrolled) path, v0.11: index
polynomials ('symvec') and rank-1 rankop-form matrices ('symmat').

The symbolic world has NO size cap: bench.link's 1e6-element pipeline
lowers to a handful of nodes because every operation is a closed form
(Faulhaber sums for dots, polynomial algebra for cumsum/diff).

Run: python3 test_linkir_sym.py
"""
import importlib.util
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linkc import parse
from linkir import Lower, Interp

FAILED = []

def expect(cond, label):
    print(('PASS' if cond else 'FAIL'), label)
    if not cond:
        FAILED.append(label)

def run_src(src, mode='interval'):
    prog = Lower(mode=mode).program(parse(src))
    return prog, Interp([]).run(prog)

def node_total(prog):
    from linkir import count_nodes
    return sum(count_nodes([st]) for st in prog)

# ------------------------------------------------- bench at full size
BENCH_VEC = ('(program (let v (iota 1000000.0)) (let w (cumsum v))'
             ' (print "cumsum_last =" (idx w 999999.0))'
             ' (print "roundtrip_err =" (norm (- (diff w) v))))')
prog, out = run_src(BENCH_VEC)
expect(out == 'cumsum_last = 499999500000\nroundtrip_err = 0\n',
       f'bench vector part at n=1e6 runs symbolically: {out.strip()!r}')
expect(node_total(prog) <= 4,
       f'bench vector part lowers to {node_total(prog)} nodes (no unrolling)')

with open('examples/bench.link') as f:
    bench_src = f.read()
prog = Lower(mode='interval').program(parse(bench_src))
expect(node_total(prog) < 100,
       f'bench.link fully lowers ({node_total(prog)} nodes incl. the loop)')

# -------------------------------------------- rankop cross-validation
spec = importlib.util.spec_from_file_location(
    'user_rankop',
    os.path.expanduser('~/workspace/user/files/rankop.py'))
if spec and os.path.exists(os.path.expanduser('~/workspace/user/files/rankop.py')):
    rk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rk)
    import numpy as np
    n = 500
    a = np.arange(n, dtype=float)          # iota
    b = np.cumsum(a)                       # cumsum(iota)
    op = rk.Op.from_uv(1.0, b, a)          # outer(cumsum(iota), iota)
    want = op.safe_sigma()
    _, out = run_src(f'(program (print (norm (outer (cumsum (iota {n}.0))'
                     f' (iota {n}.0)))))')
    got = float(out.strip())
    expect(abs(got - want) / want < 1e-9,
           f'symmat norm matches rankop.py log_sigma ({got} vs {want})')
else:
    print('SKIP rankop.py cross-check (file not found)')

# matmul stays rank-1 and uncapped: (a b^T)(c d^T) = (b.c) a d^T.
# n = 2000 -> the dense matrix would be 4M entries, far over the cap.
_, out = run_src('(program (let a (iota 2000.0)) (let b (ones 2000.0))'
                 ' (let m (matmul (outer a b) (outer b a)))'
                 ' (print (norm m)) (print (idx (matvec m b) 3.0)))')
# inner = b.b = 2000; m = 2000 * a a^T; ||m|| = 2000 * ||a||^2
# ||a||^2 = sum i^2 = 1999*2000*3999/6 = 2664667000 -> norm = 5.329334e12
import math
want_norm = 2000.0 * (1999 * 2000 * 3999 // 6)
# matvec(m, b) = 2000 * a * (a.b); a.b = sum i = 1999000; idx 3 -> 2000*3*1999000
want_mv = 2000 * 3 * 1999000
lines = out.strip().split('\n')
got_norm = float(lines[0])
expect(abs(got_norm - want_norm) / want_norm < 1e-12,
       f'symmat matmul norm closed form ({got_norm} vs {want_norm})')
expect(float(lines[1]) == want_mv,
       f'symmat matvec entry exact ({lines[1]} vs {want_mv})')

# dead operator: inner product zero -> matmul is the zero operator
_, out = run_src('(program (let m (matmul (outer (vec 1.0 1.0) (vec 1.0 -1.0))'
                 ' (outer (vec 1.0 1.0) (vec 5.0 6.0))))'
                 ' (print (norm m)) (print (matvec m (vec 1.0 1.0))))')
expect(out == '0\n[0.0, 0.0]\n', f'dead rankop (inner=0): {out.strip()!r}')

# ------------------------------------------------- reference evaluator
def ref_eval(e, env):
    """Evaluate the test ASTs over plain Python lists (floats)."""
    import math as m
    if isinstance(e, (int, float)):
        return float(e)
    if isinstance(e, str):
        return env[e]
    op, args = e[0], e[1:]
    if op == 'vec':
        return [ref_eval(a, env) for a in args]
    if op == 'iota':
        return [float(i) for i in range(int(ref_eval(args[0], env)))]
    if op == 'zeros':
        return [0.0] * int(ref_eval(args[0], env))
    if op == 'ones':
        return [1.0] * int(ref_eval(args[0], env))
    if op in ('+', '-', '*', '/'):
        vals = [ref_eval(a, env) for a in args]
        def comb(x, y):
            if isinstance(x, list) and isinstance(y, list):
                return [comb(p, q) for p, q in zip(x, y)]
            if isinstance(x, list):
                return [comb(p, y) for p in x]
            if isinstance(y, list):
                return [comb(x, q) for q in y]
            if op == '+':
                return x + y
            if op == '-':
                return x - y
            if op == '*':
                return x * y
            return x / y
        acc = vals[0]
        for v in vals[1:]:
            acc = comb(acc, v)
        return acc
    if op == 'neg':
        v = ref_eval(args[0], env)
        return [-x for x in v] if isinstance(v, list) else -v
    if op == 'cumsum':
        v = ref_eval(args[0], env)
        out, s = [], 0.0
        for x in v:
            s += x
            out.append(s)
        return out
    if op == 'diff':
        v = ref_eval(args[0], env)
        return [v[0]] + [v[i] - v[i - 1] for i in range(1, len(v))]
    if op == 'dot':
        a, b = ref_eval(args[0], env), ref_eval(args[1], env)
        return sum(x * y for x, y in zip(a, b))
    if op == 'norm':
        v = ref_eval(args[0], env)
        if isinstance(v[0], list):
            return m.sqrt(sum(x * x for row in v for x in row))
        return m.sqrt(sum(x * x for x in v))
    if op == 'idx':
        return ref_eval(args[0], env)[int(ref_eval(args[1], env))]
    if op == 'len':
        return float(len(ref_eval(args[0], env)))
    if op == 'outer':
        a, b = ref_eval(args[0], env), ref_eval(args[1], env)
        return [[x * y for y in b] for x in a]
    if op == 'gram':
        vs = [ref_eval(a, env) for a in args]
        return [[sum(x * y for x, y in zip(u, w)) for w in vs] for u in vs]
    if op == 'matvec':
        M, v = ref_eval(args[0], env), ref_eval(args[1], env)
        return [sum(x * y for x, y in zip(row, v)) for row in M]
    if op == 'matmul':
        A, B = ref_eval(args[0], env), ref_eval(args[1], env)
        return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
                 for j in range(len(B[0]))] for i in range(len(A))]
    if op == 'transpose':
        M = ref_eval(args[0], env)
        return [list(r) for r in zip(*M)]
    raise ValueError(op)

rng = random.Random(20261026)
def rand_sym_expr(n, depth):
    base = rng.choice([['iota', float(n)], ['ones', float(n)],
                       ['zeros', float(n)]])
    e = base
    for _ in range(depth):
        c = rng.randrange(6)
        if c == 0:
            e = ['cumsum', e]
        elif c == 1:
            e = ['diff', e]
        elif c == 2:
            e = ['+', e, ['iota', float(n)]]
        elif c == 3:
            e = ['*', e, rng.choice([2.0, 0.5, -3.0])]
        elif c == 4:
            e = ['+', e, rng.choice([1.5, -2.25])]
        else:
            e = ['-', ['cumsum', ['iota', float(n)]], e]
    return e

def to_lisp(x):
    if isinstance(x, list):
        return '(' + ' '.join(to_lisp(y) for y in x) + ')'
    if isinstance(x, float):
        return repr(x)
    return str(x)

ok_all = True
for trial in range(40):
    n = rng.choice([1, 2, 5, 17, 40])
    e = rand_sym_expr(n, rng.randint(1, 4))
    probe = rng.choice(['vec', 'dot', 'norm', 'idx'])
    if probe == 'vec':
        want = '[' + ', '.join(Interp.fmt_debug(x)
                               for x in ref_eval(e, {})) + ']\n'
    elif probe == 'dot':
        want = Interp.fmt(ref_eval(['dot', e, ['iota', float(n)]], {})) + '\n'
    elif probe == 'norm':
        want = Interp.fmt(ref_eval(['norm', e], {})) + '\n'
    else:
        k = rng.randrange(n)
        want = Interp.fmt(ref_eval(['idx', e, float(k)], {})) + '\n'
    if probe == 'vec':
        src = f'(program (print {to_lisp(e)}))'
    elif probe == 'dot':
        src = f'(program (print (dot {to_lisp(e)} (iota {float(n)}))))'
    elif probe == 'norm':
        src = f'(program (print (norm {to_lisp(e)})))'
    else:
        src = f'(program (print (idx {to_lisp(e)} {float(k)})))'
    _, got = run_src(src)
    if got != want:
        ok_all = False
        print(f'  mismatch trial {trial}: got {got!r} want {want!r} src {src}')
expect(ok_all, 'random symbolic expressions match the reference evaluator (40 cases)')

# matrix ops cross-check at small size (symbolic outer/matmul paths)
for trial in range(10):
    n = rng.choice([2, 3, 7])
    ea, eb = rand_sym_expr(n, 2), rand_sym_expr(n, 2)
    expr = ['matvec', ['matmul', ['outer', ea, eb], ['outer', eb, ea]], ea]
    src = f'(program (print {to_lisp(expr)}))'
    _, got = run_src(src)
    ref = ref_eval(expr, {})
    want = '[' + ', '.join(Interp.fmt_debug(x) for x in ref) + ']\n'
    expect(got == want, f'symmat chain trial {trial} matches reference')

# degree cap is explicit
try:
    e = '(iota 10.0)'
    for _ in range(14):
        e = f'(cumsum {e})'
    run_src(f'(program (print (norm {e})))')
    expect(False, 'degree cap raises')
except TypeError as ex:
    expect('degree' in str(ex), f'degree cap raises ({ex})')

# mixed world: symbolic + variable vector materialises (small n)
prog = Lower(mode='interval').program(parse(
    '(program (let x 10.0) (print (+ (iota 3.0) (vec x x x))))'))
out = Interp([]).run(prog)
expect(out == '[10.0, 11.0, 12.0]\n', f'mixed sym/vec materialises: {out.strip()!r}')

print()
if FAILED:
    print(f'{len(FAILED)} FAILURES')
    sys.exit(1)
print('all tests passed')
