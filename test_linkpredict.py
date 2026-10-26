#!/usr/bin/env python3
"""
Tests for linkpredict: static TRANSPARENT / DESTRUCTIVE / UNKNOWN
prediction for set-idx on symbolic vectors, checked against ground
truth (the program run with and without the write).

Run: python3 test_linkpredict.py
"""
import copy
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linkc import parse
from linkir import Lower, Interp
from linkpredict import predict_program

FAILED = []

def expect(cond, label):
    print(('PASS' if cond else 'FAIL'), label)
    if not cond:
        FAILED.append(label)

def run(stmts):
    prog = Lower(mode='interval').program(stmts)
    return Interp([]).run(prog)

def without_write(stmts):
    out = []
    for st in stmts:
        if isinstance(st, list) and st and st[0] == 'set-idx':
            continue
        out.append(st)
    return out

def check_case(label, src, want_verdict, want_delta=None):
    stmts = parse(src)
    sites = predict_program(stmts)
    ok = len(sites) == 1 and sites[0]['verdict'] == want_verdict
    detail = sites[0] if sites else {}
    if want_delta is not None:
        preds = [d for k, d in detail.get('predicted', []) if d is not None]
        ok &= preds == [want_delta]
    truth_same = run(stmts) == run(without_write(stmts))
    if want_verdict == 'TRANSPARENT':
        ok &= truth_same
    if want_verdict == 'DESTRUCTIVE':
        ok &= not truth_same
    expect(ok, f'{label}: verdict={detail.get("verdict")} '
               f'delta={detail.get("delta")} (want {want_verdict})')

V = '(let v (iota 5.0))'
W0 = '(let w (vec 1.0 1.0 0.0 1.0 1.0))'   # zero at slot 2
W1 = '(let w (vec 1.0 1.0 1.0 1.0 1.0))'

check_case('delta = 0 write',
           f'(program {V} (set-idx v 2.0 2.0) (print (dot v w)))'
           .replace('(dot v w)', '(dot v (vec 1.0 1.0 1.0 1.0 1.0))'),
           'TRANSPARENT')
check_case('kappa = 0 consumer (w has a zero at the slot)',
           f'(program {V} {W0} (set-idx v 2.0 99.0) (print (dot v w)))',
           'TRANSPARENT')
check_case('kappa != 0 consumer, exact Delta',
           f'(program {V} {W1} (set-idx v 2.0 99.0) (print (dot v w)))',
           'DESTRUCTIVE', Fraction(97))
check_case('printed vector changes',
           f'(program {V} (set-idx v 2.0 99.0) (print v))',
           'DESTRUCTIVE')
check_case('dead write',
           f'(program {V} (set-idx v 2.0 99.0) (print 42.0))',
           'TRANSPARENT')
check_case('idx at another slot',
           f'(program {V} (set-idx v 2.0 99.0) (print (idx v 3.0)))',
           'TRANSPARENT')
check_case('idx at the written slot',
           f'(program {V} (set-idx v 2.0 99.0) (print (idx v 2.0)))',
           'DESTRUCTIVE', Fraction(97))
check_case('cumsum consumer, weights end before the slot tail',
           f'(program {V} (let u (vec 1.0 1.0 0.0 0.0 0.0))'
           ' (set-idx v 2.0 99.0) (print (dot (cumsum v) u)))',
           'TRANSPARENT')
check_case('cumsum consumer, tail weights nonzero',
           f'(program {V} {W1} (set-idx v 2.0 99.0)'
           ' (print (dot (cumsum v) w)))',
           'DESTRUCTIVE', Fraction(3 * 97))
check_case('derived binding (let u (cumsum v)) then dot',
           f'(program {V} {W1} (set-idx v 2.0 99.0) (let u (cumsum v))'
           ' (print (dot u w)))',
           'DESTRUCTIVE', Fraction(3 * 97))
check_case('unknown written value, zero-gain consumer',
           f'(program {V} {W0} (set-idx v 2.0 (read-int))'
           ' (print (dot v w)))',
           'TRANSPARENT')
check_case('unknown written value, printed vector',
           f'(program {V} (set-idx v 2.0 (read-int)) (print v))',
           'UNKNOWN')
check_case('feedback write (value from another slot)',
           f'(program {V} {W1} (set-idx v 2.0 (idx v 4.0))'
           ' (print (dot v w)))',
           'DESTRUCTIVE', Fraction(2))
check_case('norm consumer abstains',
           f'(program {V} (set-idx v 2.0 99.0) (print (norm v)))',
           'DESTRUCTIVE')

# --- v0.14: delta-polynomial consumers and exact norm -------------
# norm: Delta(||v||^2) = 2*delta*v[2] + delta^2 = 2*97*2 + 97^2
_stmts = parse(f'(program {V} (set-idx v 2.0 99.0) (print (norm v)))')
_s = predict_program(_stmts)
_d2 = [d for k, d in _s[0].get('predicted', []) if k == 'norm2']
expect(_s[0]['verdict'] == 'DESTRUCTIVE' and _d2 == [Fraction(9797)],
       f'norm: exact Delta(||v||^2) = {_d2} (want [9797])')
# a sign flip of one element provably preserves the norm:
# delta = -4 at slot 2 (value 2 -> -2): 2*(-4)*2 + 16 = 0
check_case('norm sign-flip is transparent',
           f'(program {V} (set-idx v 2.0 -2.0) (print (norm v)))',
           'TRANSPARENT')
# (dot v w)^2: the delta^2 cross term is now exact.
# natural dot = 10, gain = 1 -> Delta = (10+97)^2 - 10^2 = 11349
_stmts = parse(f'(program {V} {W1} (set-idx v 2.0 99.0)'
               ' (print (* (dot v w) (dot v w))))')
_s = predict_program(_stmts)
_ds = [d for k, d in _s[0].get('predicted', []) if k == 'scalar']
expect(_s[0]['verdict'] == 'DESTRUCTIVE' and _ds == [Fraction(11349)],
       f'(dot)^2 consumer: exact Delta = {_ds} (want [11349])')
# dot with itself: Delta(||v||^2) through the dot path
_stmts = parse(f'(program {V} (set-idx v 2.0 99.0) (print (dot v v)))')
_s = predict_program(_stmts)
_ds = [d for k, d in _s[0].get('predicted', []) if k == 'scalar']
expect(_s[0]['verdict'] == 'DESTRUCTIVE' and _ds == [Fraction(9797)],
       f'dot(v,v): exact Delta = {_ds} (want [9797])')
# analytic consumers at a known delta: sqrt(||v||^2) and exp(idx)
_stmts = parse(f'(program {V} (set-idx v 2.0 99.0)'
               ' (print (sqrt (dot v v))))')
_s = predict_program(_stmts)
_da = [d for k, d in _s[0].get('predicted', []) if k == 'analytic']
import math as _math
_want = _math.sqrt(9827.0) - _math.sqrt(30.0)
expect(_s[0]['verdict'] == 'DESTRUCTIVE' and len(_da) == 1
       and abs(_da[0] - _want) < 1e-9,
       f'sqrt(dot(v,v)): analytic Delta = {_da} (want {_want:.6f})')
check_case('exp(idx) analytic consumer',
           f'(program {V} (set-idx v 2.0 99.0)'
           ' (print (exp (idx v 2.0))))',
           'DESTRUCTIVE')
# overwrite before any use: two sites; the first is transparent
# because the second write restores the natural value before the
# only consumer runs.
_stmts = parse(f'(program {V} {W1} (set-idx v 2.0 99.0)'
               ' (set-idx v 2.0 2.0) (print (dot v w)))')
_sites = predict_program(_stmts)
expect(len(_sites) == 2 and _sites[0]['verdict'] == 'TRANSPARENT'
       and run(_stmts) == run(without_write(_stmts)),
       f'overwrite before any use: first site '
       f'{_sites[0]["verdict"] if _sites else None} (want TRANSPARENT)')

# ------------------------------------------------- randomised battery
rng = random.Random(20261026)

def to_lisp(x):
    if isinstance(x, list):
        return '(' + ' '.join(to_lisp(y) for y in x) + ')'
    if isinstance(x, float):
        return repr(x)
    return str(x)

lies = 0
counts = {'TRANSPARENT': 0, 'DESTRUCTIVE': 0, 'UNKNOWN': 0}
delta_checked = 0
for trial in range(150):
    n = rng.choice([3, 5, 8])
    base = rng.choice([['iota', float(n)],
                       ['cumsum', ['iota', float(n)]],
                       ['+', ['iota', float(n)], ['ones', float(n)]]])
    i = rng.randrange(n)
    nat = {'iota': i, 'cumsum': i * (i + 1) // 2}.get(
        base[0] if isinstance(base, list) else '', i)
    if isinstance(base, list) and base[0] == '+':
        nat = i + 1
    elif isinstance(base, list) and base[0] == 'cumsum':
        nat = i * (i + 1) // 2
    else:
        nat = i
    written = rng.choice([float(nat), float(nat + 5), float(nat - 3),
                          ['idx', 'v', float(rng.randrange(n))]])
    w = [rng.choice([0.0, 1.0, 2.0]) for _ in range(n)]
    consumer = rng.choice([
        ['print', ['dot', 'v', 'w']],
        ['print', ['idx', 'v', float(rng.randrange(n))]],
        ['print', ['len', 'v']],
        ['print', ['dot', ['cumsum', 'v'], 'w']],
        ['print', 'v'],
        ['print', 1.0],
        ['block', ['let', 's', ['dot', 'v', 'w']], ['print', 's']],
        ['print', ['norm', 'v']],
        ['print', ['*', ['dot', 'v', 'w'], ['dot', 'v', 'w']]],
        ['print', ['sqrt', ['dot', 'v', 'v']]],
    ])
    ast = ['program', ['let', 'v', base], ['let', 'w', ['vec'] + w],
           ['set-idx', 'v', float(i), written], consumer]
    src = to_lisp(ast)
    stmts = parse(src)
    sites = predict_program(stmts)
    if len(sites) != 1:
        lies += 1
        print(f'  trial {trial}: expected 1 site, got {len(sites)}')
        continue
    verdict = sites[0]['verdict']
    counts[verdict] += 1
    same = run(stmts) == run(without_write(stmts))
    if verdict == 'TRANSPARENT' and not same:
        lies += 1
        print(f'  trial {trial}: TRANSPARENT lie: {src}')
    if verdict == 'DESTRUCTIVE' and same:
        lies += 1
        print(f'  trial {trial}: DESTRUCTIVE lie: {src}')
    if verdict == 'DESTRUCTIVE' and consumer[0] == 'print' \
            and isinstance(consumer[1], list) \
            and consumer[1][0] in ('dot', 'idx'):
        preds = [d for k, d in sites[0].get('predicted', [])
                 if d is not None]
        if len(preds) == 1:
            got = float(run(stmts)) - float(run(without_write(stmts)))
            if abs(got - float(preds[0])) > 1e-9:
                lies += 1
                print(f'  trial {trial}: Delta lie: predicted '
                      f'{preds[0]}, measured {got}: {src}')
            else:
                delta_checked += 1
expect(lies == 0, f'randomised battery: no prediction lies over '
                  f'150 programs (verdicts: {counts}, '
                  f'Deltas verified: {delta_checked})')

print()
if FAILED:
    print(f'{len(FAILED)} FAILURES')
    sys.exit(1)
print('all tests passed')
