#!/usr/bin/env python3
"""
Tests for linkstream: lazy incremental materialisation (monadic
streams), the difference engine, online Gram folds, and the
operator-level crosstalk Gram of rank-1 symbolic matrices.

Run: python3 test_linkstream.py
"""
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linkc import parse
from linkir import Lower, Interp, _peval, _selem, _sdot, _scumsum
from linkstream import Stream, poly_stream, symvec_stream, \
    gram_online, gram, symmat_gram

FAILED = []

def expect(cond, label):
    print(('PASS' if cond else 'FAIL'), label)
    if not cond:
        FAILED.append(label)

rng = random.Random(20261026)

# ------------------------------------------- difference engine exact
for trial in range(30):
    d = rng.randint(0, 5)
    poly = [Fraction(rng.randint(-6, 6), rng.choice([1, 2, 3]))
            for _ in range(d + 1)]
    ov = {i: Fraction(rng.randint(-9, 9))
          for i in rng.sample(range(50), rng.randint(0, 3))}
    shift = Fraction(rng.randint(-3, 3), 2)
    got = Stream.to_list(poly_stream(poly, ov, shift, n=300))
    want = [_selem((poly, ov, shift), i) for i in range(300)]
    expect(got == want, f'difference engine trial {trial} exact (deg {d})')

# ------------------------------------------------------- monad laws
def rand_stream():
    return Stream.from_iterable(
        [Fraction(rng.randint(-5, 5)) for _ in range(rng.randint(0, 6))])

def rand_f():
    a, b = rng.randint(-3, 3), rng.randint(0, 2)
    return lambda x: Stream.from_iterable([x + a] * (b + 1))

ok = True
for _ in range(200):
    m = rand_stream()
    xs = m.to_list()
    f, g = rand_f(), rand_f()
    a = Fraction(rng.randint(-5, 5))
    ok &= Stream.pure(a).bind(f).to_list() == f(a).to_list()          # left id
    ok &= m.bind(Stream.pure).to_list() == xs                          # right id
    ok &= m.bind(f).bind(g).to_list() == \
        m.bind(lambda x: f(x).bind(g)).to_list()                       # assoc
expect(ok, 'monad laws: left/right identity, associativity (200 cases)')

# scan = cumulative accumulation (cumsum / cumulative product)
s = Stream.from_iterable([Fraction(i) for i in range(10)])
expect(s.scan(lambda acc, x: acc + x, Fraction(0)).to_list()
       == [Fraction(i * (i + 1) // 2) for i in range(10)],
       'scan(+) reproduces cumsum')
expect(s.take(4).scan(lambda acc, x: acc * (x + 1), Fraction(1)).to_list()
       == [Fraction(1), Fraction(2), Fraction(6), Fraction(24)],
       'scan(*) reproduces cumulative product (factorials)')

# ------------------------------------- online Gram == Faulhaber form
# symvecs straight from the compiler: v = iota, w = cumsum(iota)
sink_prog = Lower(mode='interval').program(parse(
    '(program (let v (iota 1000000.0)) (let w (cumsum v)))'))
# rebuild the same symbolic values directly for streaming
from linkir import Builder
b = Builder()
v = b.lower_any(['iota', 1000000.0])
w = b.lower_any(['cumsum', ['iota', 1000000.0]])
closed = _sdot(v[2], w[2], 1000000)
online = gram(symvec_stream(1000000, v[2]), symvec_stream(1000000, w[2]))
expect(online == closed,
       f'online Gram at n=1e6 equals the closed form exactly ({online})')
partials = gram_online(symvec_stream(1000, v[2]),
                       symvec_stream(1000, w[2])).to_list()
brute = sum((_selem(v[2], i) * _selem(w[2], i) for i in range(1000)),
            Fraction(0))
expect(partials[-1] == brute and len(partials) == 1000,
       'online Gram partial sums are exact at every prefix')

# ------------------------------- operator Gram: interaction of Links
b2 = Builder()
m1 = b2.lower_any(['outer', ['iota', 7.0], ['cumsum', ['iota', 7.0]]])
m2 = b2.lower_any(['outer', ['ones', 7.0], ['iota', 7.0]])
g = symmat_gram(m1, m2)
# brute force over materialised entries
A = [[_selem(m1[1][2], i) * _selem(m1[2][2], j)
      for j in range(7)] for i in range(7)]
B = [[_selem(m2[1][2], i) * _selem(m2[2][2], j)
      for j in range(7)] for i in range(7)]
want = sum((A[i][j] * B[i][j] for i in range(7) for j in range(7)),
           Fraction(0))
expect(g == want, f'symmat crosstalk Gram tr(A^T B) from factors ({g})')

# streamed printing beyond the unroll cap
prog = Lower(mode='interval').program(parse(
    '(program (print (iota 200000.0)))'))
out = Interp([]).run(prog)
toks = out.strip()[1:-1].split(', ')
expect(len(toks) == 200000 and toks[0] == '0.0' and toks[-1] == '199999.0',
       'symvec print streams past the cap (200000 elements)')

print()
if FAILED:
    print(f'{len(FAILED)} FAILURES')
    sys.exit(1)
print('all tests passed')
