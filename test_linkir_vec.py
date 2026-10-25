#!/usr/bin/env python3
"""
Tests for linkir's vector/matrix/logical instruction support
(v0.10): every operator linkc accepts now lowers in linkir too, by
unrolling into 1x1 Links. demo.link's output must match the
linkc-compiled binary byte for byte.

Run: python3 test_linkir_vec.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linkc import parse
from linkir import Lower, Interp
from linkcheck import check_cert_data

FAILED = []

def expect(cond, label):
    print(('PASS' if cond else 'FAIL'), label)
    if not cond:
        FAILED.append(label)

def run_src(src, mode='interval'):
    prog = Lower(mode=mode).program(parse(src))
    return Interp([]).run(prog)

GOLDEN_DEMO = """\
a = [1.0, 2.0, 3.0, 4.0]
b = [4.0, 3.0, 2.0, 1.0]
dot(a,b) = 20
a+b = [5.0, 5.0, 5.0, 5.0]
2*a = [2.0, 4.0, 6.0, 8.0]
norm(a) = 5.477225575051661
outer(a,b) = [[4.0, 3.0, 2.0, 1.0], [8.0, 6.0, 4.0, 2.0], [12.0, 9.0, 6.0, 3.0], [16.0, 12.0, 8.0, 4.0]]
gram(a,b) = [[30.0, 20.0], [20.0, 30.0]]
matvec(outer(a,b),b) = [30.0, 60.0, 90.0, 120.0]
cumsum(a) = [1.0, 3.0, 6.0, 10.0]
diff(cumsum(a)) = [1.0, 2.0, 3.0, 4.0]
roundtrip error = 0
v after set-idx = [10.0, 1.0, 2.0, 3.0, 4.0]
len(v) = 5  v[0] = 10
"""

with open('examples/demo.link') as f:
    demo_src = f.read()
out = run_src(demo_src)
expect(out == GOLDEN_DEMO, 'demo.link output matches linkc binary exactly')

CASES = {
    'zeros/ones': ('(program (print (zeros 3.0)) (print (ones 2.0)))',
                   '[0.0, 0.0, 0.0]\n[1.0, 1.0]\n'),
    'scalar-vec sub/div (rsub/rdiv)':
        ('(program (let v (vec 1.0 2.0 4.0)) (print (- 10.0 v)) (print (/ 8.0 v)))',
         '[9.0, 8.0, 6.0]\n[8.0, 4.0, 2.0]\n'),
    'vec fn elementwise':
        ('(program (print (sqrt (vec 1.0 4.0 9.0))))',
         '[1.0, 2.0, 3.0]\n'),
    'matmul/transpose':
        ('(program (let a (outer (vec 1.0 2.0) (vec 3.0 4.0)))'
         ' (print (transpose a)) (print (matmul a a)))',
         '[[3.0, 6.0], [4.0, 8.0]]\n[[33.0, 44.0], [66.0, 88.0]]\n'),
    'and/or/not over comparisons':
        ('(program (print (and (< 1.0 2.0) (< 3.0 4.0)))'
         ' (print (or (< 2.0 1.0) (< 3.0 4.0)))'
         ' (print (not (< 1.0 2.0))))',
         '1\n1\n0\n'),
    'dynamic idx':
        ('(program (let v (vec 10.0 20.0 30.0)) (let i 1.0)'
         ' (print (idx v i)) (set i 2.0) (print (idx v i)))',
         '20\n30\n'),
    'if with vector branches':
        ('(program (print (if (< 1.0 2.0) (vec 1.0 2.0) (vec 9.0 9.0))))',
         '[1.0, 2.0]\n'),
    'neg vector':
        ('(program (print (neg (vec 1.0 -2.0))))',
         '[-1.0, 2.0]\n'),
    'set-idx dynamic target, static index':
        ('(program (let v (iota 3.0)) (set-idx v 2.0 99.0) (print v))',
         '[0.0, 1.0, 99.0]\n'),
}
for label, (src, want) in CASES.items():
    got = run_src(src)
    expect(got == want, f'{label}: {got.strip()!r}')

# boundary mode + certificates over unrolled vector code
sink = []
Lower(mode='boundary', cert_sink=sink).program(parse(demo_src))
data = {'version': 1, 'expressions': json.loads(json.dumps(sink))}
ok, reason = check_cert_data(data)
expect(ok, f'demo.link boundary certificate valid ({reason})')

# since v0.11, iota & friends are symbolic: no unroll cap (see
# test_linkir_sym.py). A huge len is a compile-time constant.
out = run_src('(program (print (len (iota 1000000.0))))')
expect(out == '1000000\n', f'iota 1e6 len works symbolically: {out.strip()!r}')

# dynamic set-idx index is a clear error
try:
    run_src('(program (let v (iota 3.0)) (let i 1.0) (set-idx v i 5.0))')
    expect(False, 'dynamic set-idx index raises')
except TypeError as ex:
    expect('compile-time constant' in str(ex),
           f'dynamic set-idx index raises ({ex})')

print()
if FAILED:
    print(f'{len(FAILED)} FAILURES')
    sys.exit(1)
print('all tests passed')
