#!/usr/bin/env python3
"""
Tests for the certificate checker (linkcheck.py) and the certificate
output of linkir's boundary reduction.

Run: python3 test_linkcheck.py
"""
import copy
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linkc import parse
from linkir import Lower, Interp, seam_cancel
from linkcheck import check_cert_data

FAILED = []

def expect(cond, label):
    print(('PASS' if cond else 'FAIL'), label)
    if not cond:
        FAILED.append(label)

def certify(src, integer=False):
    sink = []
    prog = Lower(mode='boundary', integer=integer,
                 cert_sink=sink).program(parse(src))
    data = {'version': 1, 'expressions': json.loads(json.dumps(sink))}
    return prog, data

def rules_of(data):
    return [s['rule'] for e in data['expressions'] for s in e['steps']]

# ---------------------------------------------------------- examples
for name in ('adversarial', 'affine', 'demo_bytes', 'stats',
             'streamsum', 'trace_demo'):
    with open(f'examples/{name}.link') as f:
        src = f.read()
    sink = []
    Lower(mode='boundary', cert_sink=sink).program(parse(src))
    data = {'version': 1, 'expressions': json.loads(json.dumps(sink))}
    ok, reason = check_cert_data(data)
    expect(ok, f'example {name}: certificate valid ({reason})')

# ------------------------------------------------- planted rule seams
PLANTED = {
    'aff-aff': '(let y (- (/ (* (+ x 3) 2) 2) 3))',
    'exp-log': '(let y (exp (log x)))',
    'mul-div-const': '(let y (/ (* x 7) 7))',
    'div-mul-const': '(let y (* (/ x 7) 7))',
    'identity-aff': '(let y (* x 1))',
}
for rule, src in PLANTED.items():
    _, data = certify(src)
    ok, reason = check_cert_data(data)
    expect(ok and rule in rules_of(data),
           f'planted {rule}: certified and valid ({reason})')

# integer divmod join certifies; (x/C)*C does not cancel in integer mode
_, data = certify('(let y (+ (* (/ x 7) 7) (mod x 7)))', integer=True)
ok, reason = check_cert_data(data)
expect(ok and 'divmod-join' in rules_of(data),
       f'integer divmod-join certified and valid ({reason})')
_, data = certify('(let y (* (/ x 7) 7))', integer=True)
expect(rules_of(data) == [], 'integer (x/C)*C: no cancellation step')

# ------------------------------- crosstalk pair rules (shared-node IR)
def hand_ir(steps_spec):
    """Build a node list by hand: var w (0), var x (1), then the chain."""
    nodes = [{'kind': 'var', 'x': None, 's': None, 'p': None, 'q': None,
              'name': 'w'},
             {'kind': 'var', 'x': None, 's': None, 'p': None, 'q': None,
              'name': 'x'}]
    prev = 1
    for kind in steps_spec:
        nodes.append({'kind': kind, 'x': prev, 's': 0, 'p': None,
                      'q': None, 'name': None})
        prev = len(nodes) - 1
    return nodes, prev

for kinds, rule, integer in ((('add', 'sub'), 'add-sub', False),
                             (('sub', 'add'), 'sub-add', False),
                             (('mul', 'div'), 'mul-div', False),
                             (('mul', 'div'), 'mul-div', True),
                             (('div', 'mul'), 'div-mul', False)):
    nodes, result = hand_ir(kinds)
    steps = []
    reduced = seam_cancel(nodes, result, integer=integer, cert=steps)
    entry = {'nodes': nodes, 'result': result, 'steps': steps,
             'reduced_nodes': reduced[0], 'reduced_result': reduced[1],
             'integer': integer}
    data = {'version': 1, 'expressions': json.loads(json.dumps([entry]))}
    ok, reason = check_cert_data(data)
    expect(ok and [s['rule'] for s in steps] == [rule],
           f'shared-crosstalk {rule} (integer={integer}): valid ({reason})')

# div-mul must NOT cancel in integer mode even with shared crosstalk
nodes, result = hand_ir(('div', 'mul'))
steps = []
seam_cancel(nodes, result, integer=True, cert=steps)
expect(steps == [], 'shared-crosstalk div-mul in integer mode: no step')

# ------------------------------------------------------------- forgery
# Claim a div-mul-const cancellation inside an integer certificate.
_, data = certify('(let y (* (/ x 7) 7))', integer=False)
forged = copy.deepcopy(data)
forged['expressions'][0]['integer'] = True
ok, _ = check_cert_data(forged)
expect(not ok, 'forgery: float div-mul-const cert relabelled integer')

# ---------------------------------------------------------------- fuzz
rng = random.Random(20261026)
PAIRS = [(2.0, 0.5), (4.0, 0.25), (10.0, 0.1), (5.0, 0.2), (0.5, 2.0)]

def rand_expr_ast(depth):
    e = 'x'
    val_range = (0.5, 3.0)      # keep exp/log safe
    for _ in range(depth):
        choice = rng.randrange(5)
        if choice == 0:
            p, ip = rng.choice(PAIRS)
            q = rng.choice([0.0, 1.5, -2.25, 3.0])
            inner = ['*', ['+', e, q / p], p] if q else ['*', e, p]
            e = ['*', inner, ip] if not q else ['+', ['*', inner, ip], -q]
            # (p*(x + q/p) ) * (1/p) - q == x exactly in reals; the two
            # outer seams are affine inverses of the inner ones.
        elif choice == 1:
            e = ['log', ['exp', e]]
        elif choice == 2:
            c = rng.choice([2.0, 7.0, 10.0])
            e = ['/', ['*', e, c], c]
        elif choice == 3:
            c = rng.choice([2.0, 7.0, 10.0])
            e = ['*', ['/', e, c], c]
        else:
            e = ['*', e, 1.0]
    return e

ok_all = True
eq_all = True
for trial in range(60):
    ast = ['let', 'y', rand_expr_ast(rng.randint(1, 4))]
    sink = []
    prog_b = Lower(mode='boundary', cert_sink=sink).program([ast])
    data = {'version': 1, 'expressions': json.loads(json.dumps(sink))}
    ok, reason = check_cert_data(data)
    ok_all &= ok
    prog_n = Lower(mode='none').program([ast])
    for xv in (0.7, 1.3, 2.9):
        ib, in_ = Interp([]), Interp([])
        ib.env['x'] = xv
        in_.env['x'] = xv
        vb, vn = ib.run(prog_b), in_.run(prog_n)
        # programs only bind y; compare through the env instead
        eq_all &= abs(ib.env['y'] - in_.env['y']) < 1e-9
expect(ok_all, 'fuzz: 60 planted programs, all certificates valid')
expect(eq_all, 'fuzz: boundary reduction preserves values')

# ----------------------------------------------------------- mutations
_, data = certify(PLANTED['aff-aff'])
base_ok, _ = check_cert_data(data)
expect(base_ok, 'mutation base certificate is valid')

def mutated(fn):
    d = copy.deepcopy(data)
    fn(d['expressions'][0])
    return d

cases = {
    'coefficient witness tampered':
        lambda e: e['steps'][0].update({'p2': '0.25'}),
    'rule name swapped':
        lambda e: e['steps'][0].update({'rule': 'exp-log'}),
    'repl redirected':
        lambda e: e['steps'][0].update({'repl': 0}),
    'step dropped':
        lambda e: e['steps'].pop(0),
    'reduced nodes tampered':
        lambda e: e['reduced_nodes'][0].update({'p': 999.0}),
    'original nodes tampered':
        lambda e: next(n for n in e['nodes']
                       if n['kind'] == 'aff').update({'q': 5.0}),
}
for label, fn in cases.items():
    ok, _ = check_cert_data(mutated(fn))
    expect(not ok, f'mutation rejected: {label}')

print()
if FAILED:
    print(f'{len(FAILED)} FAILURES')
    sys.exit(1)
print('all tests passed')
