#!/usr/bin/env python3
"""
linkcheck -- the independent checker for boundary-reduction certificates.

Design (the QED split): linkir's seam_cancel *produces* a certificate;
this module *checks* it, trusting as little as possible. It imports
nothing from the compiler. Given, per expression, the original node
list, the claimed reduced node list, and the list of cancellation
steps, it replays the steps on its own copy:

  - every step's precondition is verified in exact rational arithmetic
    (fractions.Fraction). Coefficients are read as the decimal values
    they were written as -- Fraction(repr(x)) for a float x -- which is
    the compiler's documented semantics: cancellation is exact in real
    arithmetic over the written constants.
  - the mode permissions are enforced: rules that need real division
    (div-mul, div-mul-const) are rejected for integer certificates,
    and divmod-join is accepted only for integer certificates.
  - after the replay, its own garbage collection must reproduce the
    claimed reduced nodes and result exactly.

The checker never searches for seams, never composes intervals, and
never evaluates the program: it only verifies claimed steps.

Usage:
    python3 linkcheck.py certificate.json     # exit 0 iff valid
"""
import json
import sys
from fractions import Fraction

DEAD = {'kind': 'const', 'x': None, 's': None, 'p': 0.0, 'q': None,
        'name': None}


def frac(v):
    """A coefficient as an exact rational: the decimal it was written as."""
    if isinstance(v, Fraction):
        return v
    if isinstance(v, str):
        return Fraction(v)
    if isinstance(v, int):
        return Fraction(v)
    return Fraction(repr(v))


def norm_s(s):
    if isinstance(s, list):
        return tuple(s)
    return s


def same_expr(nodes, a, b, memo=None):
    """Structural equality of two subexpressions (verification only:
    the certificate claims two dividends are the same expression)."""
    if a == b:
        return True
    if memo is None:
        memo = {}
    if (a, b) in memo:
        return memo[(a, b)]
    memo[(a, b)] = False
    na, nb = nodes[a], nodes[b]
    if na['kind'] in ('read', 'readeof'):
        return False
    ok = (na['kind'] == nb['kind'] and na['name'] == nb['name']
          and frac(na['p'] if na['p'] is not None else 0)
              == frac(nb['p'] if nb['p'] is not None else 0)
          and (na['q'] is None) == (nb['q'] is None)
          and (na['q'] is None or frac(na['q']) == frac(nb['q'])))
    if ok:
        xa, xb = na['x'], nb['x']
        ok = (xa is None and xb is None) or \
             (xa is not None and xb is not None
              and same_expr(nodes, xa, xb, memo))
    if ok:
        sa, sb = norm_s(na['s']), norm_s(nb['s'])
        if isinstance(sa, tuple) and isinstance(sb, tuple):
            ok = len(sa) == len(sb) and all(
                same_expr(nodes, x, y, memo) for x, y in zip(sa, sb))
        elif sa is None or sb is None:
            ok = sa is None and sb is None
        else:
            ok = same_expr(nodes, sa, sb, memo)
    memo[(a, b)] = ok
    return ok


def const_val(nodes, nid):
    if isinstance(nid, int) and nodes[nid]['kind'] == 'const':
        return frac(nodes[nid]['p'])
    return None


def gc(nodes, result):
    keep, stack = set(), [result]
    while stack:
        i = stack.pop()
        if i in keep:
            continue
        keep.add(i)
        nd = nodes[i]
        if nd['x'] is not None:
            stack.append(nd['x'])
        s = norm_s(nd['s'])
        if isinstance(s, tuple):
            stack.extend(s)
        elif s is not None:
            stack.append(s)
    order = sorted(keep)
    remap = {old: new for new, old in enumerate(order)}
    out = []
    for old in order:
        nd = dict(nodes[old])
        if nd['x'] is not None:
            nd['x'] = remap[nd['x']]
        s = norm_s(nd['s'])
        if isinstance(s, tuple):
            nd['s'] = tuple(remap[t] for t in s)
        elif s is not None:
            nd['s'] = remap[s]
        out.append(nd)
    return out, remap[result]


class Reject(Exception):
    pass


def check_step(nodes, result_ref, step, integer):
    """Verify one claimed cancellation and apply its rewiring to the
    checker's own copy. result_ref is a one-element list holding the
    current result id."""
    rule = step['rule']
    u = step['upper']
    if not (0 <= u < len(nodes)):
        raise Reject(f'step {rule}: upper id {u} out of range')
    nd = nodes[u]

    def rewire(old, new):
        for n in nodes:
            if n['x'] == old:
                n['x'] = new
            s = norm_s(n['s'])
            if isinstance(s, tuple):
                n['s'] = tuple(new if t == old else t for t in s)
            elif s == old:
                n['s'] = new
        if result_ref[0] == old:
            result_ref[0] = new

    if rule == 'divmod-join':
        if not integer:
            raise Reject('divmod-join in a non-integer certificate')
        if nd['kind'] != 'add':
            raise Reject('divmod-join: upper is not an add node')
        C = frac(step['C'])
        if C == 0:
            raise Reject('divmod-join: C is zero')
        mul_id, mod_id, div_id = step['mul'], step['mod'], step['div']
        if {mul_id, mod_id} != {nd['x'], norm_s(nd['s'])}:
            raise Reject('divmod-join: add operands are not the claimed mul/mod')
        m, r, d = nodes[mul_id], nodes[mod_id], nodes[div_id]
        if r['kind'] != 'mod' or const_val(nodes, r['s']) != C:
            raise Reject('divmod-join: mod node does not divide by C')
        if d['kind'] != 'div' or const_val(nodes, d['s']) != C:
            raise Reject('divmod-join: div node does not divide by C')
        if m['x'] != div_id:
            raise Reject('divmod-join: mul does not consume the div')
        if m['kind'] == 'mul':
            if const_val(nodes, m['s']) != C:
                raise Reject('divmod-join: mul does not multiply by C')
        elif m['kind'] == 'aff':
            if frac(m['p']) != C or frac(m['q']) != 0:
                raise Reject('divmod-join: aff is not multiplication by C')
        else:
            raise Reject('divmod-join: quotient path is neither mul nor aff')
        if not same_expr(nodes, d['x'], r['x']):
            raise Reject('divmod-join: div and mod dividends differ')
        if step['repl'] != r['x']:
            raise Reject('divmod-join: repl is not the shared dividend input')
        rewire(u, step['repl'])
        nodes[u] = dict(DEAD)
        return

    lower = step['lower']
    if nd['x'] != lower:
        raise Reject(f'step {rule}: node {u} does not consume node {lower}')
    l = nodes[lower]
    repl = step['repl']

    if rule == 'identity-aff':
        if not (nd['kind'] == 'aff' and frac(nd['p']) == 1
                and frac(nd['q']) == 0):
            raise Reject('identity-aff: upper is not the identity link')
        if repl != lower:
            raise Reject('identity-aff: repl is not the input')
    elif rule == 'aff-aff':
        if not (nd['kind'] == 'aff' and l['kind'] == 'aff'):
            raise Reject('aff-aff: seam is not two affine links')
        p1, q1 = frac(l['p']), frac(l['q'])
        p2, q2 = frac(nd['p']), frac(nd['q'])
        if (p1, q1, p2, q2) != (frac(step['p1']), frac(step['q1']),
                                frac(step['p2']), frac(step['q2'])):
            raise Reject('aff-aff: witnesses do not match the nodes')
        if p2 * p1 != 1 or p2 * q1 + q2 != 0:
            raise Reject('aff-aff: not an exact inverse pair in Q')
        if repl != l['x']:
            raise Reject('aff-aff: repl is not the lower input')
    elif rule == 'mul-div-const':
        C = frac(step['C'])
        if not (nd['kind'] == 'div' and l['kind'] == 'aff'
                and frac(l['q']) == 0 and frac(l['p']) == C):
            raise Reject('mul-div-const: lower is not multiplication by C')
        if const_val(nodes, nd['s']) != C:
            raise Reject('mul-div-const: divisor is not the same constant C')
        if repl != l['x']:
            raise Reject('mul-div-const: repl is not the lower input')
    elif rule == 'div-mul-const':
        if integer:
            raise Reject('div-mul-const is forbidden in integer mode')
        C = frac(step['C'])
        if not (nd['kind'] == 'aff' and frac(nd['q']) == 0
                and frac(nd['p']) == C and l['kind'] == 'div'):
            raise Reject('div-mul-const: seam shape mismatch')
        if const_val(nodes, l['s']) != C:
            raise Reject('div-mul-const: divisor is not the same constant C')
        if repl != l['x']:
            raise Reject('div-mul-const: repl is not the lower input')
    elif rule in ('add-sub', 'sub-add', 'mul-div', 'div-mul'):
        lk, uk = rule.split('-')
        if rule == 'div-mul' and integer:
            raise Reject('div-mul is forbidden in integer mode')
        if not (l['kind'] == lk and nd['kind'] == uk):
            raise Reject(f'{rule}: seam kinds are {l["kind"]}/{nd["kind"]}')
        if l['s'] is None or norm_s(l['s']) != norm_s(nd['s']) \
                or norm_s(nd['s']) != norm_s(step['s']):
            raise Reject(f'{rule}: crosstalk nodes differ')
        if repl != l['x']:
            raise Reject(f'{rule}: repl is not the lower input')
    elif rule == 'exp-log':
        if not (l['kind'] == 'fn' and nd['kind'] == 'fn'
                and {l['name'], nd['name']} == {'exp', 'log'}
                and step['lower_name'] == l['name']
                and step['upper_name'] == nd['name']):
            raise Reject('exp-log: not an exp/log seam')
        if repl != l['x']:
            raise Reject('exp-log: repl is not the lower input')
    else:
        raise Reject(f'unknown rule {rule!r}')

    rewire(u, repl)
    nodes[u] = dict(DEAD)


def nodes_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x['kind'] != y['kind'] or x['name'] != y['name'] \
                or x['x'] != y['x'] or norm_s(x['s']) != norm_s(y['s']):
            return False
        for key in ('p', 'q'):
            xv, yv = x[key], y[key]
            if (xv is None) != (yv is None):
                return False
            if xv is not None and frac(xv) != frac(yv):
                return False
    return True


def check_expression(entry):
    nodes = [dict(nd) for nd in entry['nodes']]
    result_ref = [entry['result']]
    integer = bool(entry.get('integer'))
    for step in entry['steps']:
        check_step(nodes, result_ref, step, integer)
    got_nodes, got_result = gc(nodes, result_ref[0])
    if got_result != entry['reduced_result']:
        raise Reject('final result id mismatch')
    if not nodes_equal(got_nodes, entry['reduced_nodes']):
        raise Reject('reduced node list mismatch')


def check_cert_data(data):
    if data.get('version') != 1:
        return False, f"unsupported certificate version {data.get('version')!r}"
    for i, entry in enumerate(data['expressions']):
        try:
            check_expression(entry)
        except Reject as e:
            return False, f'expression {i}: {e}'
    return True, 'ok'


def main():
    if len(sys.argv) != 2:
        print('usage: linkcheck.py certificate.json')
        return 2
    with open(sys.argv[1]) as f:
        data = json.load(f)
    ok, reason = check_cert_data(data)
    n = sum(len(e['steps']) for e in data['expressions'])
    print(f'{"VALID" if ok else "INVALID"}: {reason} '
          f'({len(data["expressions"])} expressions, {n} steps)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
