#!/usr/bin/env python3
"""
linkir v0.1 -- the 1x1 Link intermediate representation.

Every expression is decomposed into 1x1 Links. A 1x1 Link node is:

    y = F( x ; s )

  x : the through signal (exactly one)
  s : a crosstalk signal (at most one) -- another chain's output, fed sideways
  F : one elementary operation

A whole expression becomes chains of through signals, cross-fed by other
chains. Coefficients are not part of the node shape: a constant coefficient
is folded into an AFF node, a *variable* coefficient arrives as crosstalk.

Node kinds:
  const c            y = c
  var name           y = env[name]
  aff  y = p*x + q   affine link, constant p, q  (absorbs +c, *c, -c, /c, neg)
  add/sub/rsub       y = x+s / x-s / s-x
  mul/div/rdiv/mod   y = x*s / x/s / s/x / x mod s
  fn name            y = sqrt/exp/log/abs/sin/cos (x)   -- an atom: breaks affinity
  cmp op             y = 1.0 if x <op> s else 0.0       -- an atom
  read               y = next number from the input token stream
  readeof            y = 1.0 once the input stream is exhausted
  if                 y = x != 0 ? then : else

Reduction (the compiler optimisation of this IR) is interval reduction on
affine chains: consecutive AFF links compose in closed form,

    (p2,q2) o (p1,q1) = (p2*p1, p2*q1 + q2),

so any maximal affine segment collapses to a single 1x1 Link. Atoms and
dynamic (crosstalk) coefficients stop the composition -- exactly where the
affine case stops closing. Identity links (p=1, q=0) vanish, constants fold.

Usage:
    python3 linkir.py examples/affine.link [--run] [--tokens "3 4"]
"""
import argparse
import math
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from linkc import parse, Str     # shared AST reader (S-expressions) and string marker
from boundary import extract as boundary_extract, reconstruct as boundary_reconstruct, sizes as boundary_sizes

FN1 = {'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
       'abs': abs, 'sin': math.sin, 'cos': math.cos}

# Vectors and matrices are decomposed into 1x1 Links by unrolling, so a
# compile-time-known size is required. The cap keeps the decomposition
# a decomposition (bulk pipelines of 1e6 elements belong to linkc's
# native Vec code, not to an unrolled IR).
MAX_UNROLL = 100_000

def gc_nodes(nodes, result):
    """Drop unreachable nodes, renumber. Returns (nodes, result)."""
    keep, stack = set(), [result]
    while stack:
        i = stack.pop()
        if i in keep:
            continue
        keep.add(i)
        nd = nodes[i]
        if nd['x'] is not None:
            stack.append(nd['x'])
        s = nd['s']
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
        s = nd['s']
        if isinstance(s, tuple):
            nd['s'] = tuple(remap[t] for t in s)
        elif s is not None:
            nd['s'] = remap[s]
        out.append(nd)
    return out, remap[result]

def seam_cancel(nodes, result, integer=False, cert=None):
    """Boundary reduction: cancel exactly-inverse pairs at their seam.

    No interval is ever composed. Two adjacent links are erased only when
    the second undoes the first (checked at their shared boundary), and the
    chain is stitched input-to-output. Rules:
      aff(p2,q2) after aff(p1,q1) with p2*p1 == 1 and p2*q1 + q2 == 0 (exactly)
      add/sub (or sub/add) fed by the *same* crosstalk node
      mul/div (or div/mul) fed by the *same* crosstalk node
      (x * C) / C or (x / C) * C for the same constant value C
      exp/log in either order
      identity aff (p=1, q=0) is a seam with nothing: it vanishes
    Cancellation is exact in real arithmetic; in floating point it restores
    the exact-arithmetic value where stepwise evaluation would lose it.

    integer=True (for linkbin's integer dialect) disables every rule whose
    exactness relies on real division: (x / C) * C is NOT x in integer
    division, so div-then-multiply pairs never cancel there.

    cert: if a list is passed, every cancellation is recorded into it as
    a step dict (rule name, the node ids involved, the replacement, and
    the exact coefficient witnesses as decimal strings). The independent
    checker in linkcheck.py replays these steps in rational arithmetic;
    this function only *produces* the certificate, never vouches for it.
    """
    nodes = [dict(nd) for nd in nodes]

    def const_val_of(nid):
        if isinstance(nid, int) and nodes[nid]['kind'] == 'const':
            return nodes[nid]['p']
        return None

    def same_expr(a, b, memo=None):
        """Structural equality of two subexpressions. Needed because the
        lowering does not share nodes: the two occurrences of the same
        subexpression (e.g. the dividend in a div and a mod) are distinct
        node ids with identical structure."""
        if a == b:
            return True
        if memo is None:
            memo = {}
        if (a, b) in memo:
            return memo[(a, b)]
        memo[(a, b)] = False          # guard; the graph is a DAG, no cycles
        na, nb = nodes[a], nodes[b]
        if na['kind'] in ('read', 'readeof'):
            return False              # distinct reads are never the same value
        ok = (na['kind'] == nb['kind'] and na['name'] == nb['name']
              and na['p'] == nb['p'] and na['q'] == nb['q'])
        if ok:
            xa, xb, sa, sb = na['x'], nb['x'], na['s'], nb['s']
            ok = (xa is None and xb is None) or \
                 (xa is not None and xb is not None and same_expr(xa, xb, memo))
        if ok:
            if isinstance(sa, tuple) and isinstance(sb, tuple):
                ok = len(sa) == len(sb) and all(same_expr(x, y, memo) for x, y in zip(sa, sb))
            elif sa is None or sb is None:
                ok = sa is None and sb is None
            else:
                ok = same_expr(sa, sb, memo)
        memo[(a, b)] = ok
        return ok

    def divmod_join(u):
        """The remainder path is a Link too (integer mode only).

        x = C*(x/C) + (x mod C) is an identity in integer arithmetic
        (both truncating and floored conventions): the quotient flows
        through, the remainder flows sideways as a mod Link, and where
        the two paths join, the pair collapses back to x itself.
        Returns the input node id when node u is such a join.
        """
        nd = nodes[u]
        if nd['kind'] != 'add':
            return None
        for mul_id, mod_id in ((nd['x'], nd['s']), (nd['s'], nd['x'])):
            if not (isinstance(mul_id, int) and isinstance(mod_id, int)):
                continue
            m, r = nodes[mul_id], nodes[mod_id]
            if r['kind'] != 'mod':
                continue
            C = const_val_of(r['s'])
            if C is None or C == 0:
                continue
            d = None
            if m['kind'] == 'mul' and const_val_of(m['s']) == C:
                d = m['x']
            elif m['kind'] == 'aff' and m['q'] == 0.0 and m['p'] == C:
                d = m['x']
            if not isinstance(d, int):
                continue
            dv = nodes[d]
            if dv['kind'] == 'div' and same_expr(dv['x'], r['x']) and const_val_of(dv['s']) == C:
                return r['x'], {'mul': mul_id, 'mod': mod_id, 'div': d, 'C': C}
        return None

    def rewire(old, new):
        nonlocal result
        for nd in nodes:
            if nd['x'] == old:
                nd['x'] = new
            s = nd['s']
            if isinstance(s, tuple):
                nd['s'] = tuple(new if t == old else t for t in s)
            elif s == old:
                nd['s'] = new
        if result == old:
            result = new
    changed = True
    while changed:
        changed = False
        if integer:
            # divmod joins first: they consume the div/mod pair as a whole.
            for u in range(len(nodes)):
                found = divmod_join(u)
                if found is not None:
                    v, info = found
                    if cert is not None:
                        cert.append({'rule': 'divmod-join', 'upper': u, 'repl': v,
                                     'mul': info['mul'], 'mod': info['mod'],
                                     'div': info['div'], 'C': repr(info['C'])})
                    rewire(u, v)
                    nodes[u] = {'kind': 'const', 'x': None, 's': None,
                                'p': 0.0, 'q': None, 'name': None}
                    changed = True
                    break
            if changed:
                continue
        for u, nd in enumerate(nodes):
            if nd['x'] is None:
                continue
            l = nodes[nd['x']]
            repl = None
            rule = None
            if nd['kind'] == 'aff' and nd['p'] == 1.0 and nd['q'] == 0.0:
                repl = nd['x']
                rule = 'identity-aff'
            elif nd['kind'] == 'aff' and l['kind'] == 'aff':
                if nd['p'] * l['p'] == 1.0 and nd['p'] * l['q'] + nd['q'] == 0.0:
                    repl = l['x']
                    rule = 'aff-aff'
            elif nd['kind'] == 'div' and l['kind'] == 'aff' and l['q'] == 0.0:
                # (x * C) / C with the *same constant value* C: the seam joins.
                sn = nodes[nd['s']] if isinstance(nd['s'], int) else None
                if sn is not None and sn['kind'] == 'const' and sn['p'] == l['p']:
                    repl = l['x']
                    rule = 'mul-div-const'
            elif nd['kind'] == 'aff' and nd['q'] == 0.0 and l['kind'] == 'div' and not integer:
                # (x / C) * C with the same constant value C.
                # Real arithmetic only: forbidden in integer mode, where
                # (x / C) * C != x in general.
                sn = nodes[l['s']] if isinstance(l['s'], int) else None
                if sn is not None and sn['kind'] == 'const' and sn['p'] == nd['p']:
                    repl = l['x']
                    rule = 'div-mul-const'
            elif ((l['kind'], nd['kind']) in (('add', 'sub'), ('sub', 'add'),
                                               ('mul', 'div'))
                  or ((l['kind'], nd['kind']) == ('div', 'mul') and not integer)):
                # ('div','mul') is real-arithmetic only, like the constant
                # case: integer division is not invertible.
                if l['s'] is not None and l['s'] == nd['s']:
                    repl = l['x']
                    rule = f"{l['kind']}-{nd['kind']}"
            elif l['kind'] == 'fn' and nd['kind'] == 'fn':
                if {l['name'], nd['name']} == {'exp', 'log'}:
                    repl = l['x']
                    rule = 'exp-log'
            if repl is not None:
                if cert is not None:
                    step = {'rule': rule, 'upper': u, 'lower': nd['x'], 'repl': repl}
                    if rule == 'identity-aff':
                        step.update({'p': repr(nd['p']), 'q': repr(nd['q'])})
                    elif rule == 'aff-aff':
                        step.update({'p1': repr(l['p']), 'q1': repr(l['q']),
                                     'p2': repr(nd['p']), 'q2': repr(nd['q'])})
                    elif rule in ('mul-div-const', 'div-mul-const'):
                        step['C'] = repr(l['p'] if rule == 'mul-div-const' else nd['p'])
                    elif rule in ('add-sub', 'sub-add', 'mul-div', 'div-mul'):
                        step['s'] = nd['s']
                    elif rule == 'exp-log':
                        step.update({'lower_name': l['name'], 'upper_name': nd['name']})
                    cert.append(step)
                rewire(u, repl)
                nodes[u] = {'kind': 'const', 'x': None, 's': None, 'p': 0.0, 'q': None, 'name': None}
                changed = True
    return gc_nodes(nodes, result)

class Builder:
    """Lower one expression to a list of 1x1 Link nodes.

    mode: 'none'     -- pure decomposition, nothing folded or composed
          'boundary' -- constants folded/normalised, NO composition;
                        inverse pairs are cancelled afterwards at their seams
          'interval' -- additionally, affine chains are composed (interval reduction)
    """
    def __init__(self, mode='interval', integer=False, cert_sink=None,
                 shapes=None):
        self.nodes = []          # dicts: kind, x, s, p, q, name
        self.const_val = {}      # node id -> float, when known constant
        self.norm = {}           # node id -> (base_id, p, q): value = p*base + q
        self.mode = mode
        self.cert_sink = cert_sink
        # shapes: name -> ('vec', n) | ('mat', (r, c)) for variables
        # bound to vectors/matrices; shared across a whole program so
        # later statements can materialise element variables (name__i).
        self.shapes = shapes if shapes is not None else {}
        self.reduce = (mode != 'none')
        # integer=True: target is linkbin's integer dialect. Constant
        # folding is disabled (floats cannot fold i64 arithmetic exactly),
        # divisions stay true division nodes, and seam_cancel is told to
        # skip rules that need real division. Literal values themselves
        # stay exact: callers guarantee |literal| < 2**53.
        self.integer = integer

    def add(self, kind, x=None, s=None, p=None, q=None, name=None):
        nid = len(self.nodes)
        self.nodes.append({'kind': kind, 'x': x, 's': s, 'p': p, 'q': q, 'name': name})
        self.norm[nid] = (nid, 1.0, 0.0)
        return nid

    def const(self, v):
        nid = self.add('const', p=float(v))
        self.const_val[nid] = float(v)
        return nid

    def is_const(self, nid):
        return nid in self.const_val

    def mk_aff(self, x, p, q):
        """y = p*x + q, with affine-chain composition when reducing."""
        if self.reduce and not self.integer and self.is_const(x):
            return self.const(p * self.const_val[x] + q)
        if self.mode != 'interval':
            return self.add('aff', x=x, p=p, q=q)
        base, p0, q0 = self.norm[x]
        np_, nq = p * p0, p * q0 + q
        if np_ == 1.0 and nq == 0.0:
            return base                       # identity link vanishes
        nid = self.add('aff', x=base, p=np_, q=nq)
        self.norm[nid] = (base, np_, nq)
        return nid

    def mk_fn(self, name, x):
        if self.reduce and not self.integer and self.is_const(x):
            try:
                return self.const(FN1[name](self.const_val[x]))
            except (ValueError, OverflowError):
                pass            # do not fold: let the runtime raise, as written
        return self.add('fn', x=x, name=name)

    def mk_bin(self, kind, x, s):
        """Binary link with through x and crosstalk s, after normalisation."""
        if self.reduce and not self.integer and self.is_const(x) and self.is_const(s):
            a, b = self.const_val[x], self.const_val[s]
            try:
                v = {'add': a + b, 'sub': a - b, 'rsub': b - a, 'mul': a * b,
                     'div': a / b, 'rdiv': b / a,
                     'mod': a - math.floor(a / b) * b if b else 0.0}[kind]
            except (ZeroDivisionError, ValueError, OverflowError):
                pass            # do not fold: let the runtime raise, as written
            else:
                return self.const(v)
        if self.reduce:
            if kind == 'add' and self.is_const(s): return self.mk_aff(x, 1.0, self.const_val[s])
            if kind == 'add' and self.is_const(x): return self.mk_aff(s, 1.0, self.const_val[x])
            if kind == 'sub' and self.is_const(s): return self.mk_aff(x, 1.0, -self.const_val[s])
            if kind == 'sub' and self.is_const(x): return self.mk_aff(s, -1.0, self.const_val[x])
            if kind == 'mul' and self.is_const(s): return self.mk_aff(x, self.const_val[s], 0.0)
            if kind == 'mul' and self.is_const(x): return self.mk_aff(s, self.const_val[x], 0.0)
            if kind == 'div' and self.is_const(s):
                c = self.const_val[s]
                # x / c as x * (1/c) is exact only when c is a power of two.
                # Otherwise keep a true division node: the reciprocal's
                # rounding would change the value even when no seam cancels.
                # seam_cancel can still cancel this div against a matching
                # multiplication by the same constant.
                if c != 0.0 and not self.integer and math.frexp(c)[0] == 0.5:
                    return self.mk_aff(x, 1.0 / c, 0.0)
        return self.add(kind, x=x, s=s)

    def lower(self, e):
        if isinstance(e, (int, float)):
            return self.const(e)
        if isinstance(e, Str):
            raise TypeError('string inside expression')
        if isinstance(e, str):
            if e in self.shapes:
                raise TypeError(f'linkir: {e} is a {self.shapes[e][0]}, '
                                'not a scalar')
            return self.add('var', name=e)
        if isinstance(e, list) and e and e[0] in (
                'dot', 'norm', 'len', 'idx', 'and', 'or', 'not'):
            v = self.lower_any(e)
            if v[0] != 'scalar':
                raise TypeError(f'linkir: {e[0]} did not produce a scalar')
            return v[1]
        op, args = e[0], e[1:]
        if op == 'read-int':
            return self.add('read')
        if op == 'read-eof?':
            return self.add('readeof')
        if op in ('+', '-', '*', '/', 'mod'):
            kind = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', 'mod': 'mod'}[op]
            if op == '-' and len(args) == 1:
                return self.mk_aff(self.lower(args[0]), -1.0, 0.0)
            acc = self.lower(args[0])
            for a in args[1:]:
                acc = self.mk_bin(kind, acc, self.lower(a))
            return acc
        if op == 'neg':
            return self.mk_aff(self.lower(args[0]), -1.0, 0.0)
        if op in FN1:
            return self.mk_fn(op, self.lower(args[0]))
        if op in ('<', '<=', '>', '>=', '==', '!='):
            x, s = self.lower(args[0]), self.lower(args[1])
            if self.reduce and self.is_const(x) and self.is_const(s):
                a, b = self.const_val[x], self.const_val[s]
                v = {'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b,
                     '==': a == b, '!=': a != b}[op]
                return self.const(1.0 if v else 0.0)
            return self.add('cmp', x=x, s=s, name=op)
        if op == 'if':
            c = self.lower(args[0]); t = self.lower(args[1]); f = self.lower(args[2])
            return self.add('if', x=c, s=(t, f))
        raise TypeError(f'linkir: unknown operator {op}')

    # -------------------------------------------- vectors and matrices
    # Values: ('scalar', nid) | ('vec', [nid, ...]) |
    #         ('mat', [[nid, ...], ...]). Everything is unrolled into
    # 1x1 Links: a vector operation is exactly its scalar decomposition.

    def _check_size(self, n):
        if n > MAX_UNROLL:
            raise TypeError(
                f'linkir: {n} elements exceed the 1x1 unroll limit '
                f'({MAX_UNROLL}); use linkc for bulk vectors')

    def _static_size(self, e):
        nid = self.lower(e)
        if not self.is_const(nid):
            raise TypeError('linkir: vector size must be a compile-time '
                            'constant (unrolled decomposition)')
        n = int(self.const_val[nid])
        if n < 0:
            raise TypeError('linkir: negative vector size')
        self._check_size(n)
        return n

    def _dot(self, a, b):
        acc = None
        for x, y in zip(a, b):
            p = self.mk_bin('mul', x, y)
            acc = p if acc is None else self.mk_bin('add', acc, p)
        return acc if acc is not None else self.const(0.0)

    def lower_any(self, e):
        if isinstance(e, str) and e in self.shapes:
            kind, shape = self.shapes[e]
            if kind == 'vec':
                return ('vec', [self.add('var', name=f'{e}__{i}')
                                for i in range(shape)])
            r, c = shape
            return ('mat', [[self.add('var', name=f'{e}__{i}__{j}')
                             for j in range(c)] for i in range(r)])
        if not isinstance(e, list) or not e:
            return ('scalar', self.lower(e))
        op, args = e[0], e[1:]
        if op == 'vec':
            elems = []
            for a in args:
                v = self.lower_any(a)
                if v[0] != 'scalar':
                    raise TypeError('linkir: vec elements must be scalars')
                elems.append(v[1])
            self._check_size(len(elems))
            return ('vec', elems)
        if op in ('zeros', 'ones', 'iota'):
            n = self._static_size(args[0])
            return ('vec', [self.const(float(i) if op == 'iota'
                                        else (0.0 if op == 'zeros' else 1.0))
                            for i in range(n)])
        if op == 'len':
            v = self.lower_any(args[0])
            if v[0] == 'scalar':
                raise TypeError('linkir: len of a scalar')
            return ('scalar', self.const(float(len(v[1]))))
        if op == 'idx':
            v = self.lower_any(args[0])
            if v[0] != 'vec':
                raise TypeError('linkir: idx expects a vector')
            elems = v[1]
            iv = self.lower(args[1])
            if self.is_const(iv):
                i = int(self.const_val[iv])
                if not (0 <= i < len(elems)):
                    raise TypeError(f'linkir: idx {i} out of range')
                return ('scalar', elems[i])
            acc = elems[-1]
            for i in range(len(elems) - 2, -1, -1):
                c = self.add('cmp', x=iv, s=self.const(float(i)), name='==')
                acc = self.add('if', x=c, s=(elems[i], acc))
            return ('scalar', acc)
        if op == 'dot':
            a, b = self.lower_any(args[0]), self.lower_any(args[1])
            if a[0] != 'vec' or b[0] != 'vec' or len(a[1]) != len(b[1]):
                raise TypeError('linkir: dot expects two vectors of '
                                'equal length')
            return ('scalar', self._dot(a[1], b[1]))
        if op == 'norm':
            a = self.lower_any(args[0])
            if a[0] != 'vec':
                raise TypeError('linkir: norm expects a vector')
            return ('scalar', self.mk_fn('sqrt', self._dot(a[1], a[1])))
        if op == 'outer':
            a, b = self.lower_any(args[0]), self.lower_any(args[1])
            if a[0] != 'vec' or b[0] != 'vec':
                raise TypeError('linkir: outer expects two vectors')
            self._check_size(len(a[1]) * len(b[1]))
            return ('mat', [[self.mk_bin('mul', x, y) for y in b[1]]
                            for x in a[1]])
        if op == 'gram':
            vs = [self.lower_any(a) for a in args]
            if any(v[0] != 'vec' for v in vs):
                raise TypeError('linkir: gram expects vectors')
            return ('mat', [[self._dot(u[1], w[1]) for w in vs]
                            for u in vs])
        if op == 'matvec':
            m, v = self.lower_any(args[0]), self.lower_any(args[1])
            if m[0] != 'mat' or v[0] != 'vec' \
                    or (m[1] and len(m[1][0]) != len(v[1])):
                raise TypeError('linkir: matvec shape mismatch')
            return ('vec', [self._dot(row, v[1]) for row in m[1]])
        if op == 'matmul':
            a, b = self.lower_any(args[0]), self.lower_any(args[1])
            if a[0] != 'mat' or b[0] != 'mat' \
                    or (a[1] and len(a[1][0]) != len(b[1])):
                raise TypeError('linkir: matmul shape mismatch')
            cols = [list(col) for col in zip(*b[1])]
            return ('mat', [[self._dot(row, col) for col in cols]
                            for row in a[1]])
        if op == 'transpose':
            m = self.lower_any(args[0])
            if m[0] != 'mat':
                raise TypeError('linkir: transpose expects a matrix')
            return ('mat', [list(row) for row in zip(*m[1])])
        if op == 'cumsum':
            v = self.lower_any(args[0])
            if v[0] != 'vec':
                raise TypeError('linkir: cumsum expects a vector')
            out, acc = [], None
            for el in v[1]:
                acc = el if acc is None else self.mk_bin('add', acc, el)
                out.append(acc)
            return ('vec', out)
        if op == 'diff':
            v = self.lower_any(args[0])
            if v[0] != 'vec':
                raise TypeError('linkir: diff expects a vector')
            if not v[1]:
                return ('vec', [])
            out = [v[1][0]]
            for i in range(1, len(v[1])):
                out.append(self.mk_bin('sub', v[1][i], v[1][i - 1]))
            return ('vec', out)
        if op in ('and', 'or', 'not'):
            if op == 'not':
                return ('scalar', self.mk_aff(self.lower(args[0]),
                                              -1.0, 1.0))
            a, b = self.lower(args[0]), self.lower(args[1])
            if op == 'and':
                return ('scalar', self.mk_bin('mul', a, b))
            t = self.mk_bin('add', a, b)
            return ('scalar', self.add('cmp', x=t, s=self.const(0.0),
                                       name='>'))
        if op in ('+', '-', '*', '/') or op == 'neg' or op in FN1 \
                or op == 'if':
            return self._ew(op, args, e)
        return ('scalar', self.lower(e))

    def _kind_of(self, e):
        """Infer scalar/vec/mat WITHOUT lowering (linkc's infer, so
        the scalar path is chosen before any node is built)."""
        if isinstance(e, (int, float)):
            return 'scalar'
        if isinstance(e, str):
            s = self.shapes.get(e)
            return s[0] if s else 'scalar'
        if not isinstance(e, list) or not e:
            return 'scalar'
        op = e[0]
        if op in ('vec', 'zeros', 'ones', 'iota', 'cumsum', 'diff',
                  'matvec'):
            return 'vec'
        if op in ('outer', 'gram', 'matmul', 'transpose'):
            return 'mat'
        if op in ('+', '-', '*', '/'):
            ks = [self._kind_of(a) for a in e[1:]]
            if 'mat' in ks:
                return 'mat'
            if 'vec' in ks:
                return 'vec'
            return 'scalar'
        if op == 'neg' or op in FN1:
            return self._kind_of(e[1])
        if op == 'if':
            return self._kind_of(e[2])
        return 'scalar'

    def _ew(self, op, args, whole):
        """Elementwise-capable operators: scalar operands keep the
        classic scalar path (identical nodes to before); any vector or
        matrix operand switches the whole operation to its unrolled
        elementwise decomposition."""
        if self._kind_of(whole) == 'scalar':
            return ('scalar', self.lower(whole))
        if op == 'if':
            c = self.lower(args[0])
            t, f = self.lower_any(args[1]), self.lower_any(args[2])
            if t[0] == 'scalar' and f[0] == 'scalar':
                return ('scalar', self.lower(whole))
            if t[0] != f[0]:
                raise TypeError('linkir: if branches have different kinds')
            if t[0] == 'vec':
                if len(t[1]) != len(f[1]):
                    raise TypeError('linkir: if branch length mismatch')
                return ('vec', [self.add('if', x=c, s=(x, y))
                                for x, y in zip(t[1], f[1])])
            return ('mat', [[self.add('if', x=c, s=(x, y))
                             for x, y in zip(ra, rb)]
                            for ra, rb in zip(t[1], f[1])])
        vals = [self.lower_any(a) for a in args]
        if all(v[0] == 'scalar' for v in vals):
            return ('scalar', self.lower(whole))
        if op == 'neg' or (op == '-' and len(args) == 1):
            v = vals[0]
            if v[0] == 'vec':
                return ('vec', [self.mk_aff(el, -1.0, 0.0) for el in v[1]])
            return ('mat', [[self.mk_aff(el, -1.0, 0.0) for el in row]
                            for row in v[1]])
        if op in FN1:
            v = vals[0]
            if v[0] == 'vec':
                return ('vec', [self.mk_fn(op, el) for el in v[1]])
            if v[0] == 'mat':
                return ('mat', [[self.mk_fn(op, el) for el in row]
                                for row in v[1]])
            return ('scalar', self.lower(whole))
        kind = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
        acc = vals[0]
        for v in vals[1:]:
            acc = self._ew_bin(kind, acc, v)
        return acc

    def _ew_bin(self, kind, a, b):
        ka, kb = a[0], b[0]
        if ka == 'scalar' and kb == 'scalar':
            return ('scalar', self.mk_bin(kind, a[1], b[1]))
        if ka == 'vec' and kb == 'vec':
            if len(a[1]) != len(b[1]):
                raise TypeError('linkir: vector length mismatch')
            return ('vec', [self.mk_bin(kind, x, y)
                            for x, y in zip(a[1], b[1])])
        if ka == 'vec' and kb == 'scalar':
            return ('vec', [self.mk_bin(kind, x, b[1]) for x in a[1]])
        if ka == 'scalar' and kb == 'vec':
            rk = {'add': 'add', 'mul': 'mul',
                  'sub': 'rsub', 'div': 'rdiv'}[kind]
            return ('vec', [self.mk_bin(rk, y, a[1]) for y in b[1]])
        if ka == 'mat' and kb == 'mat' and kind == 'add':
            return ('mat', [[self.mk_bin('add', x, y)
                             for x, y in zip(ra, rb)]
                            for ra, rb in zip(a[1], b[1])])
        if ka == 'mat' and kb == 'scalar' and kind == 'mul':
            return ('mat', [[self.mk_bin('mul', x, b[1]) for x in row]
                            for row in a[1]])
        if ka == 'scalar' and kb == 'mat' and kind == 'mul':
            return ('mat', [[self.mk_bin('mul', x, a[1]) for x in row]
                            for row in b[1]])
        raise TypeError(f'linkir: unsupported operand kinds for {kind}: '
                        f'{ka} and {kb}')

    def finish(self, result):
        """Boundary pass if requested, then drop dead links and renumber."""
        if self.mode == 'boundary':
            if self.cert_sink is None:
                return seam_cancel(self.nodes, result, integer=self.integer)
            pre = [dict(nd) for nd in self.nodes]
            steps = []
            reduced = seam_cancel(self.nodes, result, integer=self.integer,
                                  cert=steps)
            self.cert_sink.append({'nodes': pre, 'result': result,
                                   'steps': steps,
                                   'reduced_nodes': reduced[0],
                                   'reduced_result': reduced[1],
                                   'integer': self.integer})
            return reduced
        return gc_nodes(self.nodes, result)

# ------------------------------------------------------------- statements

class Lower:
    def __init__(self, mode='interval', integer=False, cert_sink=None):
        self.mode = mode
        self.integer = integer
        self.cert_sink = cert_sink
        self.shapes = {}         # name -> ('vec', n) | ('mat', (r, c))

    def _builder(self):
        return Builder(mode=self.mode, integer=self.integer,
                       cert_sink=self.cert_sink, shapes=self.shapes)

    def expr(self, e):
        b = self._builder()
        result = b.lower(e)
        return b.finish(result)

    def expr_any(self, e):
        b = self._builder()
        return b, b.lower_any(e)

    def stmt(self, node):
        op = node[0] if isinstance(node, list) and node else None
        if op in ('let', 'set'):
            b, v = self.expr_any(node[2])
            name = node[1]
            if v[0] == 'scalar':
                self.shapes.pop(name, None)
                return (op, name, b.finish(v[1]))
            if v[0] == 'vec':
                self.shapes[name] = ('vec', len(v[1]))
                return ('block', [(op, f'{name}__{i}', b.finish(el))
                                  for i, el in enumerate(v[1])])
            rows = len(v[1])
            cols = len(v[1][0]) if rows else 0
            self.shapes[name] = ('mat', (rows, cols))
            return ('block', [(op, f'{name}__{i}__{j}', b.finish(el))
                              for i, row in enumerate(v[1])
                              for j, el in enumerate(row)])
        if op == 'set-idx':
            name = node[1]
            if self.shapes.get(name, (None,))[0] != 'vec':
                raise TypeError(f'linkir: set-idx on non-vector {name}')
            ib = self._builder()
            iv = ib.lower(node[2])
            if not ib.is_const(iv):
                raise TypeError('linkir: set-idx index must be a '
                                'compile-time constant')
            i = int(ib.const_val[iv])
            if not (0 <= i < self.shapes[name][1]):
                raise TypeError('linkir: set-idx index out of range')
            return ('set', f'{name}__{i}', self.expr(node[3]))
        if op == 'print':
            parts = []
            for a in node[1:]:
                if isinstance(a, Str):
                    parts.append(('str', str(a)))
                    continue
                b, v = self.expr_any(a)
                if v[0] == 'scalar':
                    parts.append(('expr', b.finish(v[1])))
                elif v[0] == 'vec':
                    parts.append(('vec', [b.finish(el) for el in v[1]]))
                else:
                    parts.append(('mat', [[b.finish(el) for el in row]
                                          for row in v[1]]))
            return ('print', parts)
        if op == 'for':
            if len(node) >= 5 and not isinstance(node[3], list):
                lo, hi, body = self.expr(node[2]), self.expr(node[3]), node[4:]
            else:
                lo, hi, body = None, self.expr(node[2]), node[3:]
            return ('for', node[1], lo, hi, [self.stmt(s) for s in body])
        if op == 'while':
            return ('while', self.expr(node[1]), [self.stmt(s) for s in node[2:]])
        if op == 'if':
            return ('if', self.expr(node[1]), self.stmt(node[2]),
                    self.stmt(node[3]) if len(node) > 3 else None)
        if op == 'block':
            return ('block', [self.stmt(s) for s in node[1:]])
        return ('expr', self.expr(node))

    def program(self, stmts):
        return [self.stmt(s) for s in stmts]

def count_nodes(ir_stmts):
    n = 0
    for st in ir_stmts:
        k = st[0]
        if k in ('let', 'set'):
            n += len(st[2][0])
        elif k == 'print':
            for p in st[1]:
                if p[0] == 'expr':
                    n += len(p[1][0])
                elif p[0] == 'vec':
                    n += sum(len(e[0]) for e in p[1])
                elif p[0] == 'mat':
                    n += sum(len(e[0]) for row in p[1] for e in row)
        elif k == 'for':
            n += (len(st[2][0]) if st[2] else 0) + len(st[3][0]) + count_nodes(st[4])
        elif k == 'while':
            n += len(st[1][0]) + count_nodes(st[2])
        elif k == 'if':
            n += len(st[1][0]) + count_nodes([st[2]]) + (count_nodes([st[3]]) if st[3] else 0)
        elif k == 'block':
            n += count_nodes(st[1])
        elif k == 'expr':
            n += len(st[1][0])
    return n

# ------------------------------------------------------------- interpreter

class Interp:
    def __init__(self, tokens):
        self.env = {}
        self.tokens = tokens
        self.pos = 0
        self.eof = False
        self.out = []
        self.trace_name = None
        self.trace = []

    def eval(self, exprir):
        nodes, result = exprir
        vals = [0.0] * len(nodes)
        for i, nd in enumerate(nodes):
            k = nd['kind']
            if k == 'const': v = nd['p']
            elif k == 'var': v = self.env[nd['name']]
            elif k == 'aff': v = nd['p'] * vals[nd['x']] + nd['q']
            elif k == 'add': v = vals[nd['x']] + vals[nd['s']]
            elif k == 'sub': v = vals[nd['x']] - vals[nd['s']]
            elif k == 'rsub': v = vals[nd['s']] - vals[nd['x']]
            elif k == 'mul': v = vals[nd['x']] * vals[nd['s']]
            elif k == 'div': v = vals[nd['x']] / vals[nd['s']]
            elif k == 'rdiv': v = vals[nd['s']] / vals[nd['x']]
            elif k == 'mod':
                a, b = vals[nd['x']], vals[nd['s']]
                v = a - math.floor(a / b) * b
            elif k == 'fn': v = FN1[nd['name']](vals[nd['x']])
            elif k == 'cmp':
                a, b = vals[nd['x']], vals[nd['s']]
                v = 1.0 if {'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b,
                            '==': a == b, '!=': a != b}[nd['name']] else 0.0
            elif k == 'read':
                if self.pos >= len(self.tokens):
                    self.eof = True; v = 0.0
                else:
                    v = self.tokens[self.pos]; self.pos += 1
            elif k == 'readeof':
                v = 1.0 if self.eof else 0.0
            elif k == 'if':
                t, f = nd['s']
                v = vals[t] if vals[nd['x']] != 0.0 else vals[f]
            else:
                raise TypeError(f'bad node kind {k}')
            vals[i] = v
        return vals[result]

    @staticmethod
    def fmt(v):
        if v != v or v in (float('inf'), float('-inf')):
            return repr(v)
        return str(int(v)) if float(v) == int(v) and abs(v) < 1e16 else repr(v)

    @staticmethod
    def fmt_debug(v):
        """Rust {:?} for f64 (vector/matrix elements): shortest
        round-trip, always with a decimal point."""
        if v != v:
            return 'NaN'
        if v in (float('inf'), float('-inf')):
            return 'inf' if v > 0 else '-inf'
        if v == 0.0:
            return '-0.0' if math.copysign(1.0, v) < 0 else '0.0'
        if float(v) == int(v) and abs(v) < 1e16:
            return f'{int(v)}.0'
        return repr(v)

    def run(self, stmts):
        for st in stmts:
            self.stmt(st)
        return ''.join(self.out)

    def stmt(self, st):
        k = st[0]
        if k in ('let', 'set'):
            v = self.eval(st[2])
            self.env[st[1]] = v
            if st[1] == self.trace_name:
                self.trace.append(v)
        elif k == 'print':
            line = []
            for tag, payload in st[1]:
                if tag == 'str':
                    line.append(payload)
                elif tag == 'expr':
                    line.append(self.fmt(self.eval(payload)))
                elif tag == 'vec':
                    line.append('[' + ', '.join(
                        self.fmt_debug(self.eval(e)) for e in payload) + ']')
                elif tag == 'mat':
                    line.append('[' + ', '.join(
                        '[' + ', '.join(
                            self.fmt_debug(self.eval(e)) for e in row) + ']'
                        for row in payload) + ']')
            self.out.append(' '.join(line) + '\n')
        elif k == 'for':
            _, var, lo, hi, body = st
            start = int(self.eval(lo)) if lo else 0
            end = int(self.eval(hi))
            for i in range(start, end):
                self.env[var] = float(i)
                for s in body:
                    self.stmt(s)
        elif k == 'while':
            while self.eval(st[1]) != 0.0:
                for s in st[2]:
                    self.stmt(s)
        elif k == 'if':
            if self.eval(st[1]) != 0.0:
                self.stmt(st[2])
            elif st[3]:
                self.stmt(st[3])
        elif k == 'block':
            for s in st[1]:
                self.stmt(s)
        elif k == 'expr':
            self.eval(st[1])

# ------------------------------------------------------------- dump

def dump_expr(exprir, indent='  '):
    nodes, result = exprir
    lines = []
    for i, nd in enumerate(nodes):
        k = nd['kind']
        if k == 'const': d = f"const {nd['p']:g}"
        elif k == 'var': d = f"var {nd['name']}"
        elif k == 'aff': d = f"aff  n{nd['x']} * {nd['p']:g} + {nd['q']:g}"
        elif k in ('add', 'sub', 'rsub', 'mul', 'div', 'rdiv', 'mod'):
            d = f"{k:<4} n{nd['x']} ; crosstalk n{nd['s']}"
        elif k == 'fn': d = f"fn   {nd['name']}(n{nd['x']})"
        elif k == 'cmp': d = f"cmp  n{nd['x']} {nd['name']} crosstalk n{nd['s']}"
        elif k == 'read': d = "read (input as data)"
        elif k == 'readeof': d = "readeof"
        elif k == 'if': d = f"if   n{nd['x']} ? n{nd['s'][0]} : n{nd['s'][1]}"
        else: d = k
        lines.append(f'{indent}n{i}: {d}')
    lines.append(f'{indent}=> n{result}')
    return lines

def dump(stmts, indent=''):
    lines = []
    for st in stmts:
        k = st[0]
        if k in ('let', 'set'):
            lines.append(f'{indent}{k} {st[1]} =')
            lines += dump_expr(st[2], indent + '  ')
        elif k == 'print':
            lines.append(f'{indent}print')
            for tag, payload in st[1]:
                if tag == 'str':
                    lines.append(f'{indent}  str {payload!r}')
                elif tag == 'expr':
                    lines += dump_expr(payload, indent + '  ')
                elif tag == 'vec':
                    for e in payload:
                        lines += dump_expr(e, indent + '  ')
                else:
                    for row in payload:
                        for e in row:
                            lines += dump_expr(e, indent + '  ')
        elif k == 'for':
            lines.append(f'{indent}for {st[1]}:')
            lines += dump(st[4], indent + '  ')
        elif k == 'while':
            lines.append(f'{indent}while:')
            lines += dump_expr(st[1], indent + '  ')
            lines += dump(st[2], indent + '  ')
        elif k == 'if':
            lines.append(f'{indent}if:')
            lines += dump_expr(st[1], indent + '  ')
            lines += dump([st[2]], indent + '  ')
            if st[3]:
                lines += dump([st[3]], indent + '  ')
        elif k == 'block':
            lines += dump(st[1], indent)
        elif k == 'expr':
            lines += dump_expr(st[1], indent)
    return lines

# ------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description='linkir: AST -> 1x1 Link IR')
    ap.add_argument('source')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--tokens', default='', help='input numbers for read-int, e.g. "3 4 5"')
    ap.add_argument('--no-dump', action='store_true')
    ap.add_argument('--mode', choices=['interval', 'boundary', 'none'], default='interval',
                    help='interval: compose affine chains; boundary: cancel inverse pairs at seams; none: raw')
    ap.add_argument('--sparse-input', action='store_true',
                    help='route the --tokens stream through its boundary representation first')
    ap.add_argument('--trace', metavar='NAME',
                    help='record every value assigned to NAME, then boundary-analyse the trajectory')
    ap.add_argument('--emit-ast', metavar='PATH',
                    help='write the reduced program back as a .link AST (consumable by linkc)')
    ap.add_argument('--certify', metavar='PATH',
                    help='(boundary mode) write a certificate of every seam '
                         'cancellation to PATH as JSON; see linkcheck.py')
    ap.add_argument('--check', action='store_true',
                    help='with --certify: verify the certificate with the '
                         'independent checker before writing it')
    a = ap.parse_args()

    with open(a.source) as f:
        stmts = parse(f.read())
    cert_sink = [] if a.certify else None
    reduced = Lower(mode=a.mode, cert_sink=cert_sink).program(stmts)
    if a.certify:
        import json
        data = {'version': 1, 'expressions': json.loads(json.dumps(cert_sink))}
        if a.check:
            from linkcheck import check_cert_data
            ok, reason = check_cert_data(data)
            print(f'[linkir] certificate check: {"OK" if ok else "FAILED: " + reason}')
            if not ok:
                sys.exit(1)
        with open(a.certify, 'w') as f:
            json.dump(data, f, indent=1)
        n_steps = sum(len(e['steps']) for e in cert_sink)
        print(f'[linkir] certificate written to {a.certify} '
              f'({len(cert_sink)} expressions, {n_steps} certified steps)')
    if a.emit_ast:
        from linkemit import emit_program, to_source
        with open(a.emit_ast, 'w') as f:
            f.write(to_source(emit_program(reduced, stmts)))
        print(f'[linkir] reduced AST written to {a.emit_ast}')
    raw = Lower(mode='none').program(stmts)
    n_red, n_raw = count_nodes(reduced), count_nodes(raw)
    if not a.no_dump:
        print('\n'.join(dump(reduced)))
    print(f'[linkir] 1x1 Links: {n_raw} (raw decomposition) -> {n_red} (after {a.mode} reduction)')
    if a.run:
        tokens = [float(t) for t in a.tokens.split()] if a.tokens.strip() else []
        if a.sparse_input and tokens:
            s = boundary_sizes(tokens)
            init, ev = boundary_extract(tokens)
            tokens = boundary_reconstruct(len(tokens), init, ev)
            print(f"[linkir] input as boundaries: {s['dense']} samples -> "
                  f"{s['boundary']} numbers (initial + {s['switches']} switch points); decoded and run")
        interp = Interp(tokens)
        if a.trace:
            interp.trace_name = a.trace
        out = interp.run(reduced)
        print('[linkir] run:')
        print(out, end='')
        if a.trace:
            tr = interp.trace
            init, ev = boundary_extract(tr)
            rec = boundary_reconstruct(len(tr), init, ev)
            err = max((abs(p - q) for p, q in zip(rec, tr)), default=0.0)
            print(f'[linkir] trace {a.trace}: {len(tr)} samples, {len(ev)} boundaries '
                  f'(+initial = {1 + len(ev)} numbers), reconstruction max error {err}')

if __name__ == '__main__':
    main()
