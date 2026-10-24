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

def seam_cancel(nodes, result):
    """Boundary reduction: cancel exactly-inverse pairs at their seam.

    No interval is ever composed. Two adjacent links are erased only when
    the second undoes the first (checked at their shared boundary), and the
    chain is stitched input-to-output. Rules:
      aff(p2,q2) after aff(p1,q1) with p2*p1 == 1 and p2*q1 + q2 == 0 (exactly)
      add/sub (or sub/add) fed by the *same* crosstalk node
      mul/div (or div/mul) fed by the *same* crosstalk node
      exp/log in either order
      identity aff (p=1, q=0) is a seam with nothing: it vanishes
    Cancellation is exact in real arithmetic; in floating point it restores
    the exact-arithmetic value where stepwise evaluation would lose it.
    """
    nodes = [dict(nd) for nd in nodes]
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
        for u, nd in enumerate(nodes):
            if nd['x'] is None:
                continue
            l = nodes[nd['x']]
            repl = None
            if nd['kind'] == 'aff' and nd['p'] == 1.0 and nd['q'] == 0.0:
                repl = nd['x']
            elif nd['kind'] == 'aff' and l['kind'] == 'aff':
                if nd['p'] * l['p'] == 1.0 and nd['p'] * l['q'] + nd['q'] == 0.0:
                    repl = l['x']
            elif nd['kind'] == 'div' and l['kind'] == 'aff' and l['q'] == 0.0:
                # (x * C) / C with the *same constant value* C: the seam joins.
                sn = nodes[nd['s']] if isinstance(nd['s'], int) else None
                if sn is not None and sn['kind'] == 'const' and sn['p'] == l['p']:
                    repl = l['x']
            elif nd['kind'] == 'aff' and nd['q'] == 0.0 and l['kind'] == 'div':
                # (x / C) * C with the same constant value C.
                sn = nodes[l['s']] if isinstance(l['s'], int) else None
                if sn is not None and sn['kind'] == 'const' and sn['p'] == nd['p']:
                    repl = l['x']
            elif (l['kind'], nd['kind']) in (('add', 'sub'), ('sub', 'add'),
                                             ('mul', 'div'), ('div', 'mul')):
                if l['s'] is not None and l['s'] == nd['s']:
                    repl = l['x']
            elif l['kind'] == 'fn' and nd['kind'] == 'fn':
                if {l['name'], nd['name']} == {'exp', 'log'}:
                    repl = l['x']
            if repl is not None:
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
    def __init__(self, mode='interval'):
        self.nodes = []          # dicts: kind, x, s, p, q, name
        self.const_val = {}      # node id -> float, when known constant
        self.norm = {}           # node id -> (base_id, p, q): value = p*base + q
        self.mode = mode
        self.reduce = (mode != 'none')

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
        if self.reduce and self.is_const(x):
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
        if self.reduce and self.is_const(x):
            try:
                return self.const(FN1[name](self.const_val[x]))
            except (ValueError, OverflowError):
                pass            # do not fold: let the runtime raise, as written
        return self.add('fn', x=x, name=name)

    def mk_bin(self, kind, x, s):
        """Binary link with through x and crosstalk s, after normalisation."""
        if self.reduce and self.is_const(x) and self.is_const(s):
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
                if c != 0.0 and math.frexp(c)[0] == 0.5:
                    return self.mk_aff(x, 1.0 / c, 0.0)
        return self.add(kind, x=x, s=s)

    def lower(self, e):
        if isinstance(e, (int, float)):
            return self.const(e)
        if isinstance(e, Str):
            raise TypeError('string inside expression')
        if isinstance(e, str):
            return self.add('var', name=e)
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

    def finish(self, result):
        """Boundary pass if requested, then drop dead links and renumber."""
        if self.mode == 'boundary':
            return seam_cancel(self.nodes, result)
        return gc_nodes(self.nodes, result)

# ------------------------------------------------------------- statements

class Lower:
    def __init__(self, mode='interval'):
        self.mode = mode

    def expr(self, e):
        b = Builder(mode=self.mode)
        result = b.lower(e)
        return b.finish(result)

    def stmt(self, node):
        op = node[0] if isinstance(node, list) and node else None
        if op in ('let', 'set'):
            return (op, node[1], self.expr(node[2]))
        if op == 'print':
            parts = []
            for a in node[1:]:
                parts.append(('str', str(a)) if isinstance(a, Str) else ('expr', self.expr(a)))
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
            n += sum(len(p[1][0]) for p in st[1] if p[0] == 'expr')
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
                line.append(payload if tag == 'str' else self.fmt(self.eval(payload)))
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
                else:
                    lines += dump_expr(payload, indent + '  ')
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
    a = ap.parse_args()

    with open(a.source) as f:
        stmts = parse(f.read())
    reduced = Lower(mode=a.mode).program(stmts)
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
