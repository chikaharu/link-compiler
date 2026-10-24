#!/usr/bin/env python3
"""
linkemit -- 1x1 Link IR -> AST emitter.

linkir's reductions (interval composition, boundary seam cancellation)
used to dead-end in the IR interpreter: neither linkc (Rust backend) nor
linkbin consumed them. This module emits a reduced IR program back as an
S-expression AST, so the reduction flows into the native binary:

    .link --linkir (boundary)--> reduced IR --linkemit--> .link --linkc--> binary

Safety rules (emission must not change what the program computes):
  * Expressions containing (read-int) / (read-eof?) are emitted in their
    original, unreduced form: IR evaluation is eager (an `if` node's two
    branches both evaluate), AST evaluation is not, and a duplicated or
    dropped read would change the input stream.
  * Everything else in the IR is pure, so shared nodes may be duplicated
    as trees; an expansion cap falls back to the original expression if
    the tree would balloon.
  * Non-finite constants are never emitted (fallback instead).

Known scope: scalar expressions only -- exactly linkir's lowering scope.
Programs using Vec/Mat operations do not lower, and callers fall back to
compiling the original program.
"""
import math

from linkc import Str


class EmitError(Exception):
    pass


def emit_expr(exprir, original):
    """Emit one IR expression as an AST, or return `original` unchanged."""
    nodes, result = exprir
    if any(nd['kind'] in ('read', 'readeof') for nd in nodes):
        return original
    size = [0]

    def ev(nid):
        size[0] += 1
        if size[0] > 8 * len(nodes) + 16:
            raise EmitError('tree expansion cap exceeded')
        nd = nodes[nid]
        k = nd['kind']
        if k == 'const':
            v = nd['p']
            if not math.isfinite(v):
                raise EmitError('non-finite constant')
            return v
        if k == 'var':
            return nd['name']
        x = ev(nd['x']) if nd['x'] is not None else None
        if k == 'aff':
            p, q = nd['p'], nd['q']
            if not (math.isfinite(p) and math.isfinite(q)):
                raise EmitError('non-finite affine coefficient')
            if q == 0.0:
                return x if p == 1.0 else ['*', p, x]
            if p == 1.0:
                return ['+', x, q]
            return ['+', ['*', p, x], q]
        if k in ('add', 'sub', 'mul', 'div', 'mod', 'rsub', 'rdiv'):
            s = ev(nd['s'])
            if k == 'rsub':
                return ['-', s, x]
            if k == 'rdiv':
                return ['/', s, x]
            return [{ 'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'mod': 'mod'}[k], x, s]
        if k == 'cmp':
            return [nd['name'], x, ev(nd['s'])]
        if k == 'fn':
            return [nd['name'], x]
        if k == 'if':
            t, f = nd['s']
            return ['if', x, ev(t), ev(f)]
        raise EmitError(f'cannot emit node kind {k}')

    try:
        return ev(result)
    except EmitError:
        return original


def emit_stmt(lowered, original):
    k = lowered[0]
    if k in ('let', 'set'):
        return [k, lowered[1], emit_expr(lowered[2], original[2])]
    if k == 'print':
        parts = []
        for (tag, payload), orig in zip(lowered[1], original[1:]):
            parts.append(Str(payload) if tag == 'str' else emit_expr(payload, orig))
        return ['print'] + parts
    if k == 'for':
        _, var, lo, hi, body = lowered
        has_lo = len(original) >= 5 and not isinstance(original[3], list)
        orig_body = original[4:] if has_lo else original[3:]
        out = ['for', var]
        if lo is not None:
            out.append(emit_expr(lo, original[2]))
        out.append(emit_expr(hi, original[3] if has_lo else original[2]))
        out += [emit_stmt(b, o) for b, o in zip(body, orig_body)]
        return out
    if k == 'while':
        return (['while', emit_expr(lowered[1], original[1])] +
                [emit_stmt(b, o) for b, o in zip(lowered[2], original[2:])])
    if k == 'if':
        out = ['if', emit_expr(lowered[1], original[1]),
               emit_stmt(lowered[2], original[2])]
        if lowered[3] is not None:
            out.append(emit_stmt(lowered[3], original[3]))
        return out
    if k == 'block':
        return ['block'] + [emit_stmt(b, o) for b, o in zip(lowered[1], original[1:])]
    if k == 'expr':
        return emit_expr(lowered[1], original)
    raise EmitError(f'cannot emit statement kind {k}')


def emit_program(lowered_stmts, original_stmts):
    """Pair each lowered statement with its original and emit the program."""
    if len(lowered_stmts) != len(original_stmts):
        raise EmitError('lowered/original statement count mismatch')
    return [emit_stmt(l, o) for l, o in zip(lowered_stmts, original_stmts)]


# ------------------------------------------------------------ serialization

def fmt(node):
    if isinstance(node, Str):
        return '"' + str(node).replace('\\', '\\\\').replace('"', '\\"') + '"'
    if isinstance(node, float):
        return repr(node)
    if isinstance(node, int):
        return str(node)
    if isinstance(node, list):
        return '(' + ' '.join(fmt(x) for x in node) + ')'
    return str(node)                       # symbol


def to_source(stmts):
    return '(program\n' + ''.join(f'  {fmt(s)}\n' for s in stmts) + ')\n'
