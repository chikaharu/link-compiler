#!/usr/bin/env python3
"""
linkpredict -- static transparency prediction for set-idx.

Experiment #41 showed that connecting an output pole back to an
input pole (what set-idx does) disturbs the natural computation
exactly through a crosstalk Gram kappa. This module turns that into
a compiler pass: for every set-idx on a symbolic vector it computes,
BEFORE running anything,

  delta  -- the residue the write injects (written value minus the
            natural value at that slot), and
  gains  -- the crosstalk gain of every downstream observable with
            respect to that residue (sensitivities are themselves
            symbolic vectors; a consumer's gain is a Gram dot of the
            sensitivity pattern against the consumer),

and predicts one of:

  TRANSPARENT   no observable output can change (delta == 0, a dead
                write, or every consumer's change polynomial is 0)
  DESTRUCTIVE   some observable provably changes; the exact change
                is predicted (see below)
  UNKNOWN       the analysis abstains (dynamic indices, unmodelled
                consumers, unknown written values)

v0.14 (Taylor x Cayley-Hamilton, experiment #42): scalar observables
are no longer linearised. Each observable's change is computed as an
exact delta-polynomial (the terminating Taylor expansion of a
polynomial consumer): products and powers of linear observables --
whose delta^2 cross terms v0.13 had to abstain on -- now resolve
exactly, dot consumers expand to degree 2 (<P1,S2> + <S1,P2> linear
part, <P1,P2> quadratic part), norm consumers are decided exactly
through Delta(||E||^2) (a sign flip of one element, delta = -2*v[i],
provably preserves the norm), and a top-level analytic consumer
f(polynomial) is decided at a known delta by direct evaluation.

The predictor is sound by construction: it never claims TRANSPARENT
or DESTRUCTIVE without a proof inside its model. test_linkpredict.py
checks the predictions against ground truth (running the program
with and without the write) on hand cases and a randomised battery.

Usage: python3 linkpredict.py program.link
"""
import math
import sys
from fractions import Fraction

from linkc import parse, Str
from linkir import (Builder, FN1, _selem, _sadd, _ssub, _sscale,
                    _smul, _scumsum, _sdiff, _sdot, _svals, _sfold,
                    _frac_of_float)

ZERO_SV = ([Fraction(0)], {}, Fraction(0))
DEPENDS = 'depends'
DEG_CAP = 8           # highest delta-polynomial degree kept exact


# delta-polynomials: the change of a scalar observable as an exact
# polynomial in the injected residue delta (the terminating Taylor
# expansion of a polynomial consumer). Represented as coefficient
# lists with a zero constant term; [] is the zero polynomial.
def _dp(coeffs):
    out = list(coeffs)
    while out and out[-1] == 0:
        out.pop()
    return out


def _dp_add(a, b):
    n = max(len(a), len(b))
    return _dp([(a[i] if i < len(a) else Fraction(0))
                + (b[i] if i < len(b) else Fraction(0))
                for i in range(n)])


def _dp_scale(a, s):
    if s == 0:
        return []
    return _dp([c * s for c in a])


def _dp_mul(a, b):
    if not a or not b:
        return []
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return _dp(out)


def _dp_eval(a, x):
    acc = Fraction(0)
    for c in reversed(a):
        acc = acc * x + c
    return acc


def is_zero_sv(sv):
    poly, ov, shift = _sfold(sv)
    return all(c == 0 for c in poly) and all(v == 0 for v in ov.values())


def delta_sv(i, g):
    return ([Fraction(0)], {i: g}, Fraction(0))


class Analyzer:
    def __init__(self):
        self.sym = {}        # name -> (n, symvec)          natural values
        self.syms = {}       # name -> symmat value
        self.shapes = {}     # name -> ('vec', n) | ('mat', (r, c))
        self.scalar_const = {}   # name -> Fraction | None
        self.vec_consts = {}     # name -> [Fraction] (all-const vector)
        self.mat_consts = {}     # name -> [[Fraction]] (all-const matrix)

    def builder(self):
        return Builder(shapes=self.shapes, sym=self.sym,
                       syms=self.syms)

    # ------------------------------------------------ const evaluation
    def const_value(self, e):
        """Fraction value of a scalar expression in the current
        (natural) environment, or None if not compile-time known."""
        b = self.builder()
        try:
            v = b.lower_any(e)
        except (TypeError, KeyError, IndexError):
            return None
        if v[0] != 'scalar':
            return None
        if b.is_const(v[1]):
            return _frac_of_float(b.const_val[v[1]])
        if isinstance(e, str) and e in self.scalar_const:
            return self.scalar_const[e]
        return None

    def factor_sv(self, e):
        """A vector expression as a symbolic vector of constants
        (via linkir factors), or None."""
        if isinstance(e, str) and e in self.vec_consts:
            return _svals(self.vec_consts[e])
        b = self.builder()
        try:
            v = b.lower_any(e)
        except (TypeError, KeyError, IndexError):
            return None
        f = b._factor_of(v)
        if f is None:
            return None
        if f[0] == 'poly':
            return f[2]
        return _svals(f[1])

    def mentions(self, e, name, extra=()):
        if isinstance(e, str):
            return e == name or e in extra
        if isinstance(e, list):
            return any(self.mentions(a, name, extra) for a in e)
        return False


class SiteScan:
    """Consumer scan for one set-idx site: walks the statements after
    the write, tracking how the injected residue propagates."""

    def __init__(self, an, name, n, slot):
        self.an = an
        self.name = name
        self.n = n
        self.slot = slot
        self.aux_pat = {}        # vec var -> sensitivity pattern (sv)
        self.aux_scalar = {}     # scalar var -> gain (Fraction|DEPENDS)
        self.derived_ast = {}    # vec var -> AST it was bound from
        self.effects = []        # ('scalar', gain) | ('vec', pattern) |
                                 # ('depends', why)
        self.interfering = False
        self.stopped = False

    # --------------------------------------------- pattern of vec expr
    def pattern(self, e):
        """Sensitivity of a vector expression's slots to the residue:
        a symvec P with P(j) = d E[j] / d delta, or DEPENDS."""
        an = self.an
        if isinstance(e, str):
            if e == self.name:
                return delta_sv(self.slot, Fraction(1))
            if e in self.derived_ast:
                return self.pattern(self.derived_ast[e])
            return self.aux_pat.get(e, ZERO_SV)
        if not isinstance(e, list) or not e:
            return ZERO_SV
        op, args = e[0], e[1:]
        if op == 'cumsum':
            p = self.pattern(args[0])
            return DEPENDS if p is DEPENDS else _scumsum(p)
        if op == 'diff':
            p = self.pattern(args[0])
            return DEPENDS if p is DEPENDS else _sdiff(p)
        if op == 'neg':
            p = self.pattern(args[0])
            return DEPENDS if p is DEPENDS else _sscale(p, Fraction(-1))
        if op in FN1:
            p = self.pattern(args[0])
            if p is DEPENDS:
                return DEPENDS
            return ZERO_SV if is_zero_sv(p) else DEPENDS
        if op in ('+', '-', '*', '/'):
            return self._pattern_bin(op, args)
        if op == 'matvec':
            return self._pattern_matvec(args)
        if op == 'if':
            if not self.dpoly_is_zero(args[0]):
                return DEPENDS
            p1, p2 = self.pattern(args[1]), self.pattern(args[2])
            if p1 is DEPENDS or p2 is DEPENDS:
                return DEPENDS
            return p1 if p1 == p2 else DEPENDS
        if op == 'vec':
            ov = {}
            for j, a in enumerate(args):
                dp = self.scalar_dpoly(a)
                if dp is DEPENDS or isinstance(dp, tuple) \
                        or len(dp) > 2:
                    return DEPENDS
                if len(dp) > 1 and dp[1] != 0:
                    ov[j] = dp[1]
            return ([Fraction(0)], ov, Fraction(0))
        if an.mentions(e, self.name, tuple(self.derived_ast)):
            return DEPENDS
        return ZERO_SV

    def _pattern_bin(self, op, args):
        pats = [self.pattern(a) for a in args]
        if any(p is DEPENDS for p in pats):
            return DEPENDS
        if op in ('+', '-'):
            acc = pats[0]
            for p in pats[1:]:
                acc = _sadd(acc, p) if op == '+' else _ssub(acc, p)
            return acc
        if op == '*':
            if len(args) != 2:
                return DEPENDS
            # product rule: d(E1 E2) = pat1 * val2 + pat2 * val1.
            # If both sides move, a delta^2 cross term appears and
            # the change is no longer linear in delta: abstain.
            p1, p2 = pats
            if not is_zero_sv(p1) and not is_zero_sv(p2):
                return DEPENDS
            acc = ZERO_SV
            if not is_zero_sv(p1):
                v2 = self.an.factor_sv(args[1])
                if v2 is None:
                    return DEPENDS
                acc = _sadd(acc, _smul(p1, v2))
            if not is_zero_sv(p2):
                v1 = self.an.factor_sv(args[0])
                if v1 is None:
                    return DEPENDS
                acc = _sadd(acc, _smul(p2, v1))
            return acc
        # '/': only a constant scalar divisor keeps patterns exact
        if len(args) == 2 and is_zero_sv(pats[1]):
            v2 = self.an.factor_sv(args[1])
            if v2 is not None:
                from linkir import _sfold as _sf
                poly, ov, shift = _sf(v2)
                if all(c == 0 for c in poly[1:]) and not ov:
                    c = (poly[0] if poly else Fraction(0)) + shift
                    if c != 0:
                        return _sscale(pats[0], 1 / c)
        return DEPENDS

    def _pattern_matvec(self, args):
        p = self.pattern(args[1])
        if p is DEPENDS or is_zero_sv(p):
            return p
        # matrix must be constant: get its entries via lowering
        if isinstance(args[0], str) and args[0] in self.an.mat_consts:
            rows = self.an.mat_consts[args[0]]
        else:
            b = self.an.builder()
            try:
                m = b.lower_any(args[0])
            except (TypeError, KeyError, IndexError):
                return DEPENDS
            if m[0] == 'symmat':
                m = b._materialize_symmat(m)
            if m[0] != 'mat':
                return DEPENDS
            rows = []
            for row in m[1]:
                if not all(b.is_const(el) for el in row):
                    return DEPENDS
                rows.append([_frac_of_float(b.const_val[el])
                             for el in row])
        if self.n > 20000:
            return DEPENDS
        pat_vals = [_selem(p, t) for t in range(self.n)]
        out = []
        for row in rows:
            out.append(sum((row[t] * pat_vals[t]
                            for t in range(self.n)), Fraction(0)))
        return _svals(out)

    # ------------------------------ scalar observables as delta-polys
    # A scalar observable's change is an exact polynomial in the
    # residue delta (the terminating Taylor expansion for polynomial
    # consumers). NORM is a marker triple ('#norm', base2, dpoly2):
    # the change of ||E||^2 is a delta-polynomial too, and the norm
    # itself is then decided exactly at evaluation time.
    def scalar_dpoly(self, e):
        an = self.an
        if isinstance(e, (int, float)):
            return []
        if isinstance(e, str):
            return self.aux_scalar.get(e, [])
        if not isinstance(e, list) or not e:
            return []
        op, args = e[0], e[1:]
        if op == 'idx':
            tgt = args[0]
            affected = an.mentions(tgt, self.name,
                                    tuple(self.derived_ast)) or \
                (isinstance(tgt, str) and tgt in self.aux_pat)
            if not affected:
                return []
            j = an.const_value(args[1])
            p = self.pattern(tgt)
            if p is DEPENDS:
                return DEPENDS
            if j is None:
                return [] if is_zero_sv(p) else DEPENDS
            g = _selem(p, int(j))
            return [Fraction(0), g] if g != 0 else []
        if op == 'dot':
            return self._dot_dpoly(args)
        if op == 'norm':
            return self._norm_marker(args[0])
        if op == 'len':
            return []
        if op in ('+', '-'):
            acc = self.scalar_dpoly(args[0])
            if acc is DEPENDS or (isinstance(acc, tuple)):
                return DEPENDS if acc is DEPENDS else DEPENDS
            for a in args[1:]:
                p = self.scalar_dpoly(a)
                if p is DEPENDS or isinstance(p, tuple):
                    return DEPENDS
                acc = _dp_add(acc, p) if op == '+' \
                    else _dp_add(acc, _dp_scale(p, Fraction(-1)))
            return acc
        if op == '*':
            if len(args) != 2:
                return DEPENDS
            A = self.scalar_dpoly(args[0])
            B = self.scalar_dpoly(args[1])
            if A is DEPENDS or B is DEPENDS \
                    or isinstance(A, tuple) or isinstance(B, tuple):
                return DEPENDS
            if not A and not B:
                return []
            b1 = self.scalar_base(args[0]) if B else Fraction(0)
            b2 = self.scalar_base(args[1]) if A else Fraction(0)
            if (B and b1 is None) or (A and b2 is None):
                return DEPENDS
            out = _dp_add(_dp_scale(B, b1 or Fraction(0)),
                          _dp_scale(A, b2 or Fraction(0)))
            out = _dp_add(out, _dp_mul(A, B))
            if len(out) - 1 > DEG_CAP:
                return DEPENDS
            return out
        if op == '/':
            if len(args) != 2:
                return DEPENDS
            A = self.scalar_dpoly(args[0])
            B = self.scalar_dpoly(args[1])
            if A is DEPENDS or B is DEPENDS \
                    or isinstance(A, tuple) or isinstance(B, tuple):
                return DEPENDS
            if B:
                return DEPENDS       # moving denominator: not a poly
            vb = self.scalar_base(args[1])
            if vb is None or vb == 0:
                return DEPENDS
            return _dp_scale(A, 1 / vb)
        if op == 'neg':
            p = self.scalar_dpoly(args[0])
            if p is DEPENDS or isinstance(p, tuple):
                return DEPENDS
            return _dp_scale(p, Fraction(-1))
        if op in FN1:
            p = self.scalar_dpoly(args[0])
            if p is DEPENDS or isinstance(p, tuple):
                return DEPENDS
            return [] if not p else DEPENDS
        if op in ('<', '<=', '>', '>=', '==', '!='):
            for a in args:
                p = self.scalar_dpoly(a)
                if p is DEPENDS or isinstance(p, tuple) or p:
                    return DEPENDS
            return []
        if op == 'if':
            pc = self.scalar_dpoly(args[0])
            if pc is DEPENDS or isinstance(pc, tuple) or pc:
                return DEPENDS
            p1 = self.scalar_dpoly(args[1])
            p2 = self.scalar_dpoly(args[2])
            if p1 is DEPENDS or p2 is DEPENDS \
                    or isinstance(p1, tuple) or isinstance(p2, tuple):
                return DEPENDS
            return p1 if p1 == p2 else DEPENDS
        if op in ('read-int', 'read-eof?'):
            return []
        if an.mentions(e, self.name, tuple(self.derived_ast)):
            return DEPENDS
        return []

    def scalar_base(self, e):
        """Natural value of a scalar expression, for the product
        rule: the analyser's const_value, extended structurally to
        dot products and constant index reads (whose folded values
        the builder does not register as scalar constants)."""
        an = self.an
        v = an.const_value(e)
        if v is not None:
            return v
        if isinstance(e, list) and e:
            if e[0] == 'dot':
                S1 = an.factor_sv(e[1])
                S2 = an.factor_sv(e[2])
                if S1 is not None and S2 is not None:
                    n = an.sym[e[1]][0] if isinstance(e[1], str) \
                        and e[1] in an.sym else None
                    return _sdot(S1, S2, n)
            if e[0] == 'idx':
                S = an.factor_sv(e[1])
                j = an.const_value(e[2])
                if S is not None and j is not None:
                    return _selem(S, int(j))
        return None

    def _dot_dpoly(self, args):
        """Complete degree-2 expansion of a dot consumer:
        Delta = delta*(<P1,S2> + <S1,P2>) + delta^2 * <P1,P2>.
        Natural-value factors are only demanded for sides whose
        partner actually moves (a derived binding's values may not
        be known to the analyser, and are not needed then)."""
        an = self.an
        a, b_ = args
        ma = an.mentions(a, self.name, tuple(self.derived_ast)) or \
            (isinstance(a, str) and a in self.aux_pat)
        mb = an.mentions(b_, self.name, tuple(self.derived_ast)) or \
            (isinstance(b_, str) and b_ in self.aux_pat)
        if not ma and not mb:
            return []
        P1 = self.pattern(a) if ma else ZERO_SV
        P2 = self.pattern(b_) if mb else ZERO_SV
        if P1 is DEPENDS or P2 is DEPENDS:
            return DEPENDS
        lin = Fraction(0)
        if not is_zero_sv(P1):
            S2 = an.factor_sv(b_)
            if S2 is None:
                return DEPENDS
            lin += _sdot(P1, S2, self.n)
        if not is_zero_sv(P2):
            S1 = an.factor_sv(a)
            if S1 is None:
                return DEPENDS
            lin += _sdot(S1, P2, self.n)
        quad = _sdot(P1, P2, self.n)
        return _dp([Fraction(0), lin, quad])

    def _norm_marker(self, e):
        an = self.an
        if not an.mentions(e, self.name, tuple(self.derived_ast)) \
                and not (isinstance(e, str) and e in self.aux_pat):
            return []
        S = an.factor_sv(e)
        if S is None:
            return DEPENDS
        P = self.pattern(e)
        if P is DEPENDS:
            return DEPENDS
        base2 = _sdot(S, S, self.n)
        dpoly2 = _dp([Fraction(0),
                      2 * _sdot(S, P, self.n),
                      _sdot(P, P, self.n)])
        return ('#norm', base2, dpoly2)

    def scalar_gain(self, e):
        """The linear (delta^1) coefficient, for the places that need
        a plain gain (vector patterns, branch conditions)."""
        p = self.scalar_dpoly(e)
        if p is DEPENDS or isinstance(p, tuple):
            return DEPENDS
        return p[1] if len(p) > 1 else Fraction(0)

    def dpoly_is_zero(self, e):
        p = self.scalar_dpoly(e)
        if p is DEPENDS:
            return False
        if isinstance(p, tuple):
            return not p[2]
        return not p

    def _scalar_effect(self, a):
        """Record the effect of printing scalar expression a."""
        an = self.an
        p = self.scalar_dpoly(a)
        if isinstance(p, tuple):
            self.effects.append(('norm', p[1], p[2]))
            return
        if p is DEPENDS:
            # a top-level analytic consumer f(polynomial argument)
            # is still decidable at a known delta
            if isinstance(a, list) and a and a[0] in FN1:
                sub = self.scalar_dpoly(a[1])
                base = self.scalar_base(a[1])
                if sub is not DEPENDS and not isinstance(sub, tuple) \
                        and base is not None:
                    if sub:
                        self.effects.append(
                            ('analytic', a[0], base, sub))
                    else:
                        self.effects.append(('scalar', []))
                    return
            self.effects.append(('depends', 'scalar consumer'))
            return
        self.effects.append(('scalar', p))

    # ------------------------------------------------ statement scan
    def scan_stmts(self, stmts):
        for st in stmts:
            if self.stopped:
                return
            self.scan_stmt(st)

    def scan_stmt(self, st):
        if not isinstance(st, list) or not st:
            return
        op = st[0]
        an = self.an
        if op == 'print':
            for a in st[1:]:
                if isinstance(a, Str):
                    continue
                kind = an.builder()._kind_of(a) \
                    if not isinstance(a, str) else \
                    ('vec' if a == self.name or a in self.derived_ast
                     or a in self.aux_pat else
                     'mat' if a in an.syms or
                     an.shapes.get(a, ('scalar',))[0] == 'mat' else
                     'scalar')
                if kind == 'scalar':
                    self._scalar_effect(a)
                elif kind == 'vec':
                    p = self.pattern(a)
                    self.effects.append(
                        ('depends', 'printed vector') if p is DEPENDS
                        else ('vec', p))
                else:
                    self.effects.append(
                        ('depends', 'printed matrix')
                        if an.mentions(a, self.name,
                                       tuple(self.derived_ast))
                        else ('scalar', Fraction(0)))
            return
        if op in ('let', 'set'):
            name, val = st[1], st[2]
            if name == self.name:
                self.stopped = True
                return
            b = an.builder()
            try:
                kind = b._kind_of(val)
            except Exception:
                kind = 'scalar'
            if kind == 'scalar':
                self.aux_scalar[name] = self.scalar_dpoly(val)
            elif kind == 'vec':
                if an.mentions(val, self.name, tuple(self.derived_ast)) \
                        or (isinstance(val, str)
                            and val in self.derived_ast):
                    self.derived_ast[name] = val
                p = self.pattern(val)
                if p is not DEPENDS:
                    self.aux_pat[name] = p
                elif name in self.aux_pat:
                    del self.aux_pat[name]
            else:
                if an.mentions(val, self.name, tuple(self.derived_ast)):
                    self.effects.append(('depends',
                                         'matrix binding from target'))
            return
        if op == 'set-idx':
            target, idx_e, val = st[1], st[2], st[3]
            if target == self.name:
                j = an.const_value(idx_e)
                if j is not None and int(j) == self.slot:
                    self.stopped = True     # overwritten before use
                else:
                    self.interfering = True
                return
            dp = self.scalar_dpoly(val)
            if dp is DEPENDS or isinstance(dp, tuple) or len(dp) > 2:
                self.effects.append(('depends', 'set-idx value'))
                return
            g = dp[1] if len(dp) > 1 else Fraction(0)
            if g != 0:
                base = self.aux_pat.get(target, ZERO_SV)
                j = an.const_value(idx_e)
                if j is None:
                    self.effects.append(('depends',
                                         'set-idx dynamic index'))
                else:
                    ov = dict(base[1])
                    ov[int(j)] = _selem(base, int(j)) + g
                    self.aux_pat[target] = (base[0], ov, base[2])
            return
        if op in ('for', 'while'):
            if op == 'for' and len(st) >= 5 \
                    and not isinstance(st[3], list):
                body = st[4:]
            elif op == 'for':
                body = st[3:]
            else:
                body = st[2:]
            self.scan_stmts(body)
            return
        if op == 'if':
            if not self.dpoly_is_zero(st[1]):
                self.effects.append(('depends', 'branch condition'))
            for branch in st[2:]:
                if isinstance(branch, list) and branch:
                    self.scan_stmts(
                        branch if isinstance(branch[0], list)
                        else [branch])
            return
        if op == 'block':
            self.scan_stmts(st[1:])
            return


def flatten_events(stmts, out):
    """Linearise statements in execution order (control bodies are
    inlined once; rebinding inside control flow stays conservative
    for the consumer scan, which stops at rebinding markers)."""
    for st in stmts:
        if not isinstance(st, list) or not st:
            continue
        op = st[0]
        if op == 'block':
            flatten_events(st[1:], out)
        elif op in ('for', 'while'):
            if op == 'for' and len(st) >= 5 \
                    and not isinstance(st[3], list):
                body = st[4:]
            elif op == 'for':
                body = st[3:]
            else:
                body = st[2:]
            out.append(('ctrl', st))
            flatten_events(body, out)
        elif op == 'if':
            out.append(('ctrl', st))
            for branch in st[2:]:
                if isinstance(branch, list) and branch:
                    if isinstance(branch[0], list):
                        flatten_events(branch, out)
                    else:
                        flatten_events([branch], out)
        else:
            out.append(('stmt', st))


def predict_program(stmts):
    """Return a list of site reports (dicts)."""
    an = Analyzer()
    events = []
    flatten_events(stmts, events)
    sites = []
    i = 0
    while i < len(events):
        tag, st = events[i]
        i += 1
        if tag != 'stmt':
            continue
        op = st[0]
        if op in ('let', 'set'):
            name, val = st[1], st[2]
            b = an.builder()
            try:
                v = b.lower_any(val)
            except (TypeError, KeyError, IndexError):
                v = None
            an.sym.pop(name, None)
            an.syms.pop(name, None)
            an.shapes.pop(name, None)
            an.scalar_const.pop(name, None)
            an.vec_consts.pop(name, None)
            an.mat_consts.pop(name, None)
            if v is None:
                an.scalar_const[name] = None
            elif v[0] == 'symvec':
                an.sym[name] = (v[1], v[2])
            elif v[0] == 'symmat':
                an.syms[name] = v
            elif v[0] == 'vec':
                an.shapes[name] = ('vec', len(v[1]))
                if all(b.is_const(el) for el in v[1]):
                    an.vec_consts[name] = [
                        _frac_of_float(b.const_val[el]) for el in v[1]]
            elif v[0] == 'mat':
                r = len(v[1])
                an.shapes[name] = ('mat', (r, len(v[1][0]) if r else 0))
                if all(b.is_const(el) for row in v[1] for el in row):
                    an.mat_consts[name] = [
                        [_frac_of_float(b.const_val[el]) for el in row]
                        for row in v[1]]
            else:
                an.scalar_const[name] = an.const_value(val)
            continue
        if op == 'set-idx':
            name, idx_e, val = st[1], st[2], st[3]
            sites.append(an.site_report(name, idx_e, val, events, i))
            # apply the write to the analysis environment
            if name in an.sym:
                n, sv = an.sym[name]
                j = an.const_value(idx_e)
                w = an.const_value(val)
                if j is not None and w is not None:
                    poly, ov, shift = sv
                    ov = dict(ov)
                    ov[int(j)] = w
                    an.sym[name] = (n, (poly, ov, shift))
                else:
                    del an.sym[name]
                    an.shapes[name] = ('vec', n)
            continue
    return sites


def _site_report(self, name, idx_e, val, events, start):
    rep = {'target': name}
    if name not in self.sym:
        later = [st for _, st in events[start:]]
        used = any(self.mentions(st, name) for st in later)
        rep.update(slot=None, delta=None,
                   verdict='TRANSPARENT' if not used else 'UNKNOWN',
                   reasons=['dead write (target never used again)']
                   if not used else ['target is not symbolic'])
        return rep
    n, sv = self.sym[name]
    j = self.const_value(idx_e)
    if j is None:
        rep.update(slot=None, delta=None, verdict='UNKNOWN',
                   reasons=['dynamic set-idx index'])
        return rep
    slot = int(j)
    natural = _selem(sv, slot)
    w = self.const_value(val)
    delta = None if w is None else w - natural
    rep.update(slot=slot, delta=delta)
    if delta is not None and delta == 0:
        rep.update(verdict='TRANSPARENT',
                   reasons=['written value equals the natural value '
                            '(delta = 0)'])
        return rep
    scan = SiteScan(self, name, n, slot)
    # feed the scan the remaining events as statements
    rest = []
    for tag, st in events[start:]:
        rest.append(st)
    scan.scan_stmts(rest)
    changed = []
    unsure = [e[1] for e in scan.effects if e[0] == 'depends']
    for e in scan.effects:
        k = e[0]
        if k == 'scalar':
            if delta is not None:
                dv = _dp_eval(e[1], delta)
                if dv != 0:
                    changed.append(('scalar', dv))
            elif e[1]:
                unsure.append('scalar consumer')
        elif k == 'vec':
            if not is_zero_sv(e[1]):
                if delta is not None:
                    changed.append(('vec', None))
                else:
                    unsure.append('printed vector')
        elif k == 'norm':
            base2, dp2 = e[1], e[2]
            if delta is not None:
                d2 = _dp_eval(dp2, delta)
                if d2 != 0:
                    changed.append(('norm2', d2))
                    rep['delta_norm'] = math.sqrt(float(base2 + d2)) \
                        - math.sqrt(float(base2))
            elif dp2:
                unsure.append('norm consumer')
        elif k == 'analytic':
            _, fn, base, dp = e
            if delta is not None:
                da = _dp_eval(dp, delta)
                if da != 0:
                    f = abs if fn == 'abs' else getattr(math, fn)
                    changed.append(
                        ('analytic',
                         f(float(base + da)) - f(float(base))))
            elif dp:
                unsure.append('analytic consumer')
    if scan.interfering and any(e[0] == 'depends' for e in scan.effects):
        unsure.append('interfering write plus unmodelled consumer')
    if changed:
        rep.update(verdict='DESTRUCTIVE',
                   reasons=['a downstream consumer provably changes'],
                   predicted=changed)
        return rep
    if unsure:
        rep.update(verdict='UNKNOWN', reasons=unsure)
        return rep
    rep.update(verdict='TRANSPARENT',
               reasons=['every downstream consumer is provably '
                        'unchanged (delta-polynomials all vanish)'])
    return rep


Analyzer.site_report = _site_report


def main():
    if len(sys.argv) != 2:
        print('usage: linkpredict.py program.link')
        return 2
    with open(sys.argv[1]) as f:
        stmts = parse(f.read())
    sites = predict_program(stmts)
    if not sites:
        print('no set-idx sites')
        return 0
    for k, rep in enumerate(sites):
        d = rep['delta']
        d_s = 'unknown' if d is None else str(d)
        print(f'site {k}: set-idx {rep["target"]}[{rep["slot"]}] '
              f'delta={d_s} -> {rep["verdict"]}')
        for r in rep['reasons']:
            print(f'    {r}')
        for kind, dd in rep.get('predicted', []):
            if dd is not None:
                print(f'    predicted change in a printed scalar: '
                      f'{dd} (exact)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
