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
                write, or every consumer's gain is exactly 0)
  DESTRUCTIVE   some observable provably changes; the exact change
                Delta = gain * delta is predicted for linear
                (scalar) observables
  UNKNOWN       the analysis abstains (dynamic indices, nonlinear
                or unmodelled consumers, unknown written values)

The predictor is sound by construction: it never claims TRANSPARENT
or DESTRUCTIVE without a proof inside its model. test_linkpredict.py
checks the predictions against ground truth (running the program
with and without the write) on hand cases and a randomised battery.

Usage: python3 linkpredict.py program.link
"""
import sys
from fractions import Fraction

from linkc import parse, Str
from linkir import (Builder, FN1, _selem, _sadd, _ssub, _sscale,
                    _smul, _scumsum, _sdiff, _sdot, _svals, _sfold,
                    _frac_of_float)

ZERO_SV = ([Fraction(0)], {}, Fraction(0))
DEPENDS = 'depends'


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
            g = self.scalar_gain(args[0])
            if g is DEPENDS or (g != 0):
                return DEPENDS
            p1, p2 = self.pattern(args[1]), self.pattern(args[2])
            if p1 is DEPENDS or p2 is DEPENDS:
                return DEPENDS
            return p1 if p1 == p2 else DEPENDS
        if op == 'vec':
            ov = {}
            for j, a in enumerate(args):
                g = self.scalar_gain(a)
                if g is DEPENDS:
                    return DEPENDS
                if g != 0:
                    ov[j] = g
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

    # ------------------------------------------- gain of scalar expr
    def scalar_gain(self, e):
        """d(scalar expression)/d delta: a Fraction, or DEPENDS."""
        an = self.an
        if isinstance(e, (int, float)):
            return Fraction(0)
        if isinstance(e, str):
            return self.aux_scalar.get(e, Fraction(0))
        if not isinstance(e, list) or not e:
            return Fraction(0)
        op, args = e[0], e[1:]
        if op == 'idx':
            tgt = args[0]
            affected = an.mentions(tgt, self.name,
                                    tuple(self.derived_ast)) or \
                (isinstance(tgt, str) and tgt in self.aux_pat)
            if not affected:
                return Fraction(0)
            j = an.const_value(args[1])
            p = self.pattern(tgt)
            if p is DEPENDS:
                return DEPENDS
            if j is None:
                return Fraction(0) if is_zero_sv(p) else DEPENDS
            return _selem(p, int(j))
        if op == 'dot':
            a, b_ = args
            for vec_e, w_e in ((a, b_), (b_, a)):
                if an.mentions(vec_e, self.name,
                               tuple(self.derived_ast)) \
                        or (isinstance(vec_e, str)
                            and vec_e in self.aux_pat) \
                        or (isinstance(vec_e, str)
                            and vec_e in self.derived_ast):
                    p = self.pattern(vec_e)
                    if p is DEPENDS:
                        return DEPENDS
                    w = an.factor_sv(w_e)
                    if w is None:
                        return DEPENDS
                    return _sdot(p, w, self.n)
            return Fraction(0)
        if op == 'norm':
            if not an.mentions(args[0], self.name,
                               tuple(self.derived_ast)):
                return Fraction(0)
            p = self.pattern(args[0])
            if p is DEPENDS:
                return DEPENDS
            return Fraction(0) if is_zero_sv(p) else DEPENDS
        if op == 'len':
            return Fraction(0)
        if op in ('+', '-', '*', '/'):
            gains = [self.scalar_gain(a) for a in args]
            if any(g is DEPENDS for g in gains):
                return DEPENDS
            if all(g == 0 for g in gains):
                return Fraction(0)
            if op in ('+', '-'):
                acc = gains[0]
                for g in gains[1:]:
                    acc = acc + g if op == '+' else acc - g
                return acc
            if len(args) != 2:
                return DEPENDS
            vals = [an.const_value(a) for a in args]
            g1, g2 = gains
            if op == '*':
                if g1 != 0 and g2 != 0:
                    return DEPENDS     # delta^2 cross term
                if g1 != 0 and vals[1] is None:
                    return DEPENDS
                if g2 != 0 and vals[0] is None:
                    return DEPENDS
                return g1 * (vals[1] or Fraction(0)) \
                    + g2 * (vals[0] or Fraction(0))
            # '/': a moving denominator makes the change nonlinear
            va, vb = vals
            if g2 != 0:
                return DEPENDS
            if vb is None or vb == 0:
                return DEPENDS
            return g1 / vb
        if op == 'neg':
            g = self.scalar_gain(args[0])
            return DEPENDS if g is DEPENDS else -g
        if op in FN1:
            g = self.scalar_gain(args[0])
            if g is DEPENDS:
                return DEPENDS
            return Fraction(0) if g == 0 else DEPENDS
        if op in ('<', '<=', '>', '>=', '==', '!='):
            gains = [self.scalar_gain(a) for a in args]
            if any(g is DEPENDS or g != 0 for g in gains):
                return DEPENDS
            return Fraction(0)
        if op == 'if':
            gc = self.scalar_gain(args[0])
            if gc is DEPENDS or gc != 0:
                return DEPENDS
            g1, g2 = self.scalar_gain(args[1]), self.scalar_gain(args[2])
            if g1 is DEPENDS or g2 is DEPENDS:
                return DEPENDS
            return g1 if g1 == g2 else DEPENDS
        if op in ('read-int', 'read-eof?'):
            return Fraction(0)
        if an.mentions(e, self.name, tuple(self.derived_ast)):
            return DEPENDS
        return Fraction(0)

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
                    g = self.scalar_gain(a)
                    self.effects.append(
                        ('depends', 'scalar consumer') if g is DEPENDS
                        else ('scalar', g))
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
                self.aux_scalar[name] = self.scalar_gain(val)
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
            g = self.scalar_gain(val)
            if g is DEPENDS:
                self.effects.append(('depends', 'set-idx value'))
            elif g != 0:
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
            g = self.scalar_gain(st[1])
            if g is DEPENDS or g != 0:
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
    effects = scan.effects
    has_dep = any(e[0] == 'depends' for e in effects)
    nonzero = []
    for kind, payload in effects:
        if kind == 'scalar' and payload != 0:
            nonzero.append(('scalar', payload))
        elif kind == 'vec' and not is_zero_sv(payload):
            nonzero.append(('vec', payload))
    if not nonzero and not has_dep:
        rep.update(verdict='TRANSPARENT',
                   reasons=['every downstream consumer has zero '
                            'crosstalk gain at this slot'])
        return rep
    if delta is not None and nonzero:
        deltas = [(kind, (g * delta if kind == 'scalar' else None))
                  for kind, g in nonzero]
        rep.update(verdict='DESTRUCTIVE',
                   reasons=['a downstream consumer has nonzero '
                            'crosstalk gain at this slot'],
                   predicted=deltas)
        return rep
    if scan.interfering and has_dep:
        rep.update(verdict='UNKNOWN',
                   reasons=['interfering write plus unmodelled '
                            'consumer'])
        return rep
    rep.update(verdict='UNKNOWN',
               reasons=['unmodelled consumer or unknown written value'])
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
