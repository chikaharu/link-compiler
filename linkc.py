#!/usr/bin/env python3
"""
linkc v0.1 -- Link compiler: AST syntax (S-expressions) -> Rust source -> native binary.

The input program *is* an AST. There is no separate surface syntax to parse
into a tree: the .link file is the tree, written in parentheses.

    (program
      (let a (vec 1.0 2.0 3.0 4.0))
      (let b (vec 4.0 3.0 2.0 1.0))
      (print "dot =" (dot a b))
      (print "diff(cumsum(a)) =" (diff (cumsum a))))

Types (statically inferred, checked at compile time):
    Scalar  f64
    Vec     Vec<f64>
    Mat     Vec<Vec<f64>>
    Bool    bool
    Str     &'static str (literals only, for print labels)

Link-core operations are ordinary mathematical operations, named with the
standard terms: dot (inner product), outer (outer product), gram (Gram
matrix), cumsum (cumulative-sum operator), diff (discrete difference,
the exact inverse of cumsum), matvec, matmul, transpose, norm.

Usage:
    python3 linkc.py examples/demo.link -o out/demo [--run]
"""
import argparse
import os
import shutil
import subprocess
import sys

# ---------------------------------------------------------------- parser

class Str(str):
    """String literal node (distinct from a symbol, which is a plain str)."""

def tokenize(src):
    toks, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
        elif c == ';':                      # comment to end of line
            while i < n and src[i] != '\n':
                i += 1
        elif c in '()':
            toks.append(c); i += 1
        elif c == '"':
            j = i + 1; buf = []
            while j < n and src[j] != '"':
                if src[j] == '\\' and j + 1 < n:
                    buf.append(src[j + 1]); j += 2
                else:
                    buf.append(src[j]); j += 1
            if j >= n:
                raise SyntaxError('unterminated string literal')
            toks.append(Str(''.join(buf))); i = j + 1
        else:
            j = i
            while j < n and not src[j].isspace() and src[j] not in '();"':
                j += 1
            toks.append(src[i:j]); i = j
    return toks

def parse(src):
    toks = tokenize(src)
    pos = 0
    def one():
        nonlocal pos
        if pos >= len(toks):
            raise SyntaxError('unexpected end of input')
        t = toks[pos]; pos += 1
        if isinstance(t, Str):
            return t
        if t == '(':
            out = []
            while pos < len(toks) and toks[pos] != ')':
                out.append(one())
            if pos >= len(toks):
                raise SyntaxError('missing )')
            pos += 1
            return out
        if t == ')':
            raise SyntaxError('unexpected )')
        try:
            return int(t)
        except ValueError:
            pass
        try:
            return float(t)
        except ValueError:
            return t                      # symbol
    forms = []
    while pos < len(toks):
        forms.append(one())
    if len(forms) == 1 and isinstance(forms[0], list) and forms[0] and forms[0][0] == 'program':
        return forms[0][1:]
    return forms                          # bare sequence of statements

# ---------------------------------------------------------------- codegen

SCALAR, VEC, MAT, BOOL, STR = 'Scalar', 'Vec', 'Mat', 'Bool', 'Str'

HELPERS = r'''
fn vadd(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() }
fn vsub(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() }
fn vmul(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x * y).collect() }
fn vdiv(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x / y).collect() }
fn vs_add(a: &Vec<f64>, s: f64) -> Vec<f64> { a.iter().map(|x| x + s).collect() }
fn vs_sub(a: &Vec<f64>, s: f64) -> Vec<f64> { a.iter().map(|x| x - s).collect() }
fn sv_sub(s: f64, a: &Vec<f64>) -> Vec<f64> { a.iter().map(|x| s - x).collect() }
fn vs_mul(a: &Vec<f64>, s: f64) -> Vec<f64> { a.iter().map(|x| x * s).collect() }
fn vs_div(a: &Vec<f64>, s: f64) -> Vec<f64> { a.iter().map(|x| x / s).collect() }
fn sv_div(s: f64, a: &Vec<f64>) -> Vec<f64> { a.iter().map(|x| s / x).collect() }
fn vneg(a: &Vec<f64>) -> Vec<f64> { a.iter().map(|x| -x).collect() }
fn vmap(a: &Vec<f64>, f: fn(f64) -> f64) -> Vec<f64> { a.iter().map(|&x| f(x)).collect() }
fn madd(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    a.iter().zip(b).map(|(r1, r2)| r1.iter().zip(r2).map(|(x, y)| x + y).collect()).collect()
}
fn msmul(a: &Vec<Vec<f64>>, s: f64) -> Vec<Vec<f64>> {
    a.iter().map(|r| r.iter().map(|x| x * s).collect()).collect()
}
fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn outer(a: &Vec<f64>, b: &Vec<f64>) -> Vec<Vec<f64>> {
    a.iter().map(|&x| b.iter().map(|&y| x * y).collect()).collect()
}
fn gram(vs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    vs.iter().map(|u| vs.iter().map(|v| dot(u, v)).collect()).collect()
}
fn matvec(m: &Vec<Vec<f64>>, v: &Vec<f64>) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}
fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let bt = transpose(b);
    a.iter().map(|row| bt.iter().map(|col| dot(row, col)).collect()).collect()
}
fn transpose(m: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if m.is_empty() { return Vec::new(); }
    (0..m[0].len()).map(|j| m.iter().map(|row| row[j]).collect()).collect()
}
fn cumsum(v: &Vec<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len()); let mut s = 0.0;
    for &x in v { s += x; out.push(s); } out
}
fn diff(v: &Vec<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len()); let mut prev = 0.0;
    for &x in v { out.push(x - prev); prev = x; } out
}
fn norm(v: &Vec<f64>) -> f64 { dot(v, v).sqrt() }
// ---------------------------------------------------------------- TexNP
// NumPy-equivalent core, written with the Rust standard library only:
// no external crates, no BLAS/LAPACK, no Python/NumPy at run time.
fn np_arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut out = Vec::new(); let mut x = start;
    if step > 0.0 { while x < stop { out.push(x); x += step; } }
    else if step < 0.0 { while x > stop { out.push(x); x += step; } }
    out
}
fn np_linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![start]; }
    let step = (stop - start) / ((n - 1) as f64);
    let mut out: Vec<f64> = (0..n).map(|i| start + step * (i as f64)).collect();
    out[n - 1] = stop; out
}
fn np_eye(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}
fn np_sum_v(v: &Vec<f64>) -> f64 { v.iter().sum() }
fn np_sum_m(m: &Vec<Vec<f64>>) -> f64 { m.iter().map(|r| r.iter().sum::<f64>()).sum() }
fn np_prod_v(v: &Vec<f64>) -> f64 { v.iter().product() }
fn np_mean_v(v: &Vec<f64>) -> f64 { if v.is_empty() { f64::NAN } else { np_sum_v(v) / (v.len() as f64) } }
fn np_mean_m(m: &Vec<Vec<f64>>) -> f64 {
    let n: usize = m.iter().map(|r| r.len()).sum();
    if n == 0 { f64::NAN } else { np_sum_m(m) / (n as f64) }
}
fn np_var_v(v: &Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let mu = np_mean_v(v);
    v.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (v.len() as f64)
}
fn np_std_v(v: &Vec<f64>) -> f64 { np_var_v(v).sqrt() }
fn np_min_v(v: &Vec<f64>) -> f64 { v.iter().cloned().fold(f64::INFINITY, f64::min) }
fn np_max_v(v: &Vec<f64>) -> f64 { v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
fn np_argmin_v(v: &Vec<f64>) -> f64 {
    let mut bi = 0usize; for i in 1..v.len() { if v[i] < v[bi] { bi = i; } } bi as f64
}
fn np_argmax_v(v: &Vec<f64>) -> f64 {
    let mut bi = 0usize; for i in 1..v.len() { if v[i] > v[bi] { bi = i; } } bi as f64
}
fn np_median_v(v: &Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let mut s = v.clone(); s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = s.len() / 2;
    if s.len() % 2 == 1 { s[mid] } else { 0.5 * (s[mid - 1] + s[mid]) }
}
fn np_cumprod(v: &Vec<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len()); let mut p = 1.0;
    for &x in v { p *= x; out.push(p); } out
}
fn np_concat(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut out = a.clone(); out.extend(b.iter().cloned()); out
}
fn np_reverse(v: &Vec<f64>) -> Vec<f64> { v.iter().rev().cloned().collect() }
fn np_sort(v: &Vec<f64>) -> Vec<f64> {
    let mut out = v.clone(); out.sort_by(|a, b| a.partial_cmp(b).unwrap()); out
}
fn np_argsort(v: &Vec<f64>) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    idx.iter().map(|&i| i as f64).collect()
}
fn np_slice(v: &Vec<f64>, start: f64, end: f64) -> Vec<f64> {
    let n = v.len() as i64;
    let s = (start as i64).max(0).min(n) as usize;
    let e = (end as i64).max(0).min(n) as usize;
    if e <= s { Vec::new() } else { v[s..e].to_vec() }
}
fn np_reshape(v: &Vec<f64>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    assert_eq!(v.len(), rows * cols, "reshape: size mismatch");
    (0..rows).map(|i| v[i * cols..(i + 1) * cols].to_vec()).collect()
}
fn np_flatten(m: &Vec<Vec<f64>>) -> Vec<f64> { m.iter().flat_map(|r| r.iter().cloned()).collect() }
fn np_trace(m: &Vec<Vec<f64>>) -> f64 {
    let n = m.len().min(m.first().map(|r| r.len()).unwrap_or(0));
    (0..n).map(|i| m[i][i]).sum()
}
fn np_diag(v: &Vec<f64>) -> Vec<Vec<f64>> {
    let n = v.len();
    (0..n).map(|i| (0..n).map(|j| if i == j { v[i] } else { 0.0 }).collect()).collect()
}
fn np_diag_of(m: &Vec<Vec<f64>>) -> Vec<f64> {
    let n = m.len().min(m.first().map(|r| r.len()).unwrap_or(0));
    (0..n).map(|i| m[i][i]).collect()
}
fn np_clip_v(v: &Vec<f64>, lo: f64, hi: f64) -> Vec<f64> {
    v.iter().map(|x| x.max(lo).min(hi)).collect()
}
fn np_maximum_vv(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x.max(*y)).collect()
}
fn np_maximum_vs(a: &Vec<f64>, s: f64) -> Vec<f64> { a.iter().map(|x| x.max(s)).collect() }
fn np_minimum_vv(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x.min(*y)).collect()
}
fn np_minimum_vs(a: &Vec<f64>, s: f64) -> Vec<f64> { a.iter().map(|x| x.min(s)).collect() }
fn np_pow_v(a: &Vec<f64>, s: f64) -> Vec<f64> { a.iter().map(|x| x.powf(s)).collect() }
fn np_pow_sv(s: f64, a: &Vec<f64>) -> Vec<f64> { a.iter().map(|x| s.powf(*x)).collect() }
// --- linear algebra without LAPACK: Gauss elimination, partial pivoting
fn np_det(m_in: &Vec<Vec<f64>>) -> f64 {
    let n = m_in.len(); let mut m = m_in.clone(); let mut det = 1.0;
    for c in 0..n {
        let mut piv = c;
        for r in (c + 1)..n { if m[r][c].abs() > m[piv][c].abs() { piv = r; } }
        if m[piv][c] == 0.0 { return 0.0; }
        if piv != c { m.swap(c, piv); det = -det; }
        det *= m[c][c];
        let p = m[c][c];
        for r in (c + 1)..n { let f = m[r][c] / p; if f != 0.0 {
            for k in c..n { let mc = m[c][k]; m[r][k] -= f * mc; } } }
    }
    det
}
fn np_solve(a_in: &Vec<Vec<f64>>, b_in: &Vec<f64>) -> Vec<f64> {
    let n = a_in.len();
    let mut m: Vec<Vec<f64>> = a_in.iter().zip(b_in).map(|(row, &bi)| {
        let mut r = row.clone(); r.push(bi); r }).collect();
    for c in 0..n {
        let mut piv = c;
        for r in (c + 1)..n { if m[r][c].abs() > m[piv][c].abs() { piv = r; } }
        assert!(m[piv][c] != 0.0, "solve: singular matrix");
        if piv != c { m.swap(c, piv); }
        let p = m[c][c];
        for k in c..=n { let v = m[c][k] / p; m[c][k] = v; }
        for r in 0..n { if r != c { let f = m[r][c]; if f != 0.0 {
            for k in c..=n { let mc = m[c][k]; m[r][k] -= f * mc; } } } }
    }
    (0..n).map(|i| m[i][n]).collect()
}
fn np_inv(a_in: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a_in.len();
    let mut m: Vec<Vec<f64>> = a_in.iter().enumerate().map(|(i, row)| {
        let mut r = row.clone();
        r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 })); r }).collect();
    for c in 0..n {
        let mut piv = c;
        for r in (c + 1)..n { if m[r][c].abs() > m[piv][c].abs() { piv = r; } }
        assert!(m[piv][c] != 0.0, "inv: singular matrix");
        if piv != c { m.swap(c, piv); }
        let p = m[c][c];
        for k in c..(2 * n) { let v = m[c][k] / p; m[c][k] = v; }
        for r in 0..n { if r != c { let f = m[r][c]; if f != 0.0 {
            for k in c..(2 * n) { let mc = m[c][k]; m[r][k] -= f * mc; } } } }
    }
    m.iter().map(|r| r[n..].to_vec()).collect()
}
fn np_rank(m_in: &Vec<Vec<f64>>) -> f64 {
    let mut m = m_in.clone();
    let rows = m.len(); if rows == 0 { return 0.0; }
    let cols = m[0].len();
    let scale = m.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f64, f64::max);
    let tol = scale * 1e-10; let mut rank = 0usize; let mut c = 0usize;
    while rank < rows && c < cols {
        let mut piv = rank;
        for r in (rank + 1)..rows { if m[r][c].abs() > m[piv][c].abs() { piv = r; } }
        if m[piv][c].abs() <= tol { c += 1; continue; }
        if piv != rank { m.swap(rank, piv); }
        let p = m[rank][c];
        for r in (rank + 1)..rows { let f = m[r][c] / p; if f != 0.0 {
            for k in c..cols { let mc = m[rank][k]; m[r][k] -= f * mc; } } }
        rank += 1; c += 1;
    }
    rank as f64
}
fn np_det_rank1(a: &Vec<f64>, b: &Vec<f64>, coef: f64) -> f64 {
    // det of a rank-1 matrix coef*a*b^T: exactly 0 for n > 1
    if a.len() > 1 { 0.0 } else if a.is_empty() { 1.0 } else { coef * a[0] * b[0] }
}
fn np_matpow(m: &Vec<Vec<f64>>, k: f64) -> Vec<Vec<f64>> {
    let n = m.len(); let mut out = np_eye(n); let mut base = m.clone();
    let mut e = k as i64; assert!(e >= 0, "matpow: negative exponent not supported");
    while e > 0 { if e % 2 == 1 { out = matmul(&out, &base); } base = matmul(&base, &base); e /= 2; }
    out
}
fn read_num() -> f64 {
    use std::sync::OnceLock;
    static TOKENS: OnceLock<Vec<String>> = OnceLock::new();
    static POS: OnceLock<std::sync::Mutex<usize>> = OnceLock::new();
    let toks = TOKENS.get_or_init(|| {
        let mut s = String::new();
        let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut s);
        s.split_whitespace().map(|t| t.to_string()).collect()
    });
    let posm = POS.get_or_init(|| std::sync::Mutex::new(0));
    let mut p = posm.lock().unwrap();
    if *p >= toks.len() { return 0.0; }
    let v = toks[*p].parse::<f64>().unwrap_or(0.0);
    *p += 1;
    v
}
'''

class Codegen:
    def __init__(self):
        self.env = {}                     # name -> type
        self.factors = {}                 # Mat name -> (a_node, b_node, coef_node|None)
        self.lazy = set()                 # Mat names bound ONLY as factors (never materialised)
        self.lines = []
        self.tmp = 0

    def materialize_expr(self, name):
        """Rust expression building the dense matrix of a factorised name."""
        a, b, coef = self.factors[name]
        base = f'outer({self.ref(a)}, {self.ref(b)})'
        if coef is not None:
            return f'msmul(&({base}), {self.coef_expr(coef)})'
        return base

    def flush_lazy(self, indent):
        """Materialise every lazy (factor-only) binding before a write makes
        its factors stale, so the name stays usable as a dense matrix."""
        for name in sorted(self.lazy):
            self.lines.append(f'{indent}let mut {name} = {self.materialize_expr(name)};')
        self.lazy.clear()

    # -- rankop factorisation: recognise Mat expressions of the form
    #    coef * a b^T (rank-1, factors kept as AST nodes). Consumers such as
    #    matmul/matvec/trace/det/sum can then work on the factors in O(n)
    #    or O(n^2) instead of materialising and multiplying densely in
    #    O(n^2)/O(n^3). Anything unrecognised returns None (dense path).
    def factorize(self, node):
        if isinstance(node, str) and not isinstance(node, Str):
            return self.factors.get(node)
        if not isinstance(node, list) or not node:
            return None
        op = node[0]
        if op == 'outer' and len(node) == 3:
            return (node[1], node[2], None)
        if op == 'transpose' and len(node) == 2:
            f = self.factorize(node[1])
            return (f[1], f[0], f[2]) if f else None
        if op == '*' and len(node) == 3:
            for i, j in ((1, 2), (2, 1)):
                try:
                    if self.infer(node[i]) != SCALAR:
                        continue
                except TypeError:
                    continue
                f = self.factorize(node[j])
                if f:
                    coef = node[i] if f[2] is None else ['*', f[2], node[i]]
                    return (f[0], f[1], coef)
            return None
        if op == 'matmul' and len(node) == 3:
            fx, fy = self.factorize(node[1]), self.factorize(node[2])
            if fx and fy:
                parts = [c for c in (fx[2], fy[2]) if c is not None]
                parts.append(['dot', fx[1], fy[0]])
                coef = parts[0] if len(parts) == 1 else ['*'] + parts
                return (fx[0], fy[1], coef)
            return None
        return None

    def coef_expr(self, coef):
        return self.expr(coef) if coef is not None else '1.0'

    def err(self, msg):
        raise TypeError(msg)

    # -- reference form: &x for a variable, &(expr) otherwise
    def ref(self, node):
        if isinstance(node, str) and not isinstance(node, Str) and node in self.env:
            if node in self.lazy:
                return '&(' + self.materialize_expr(node) + ')'
            return '&' + node
        return '&(' + self.expr(node) + ')'

    def infer(self, node):
        if isinstance(node, (int, float)):
            return SCALAR
        if isinstance(node, Str):
            return STR
        if isinstance(node, str):
            if node in ('true', 'false'):
                return BOOL
            if node not in self.env:
                self.err(f'undefined variable: {node}')
            return self.env[node]
        if not isinstance(node, list) or not node:
            self.err(f'bad expression: {node!r}')
        op = node[0]
        if op in ('+', '-', '*', '/'):
            ts = [self.infer(a) for a in node[1:]]
            if MAT in ts: return MAT
            if VEC in ts: return VEC
            return SCALAR
        if op == 'neg': return self.infer(node[1])
        if op in ('sqrt', 'exp', 'log', 'log2', 'log10', 'abs', 'sin', 'cos', 'tan',
                  'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                  'floor', 'ceil', 'round', 'trunc'):
            return self.infer(node[1])
        if op in ('<', '<=', '>', '>=', '==', '!=', 'and', 'or', 'not'):
            return BOOL
        if op == 'if': return self.infer(node[2])
        if op == 'read-int': return SCALAR
        if op == 'vec': return VEC
        if op in ('zeros', 'ones', 'iota'): return VEC
        if op in ('len', 'idx', 'dot', 'norm'): return SCALAR
        if op == 'outer': return MAT
        if op == 'gram': return MAT
        if op == 'matvec': return VEC
        if op == 'matmul': return MAT
        if op == 'transpose': return MAT
        if op in ('cumsum', 'diff'): return VEC
        # ---- TexNP (NumPy-equivalent core, dependency-free) ----
        if op in ('arange', 'linspace', 'full'): return VEC
        if op in ('zeros2', 'ones2', 'full2', 'eye', 'mat', 'diag',
                  'reshape', 'inv', 'matpow'): return MAT
        if op in ('sum', 'prod', 'mean', 'var', 'std', 'vmin', 'vmax',
                  'argmin', 'argmax', 'median', 'trace', 'det', 'rank',
                  'rows', 'cols'): return SCALAR
        if op in ('cumprod', 'concat', 'reverse', 'sort', 'argsort',
                  'slice', 'flatten', 'diag-of', 'solve'): return VEC
        if op in ('pow', 'clip', 'maximum', 'minimum'):
            return self.infer(node[1])
        self.err(f'unknown operator: {op}')

    # -- expression -> Rust code producing an owned value
    def expr(self, node):
        if isinstance(node, (int, float)):
            s = repr(float(node))
            return s if ('.' in s or 'e' in s or 'inf' in s or 'NaN' in s) else s + '.0'
        if isinstance(node, Str):
            return '"' + node.replace('\\', '\\\\').replace('"', '\\"') + '"'
        if isinstance(node, str):
            if node == 'true': return 'true'
            if node == 'false': return 'false'
            if node in self.lazy:
                return self.materialize_expr(node)
            t = self.infer(node)
            return node + '.clone()' if t in (VEC, MAT) else node
        op, args = node[0], node[1:]

        if op in ('+', '-', '*', '/'):
            fn = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            if op == '-' and len(args) == 1:
                t = self.infer(args[0])
                return f'vneg({self.ref(args[0])})' if t == VEC else f'(-{self.expr(args[0])})'
            code = self.expr(args[0]); ct = self.infer(args[0])
            for a in args[1:]:
                at = self.infer(a)
                if ct == SCALAR and at == SCALAR:
                    code = f'({code} {op} {self.expr(a)})'
                elif ct == VEC and at == VEC:
                    code = f'v{fn}(&({code}), {self.ref(a)})'
                elif ct == VEC and at == SCALAR:
                    code = f'vs_{fn}(&({code}), {self.expr(a)})'
                elif ct == SCALAR and at == VEC:
                    if fn in ('add', 'mul'):
                        code = f'vs_{fn}({self.ref(a)}, {code})'
                    else:
                        code = f'sv_{fn}({code}, {self.ref(a)})'
                    ct = VEC; continue
                elif ct == MAT and at == SCALAR and fn == 'mul':
                    code = f'msmul(&({code}), {self.expr(a)})'
                elif ct == SCALAR and at == MAT and fn == 'mul':
                    code = f'msmul({self.ref(a)}, {code})'
                    ct = MAT; continue
                elif ct == MAT and at == MAT and fn == 'add':
                    code = f'madd(&({code}), {self.ref(a)})'
                else:
                    self.err(f'type mismatch for {op}: {ct} and {at}')
                ct = ct if ct != SCALAR else at
            return code
        if op == 'neg':
            t = self.infer(args[0])
            return f'vneg({self.ref(args[0])})' if t == VEC else f'(-{self.expr(args[0])})'
        if op in ('sqrt', 'exp', 'log', 'log2', 'log10', 'abs', 'sin', 'cos', 'tan',
                  'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                  'floor', 'ceil', 'round', 'trunc'):
            f = {'sqrt': 'sqrt', 'exp': 'exp', 'log': 'ln', 'log2': 'log2',
                 'log10': 'log10', 'abs': 'abs', 'sin': 'sin', 'cos': 'cos',
                 'tan': 'tan', 'asin': 'asin', 'acos': 'acos', 'atan': 'atan',
                 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'floor': 'floor',
                 'ceil': 'ceil', 'round': 'round', 'trunc': 'trunc'}[op]
            t = self.infer(args[0])
            if t == VEC:
                return f'vmap({self.ref(args[0])}, f64::{f})'
            if t == MAT:
                self.err(f'{op} on Mat not supported; flatten first')
            # f64::f(x), not x.f(): a literal receiver (16.0.sqrt()) is an
            # ambiguous float type to rustc (E0689).
            return f'f64::{f}({self.expr(args[0])})'
        if op in ('<', '<=', '>', '>=', '==', '!='):
            rust = {'==': '==', '!=': '!='}.get(op, op)
            return f'({self.expr(args[0])} {rust} {self.expr(args[1])})'
        if op == 'and': return f'({self.expr(args[0])} && {self.expr(args[1])})'
        if op == 'or':  return f'({self.expr(args[0])} || {self.expr(args[1])})'
        if op == 'not': return f'(!{self.expr(args[0])})'
        if op == 'if':
            return (f'if {self.expr(args[0])} {{ {self.expr(args[1])} }} '
                    f'else {{ {self.expr(args[2])} }}')
        if op == 'read-int':
            return 'read_num()'
        if op == 'vec':
            return 'vec![' + ', '.join(self.expr(a) for a in args) + ']'
        if op == 'zeros': return f'vec![0.0; ({self.expr(args[0])}) as usize]'
        if op == 'ones':  return f'vec![1.0; ({self.expr(args[0])}) as usize]'
        if op == 'iota':  return f'(0..({self.expr(args[0])}) as usize).map(|i| i as f64).collect::<Vec<f64>>()'
        if op == 'len':   return f'({self.expr(args[0])}).len() as f64'
        if op == 'idx':   return f'({self.expr(args[0])})[({self.expr(args[1])}) as usize]'
        if op == 'dot':   return f'dot({self.ref(args[0])}, {self.ref(args[1])})'
        if op == 'norm':  return f'norm({self.ref(args[0])})'
        if op == 'outer': return f'outer({self.ref(args[0])}, {self.ref(args[1])})'
        if op == 'gram':
            inner = ', '.join(self.expr(a) for a in args)
            return f'gram(&vec![{inner}])'
        if op == 'matvec':
            f = self.factorize(args[0])
            if f:  # (coef*a b^T) x = a * (coef * (b.x))          -- O(n)
                return (f'vs_mul({self.ref(f[0])}, '
                        f'({self.coef_expr(f[2])} * dot({self.ref(f[1])}, {self.ref(args[1])})))')
            return f'matvec({self.ref(args[0])}, {self.ref(args[1])})'
        if op == 'matmul':
            fx, fy = self.factorize(args[0]), self.factorize(args[1])
            if fx and fy:  # (a b^T)(c d^T) = (b.c) * a d^T        -- O(n) + O(n^2) materialise
                base = f'outer({self.ref(fx[0])}, {self.ref(fy[1])})'
                parts = [c for c in (fx[2], fy[2]) if c is not None]
                parts.append(['dot', fx[1], fy[0]])
                coef = parts[0] if len(parts) == 1 else ['*'] + parts
                return f'msmul(&({base}), {self.expr(coef)})'
            if fx:  # (a b^T) M = a (M^T b)^T                       -- O(n^2)
                base = (f'outer({self.ref(fx[0])}, '
                        f'&(matvec(&transpose({self.ref(args[1])}), {self.ref(fx[1])})))')
                if fx[2] is not None:
                    return f'msmul(&({base}), {self.expr(fx[2])})'
                return base
            if fy:  # M (a b^T) = (M a) b^T                         -- O(n^2)
                base = (f'outer(&(matvec({self.ref(args[0])}, {self.ref(fy[0])})), '
                        f'{self.ref(fy[1])})')
                if fy[2] is not None:
                    return f'msmul(&({base}), {self.expr(fy[2])})'
                return base
            return f'matmul({self.ref(args[0])}, {self.ref(args[1])})'
        if op == 'transpose':
            f = self.factorize(args[0])
            if f:  # (coef*a b^T)^T = coef*b a^T
                base = f'outer({self.ref(f[1])}, {self.ref(f[0])})'
                if f[2] is not None:
                    return f'msmul(&({base}), {self.expr(f[2])})'
                return base
            return f'transpose({self.ref(args[0])})'
        if op == 'cumsum':    return f'cumsum({self.ref(args[0])})'
        if op == 'diff':      return f'diff({self.ref(args[0])})'
        # ---- TexNP ----
        if op == 'arange':
            if len(args) == 1:
                return f'np_arange(0.0, {self.expr(args[0])}, 1.0)'
            if len(args) == 2:
                return f'np_arange({self.expr(args[0])}, {self.expr(args[1])}, 1.0)'
            return (f'np_arange({self.expr(args[0])}, {self.expr(args[1])}, '
                    f'{self.expr(args[2])})')
        if op == 'linspace':
            return (f'np_linspace({self.expr(args[0])}, {self.expr(args[1])}, '
                    f'({self.expr(args[2])}) as usize)')
        if op == 'full':
            return f'vec![{self.expr(args[1])}; ({self.expr(args[0])}) as usize]'
        if op == 'zeros2':
            return (f'vec![vec![0.0; ({self.expr(args[1])}) as usize]; '
                    f'({self.expr(args[0])}) as usize]')
        if op == 'ones2':
            return (f'vec![vec![1.0; ({self.expr(args[1])}) as usize]; '
                    f'({self.expr(args[0])}) as usize]')
        if op == 'full2':
            return (f'vec![vec![{self.expr(args[2])}; ({self.expr(args[1])}) as usize]; '
                    f'({self.expr(args[0])}) as usize]')
        if op == 'eye':
            return f'np_eye(({self.expr(args[0])}) as usize)'
        if op == 'mat':
            inner = ', '.join(self.expr(a) for a in args)
            return f'vec![{inner}]'
        if op in ('sum', 'mean'):
            t = self.infer(args[0])
            if t == MAT:
                f = self.factorize(args[0])
                if f:  # sum(coef*a b^T) = coef * sum(a) * sum(b)   -- O(n)
                    s = (f'({self.coef_expr(f[2])} * np_sum_v({self.ref(f[0])}) '
                         f'* np_sum_v({self.ref(f[1])}))')
                    if op == 'sum':
                        return s
                    return (f'({s} / ((({self.expr(f[0])}).len() as f64) '
                            f'* (({self.expr(f[1])}).len() as f64)))')
            suf = 'm' if t == MAT else 'v'
            return f'np_{op}_{suf}({self.ref(args[0])})'
        if op in ('prod', 'var', 'std', 'median'):
            return f'np_{op}_v({self.ref(args[0])})'
        if op == 'vmin': return f'np_min_v({self.ref(args[0])})'
        if op == 'vmax': return f'np_max_v({self.ref(args[0])})'
        if op == 'argmin': return f'np_argmin_v({self.ref(args[0])})'
        if op == 'argmax': return f'np_argmax_v({self.ref(args[0])})'
        if op == 'cumprod': return f'np_cumprod({self.ref(args[0])})'
        if op == 'concat':
            return f'np_concat({self.ref(args[0])}, {self.ref(args[1])})'
        if op == 'reverse': return f'np_reverse({self.ref(args[0])})'
        if op == 'sort':    return f'np_sort({self.ref(args[0])})'
        if op == 'argsort': return f'np_argsort({self.ref(args[0])})'
        if op == 'slice':
            return (f'np_slice({self.ref(args[0])}, {self.expr(args[1])}, '
                    f'{self.expr(args[2])})')
        if op == 'reshape':
            return (f'np_reshape({self.ref(args[0])}, ({self.expr(args[1])}) as usize, '
                    f'({self.expr(args[2])}) as usize)')
        if op == 'flatten': return f'np_flatten({self.ref(args[0])})'
        if op == 'trace':
            f = self.factorize(args[0])
            if f:  # tr(coef*a b^T) = coef * (a.b)                  -- O(n)
                return f'({self.coef_expr(f[2])} * dot({self.ref(f[0])}, {self.ref(f[1])}))'
            return f'np_trace({self.ref(args[0])})'
        if op == 'diag':    return f'np_diag({self.ref(args[0])})'
        if op == 'diag-of': return f'np_diag_of({self.ref(args[0])})'
        if op == 'rows':    return f'({self.expr(args[0])}).len() as f64'
        if op == 'cols':
            return (f'({self.expr(args[0])}).first().map(|r| r.len()).unwrap_or(0) as f64')
        if op == 'det':
            f = self.factorize(args[0])
            if f:  # rank-1 determinant: 0 for n > 1                -- O(1)
                return (f'np_det_rank1({self.ref(f[0])}, {self.ref(f[1])}, '
                        f'{self.coef_expr(f[2])})')
            return f'np_det({self.ref(args[0])})'
        if op == 'rank':    return f'np_rank({self.ref(args[0])})'
        if op == 'solve':
            return f'np_solve({self.ref(args[0])}, {self.ref(args[1])})'
        if op == 'inv':     return f'np_inv({self.ref(args[0])})'
        if op == 'matpow':
            return f'np_matpow({self.ref(args[0])}, {self.expr(args[1])})'
        if op == 'pow':
            t0, t1 = self.infer(args[0]), self.infer(args[1])
            if t0 == VEC and t1 == SCALAR:
                return f'np_pow_v({self.ref(args[0])}, {self.expr(args[1])})'
            if t0 == SCALAR and t1 == VEC:
                return f'np_pow_sv({self.expr(args[0])}, {self.ref(args[1])})'
            if t0 == SCALAR and t1 == SCALAR:
                return f'({self.expr(args[0])}).powf({self.expr(args[1])})'
            self.err('pow: unsupported type combination')
        if op == 'clip':
            t = self.infer(args[0])
            if t == VEC:
                return (f'np_clip_v({self.ref(args[0])}, {self.expr(args[1])}, '
                        f'{self.expr(args[2])})')
            return (f'({self.expr(args[0])}).max({self.expr(args[1])})'
                    f'.min({self.expr(args[2])})')
        if op in ('maximum', 'minimum'):
            t0, t1 = self.infer(args[0]), self.infer(args[1])
            if t0 == VEC and t1 == VEC:
                return f'np_{op}_vv({self.ref(args[0])}, {self.ref(args[1])})'
            if t0 == VEC and t1 == SCALAR:
                return f'np_{op}_vs({self.ref(args[0])}, {self.expr(args[1])})'
            if t0 == SCALAR and t1 == SCALAR:
                f = 'max' if op == 'maximum' else 'min'
                return f'({self.expr(args[0])}).{f}({self.expr(args[1])})'
            self.err(f'{op}: unsupported type combination')
        self.err(f'unknown operator: {op}')

    # -- statements
    def stmt(self, node, indent='    '):
        if isinstance(node, list) and node and isinstance(node[0], str):
            op = node[0]
            if op == 'let':
                name, val = node[1], node[2]
                t = self.infer(val)
                f = self.factorize(val) if t == MAT else None
                # rankop lazy binding: if both factors are plain variables,
                # bind ONLY the factors (plus the coefficient as a hidden
                # scalar). The n x n matrix is never built unless a dense
                # consumer asks for it. This is what lifts the size limit:
                # a rank-1 matrix of size 100000 needs 2 vectors, not 80 GB.
                if f and all(isinstance(x, str) and not isinstance(x, Str)
                             and self.env.get(x) == VEC for x in (f[0], f[1])):
                    stored_coef = None
                    if f[2] is not None:
                        stored_coef = '__coef_' + name
                        self.lines.append(
                            f'{indent}let {stored_coef}: f64 = {self.expr(f[2])};')
                        self.env[stored_coef] = SCALAR
                    self.env[name] = MAT
                    self.factors[name] = (f[0], f[1], stored_coef)
                    self.lazy.add(name)
                    return
                self.lines.append(f'{indent}let mut {name} = {self.expr(val)};')
                self.env[name] = t
                self.lazy.discard(name)
                if f:
                    self.factors[name] = f
                else:
                    self.factors.pop(name, None)
                return
            if op == 'set':
                name, val = node[1], node[2]
                if name not in self.env: self.err(f'set of undefined variable: {name}')
                # a write may invalidate any recorded factorisation:
                # materialise lazy bindings first so they survive as dense
                self.flush_lazy(indent)
                self.factors.clear()
                self.lines.append(f'{indent}{name} = {self.expr(val)};')
                return
            if op == 'set-idx':
                self.flush_lazy(indent)
                self.factors.clear()
                self.lines.append(
                    f'{indent}{{ let __i = ({self.expr(node[2])}) as usize; '
                    f'{node[1]}[__i] = {self.expr(node[3])}; }}')
                return
            if op == 'print':
                fmts, vals = [], []
                for a in node[1:]:
                    t = self.infer(a)
                    fmts.append('{:?}' if t in (VEC, MAT) else '{}')
                    vals.append(self.expr(a))
                joined = ' '.join(fmts)
                suffix = (', ' + ', '.join(vals)) if vals else ''
                self.lines.append(f'{indent}println!("{joined}"{suffix});')
                return
            if op == 'for':
                var = node[1]
                if len(node) < 4: self.err('for needs a bound and a body')
                # (for i n body...) or (for i start end body...):
                # the start/end form is recognised by node[3] not being a statement list.
                if len(node) >= 5 and not isinstance(node[3], list):
                    lo, hi, body = node[2], node[3], node[4:]
                    rng = f'({self.expr(lo)}) as i64..({self.expr(hi)}) as i64'
                else:
                    lo, body = None, node[3:]
                    rng = f'0..({self.expr(node[2])}) as i64'
                self.lines.append(f'{indent}for __i in {rng} {{')
                self.lines.append(f'{indent}    let {var}: f64 = __i as f64;')
                had = self.env.get(var); self.env[var] = SCALAR
                for s in body: self.stmt(s, indent + '    ')
                if had is None: self.env.pop(var, None)
                else: self.env[var] = had
                self.lines.append(f'{indent}}}')
                return
            if op == 'while':
                self.lines.append(f'{indent}while {self.expr(node[1])} {{')
                for s in node[2:]: self.stmt(s, indent + '    ')
                self.lines.append(f'{indent}}}')
                return
            if op == 'block':
                for s in node[1:]: self.stmt(s, indent)
                return
            if op == 'if':
                self.lines.append(f'{indent}if {self.expr(node[1])} {{')
                self.stmt(node[2], indent + '    ')
                self.lines.append(f'{indent}}} else {{')
                self.stmt(node[3], indent + '    ')
                self.lines.append(f'{indent}}}')
                return
        # expression statement (e.g. set-idx handled above; bare expr is evaluated then dropped)
        self.lines.append(f'{indent}let _ = {self.expr(node)};')

    def compile(self, stmts):
        for s in stmts:
            self.stmt(s)
        body = '\n'.join(self.lines)
        return ('// generated by linkc v0.1 -- do not edit\n'
                '#![allow(dead_code, unused_mut, unused_assignments)]\n'
                + HELPERS + '\nfn main() {\n' + body + '\n}\n')

# ---------------------------------------------------------------- driver

def find_rustc():
    p = shutil.which('rustc')
    if p: return p
    cand = os.path.expanduser('~/.cargo/bin/rustc')
    return cand if os.path.exists(cand) else None

def main():
    ap = argparse.ArgumentParser(description='Link compiler: AST -> Rust -> native binary')
    ap.add_argument('source')
    ap.add_argument('-o', '--out', required=True, help='output binary path')
    ap.add_argument('--emit-rs', help='where to write generated Rust (default: <out>.rs)')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--opt', default='3', help='rustc opt-level (default 3)')
    ap.add_argument('--boundary', action='store_true',
                    help='run linkir boundary reduction (seam cancellation) before codegen; '
                         'scalar programs only, others compile as-is')
    a = ap.parse_args()

    with open(a.source) as f:
        stmts = parse(f.read())
    if a.boundary:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from linkir import Lower, count_nodes
            from linkemit import emit_program
            lowered = Lower(mode='boundary').program(stmts)
            raw = Lower(mode='none').program(stmts)
        except Exception as e:
            print(f'[linkc] boundary reduction not applicable ({e}); compiling as-is',
                  file=sys.stderr)
        else:
            print(f'[linkc] boundary reduction: {count_nodes(raw)} -> '
                  f'{count_nodes(lowered)} 1x1 Links')
            stmts = emit_program(lowered, stmts)
    rs = Codegen().compile(stmts)
    rs_path = a.emit_rs or (a.out + '.rs')
    os.makedirs(os.path.dirname(os.path.abspath(rs_path)), exist_ok=True)
    with open(rs_path, 'w') as f:
        f.write(rs)
    print(f'[linkc] AST -> Rust : {rs_path} ({len(rs.splitlines())} lines)')

    rustc = find_rustc()
    if not rustc:
        print('[linkc] rustc not found; Rust source emitted only.', file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cmd = [rustc, '-C', f'opt-level={a.opt}', '--edition', '2021', rs_path, '-o', a.out]
    print(f'[linkc] Rust -> binary : {a.out}')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode
    if a.run:
        print('[linkc] run:')
        return subprocess.run([os.path.abspath(a.out)]).returncode
    return 0

if __name__ == '__main__':
    sys.exit(main())
