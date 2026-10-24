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
        self.lines = []
        self.tmp = 0

    def err(self, msg):
        raise TypeError(msg)

    # -- reference form: &x for a variable, &(expr) otherwise
    def ref(self, node):
        if isinstance(node, str) and not isinstance(node, Str) and node in self.env:
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
        if op in ('sqrt', 'exp', 'log', 'abs', 'sin', 'cos'):
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
                elif ct == MAT and at == MAT and fn == 'add':
                    code = f'madd(&({code}), {self.ref(a)})'
                else:
                    self.err(f'type mismatch for {op}: {ct} and {at}')
                ct = ct if ct != SCALAR else at
            return code
        if op == 'neg':
            t = self.infer(args[0])
            return f'vneg({self.ref(args[0])})' if t == VEC else f'(-{self.expr(args[0])})'
        if op in ('sqrt', 'exp', 'log', 'abs', 'sin', 'cos'):
            f = {'sqrt': 'sqrt', 'exp': 'exp', 'log': 'ln', 'abs': 'abs', 'sin': 'sin', 'cos': 'cos'}[op]
            t = self.infer(args[0])
            if t == VEC:
                return f'vmap({self.ref(args[0])}, f64::{f})'
            return f'{self.expr(args[0])}.{f}()'
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
        if op == 'matvec':    return f'matvec({self.ref(args[0])}, {self.ref(args[1])})'
        if op == 'matmul':    return f'matmul({self.ref(args[0])}, {self.ref(args[1])})'
        if op == 'transpose': return f'transpose({self.ref(args[0])})'
        if op == 'cumsum':    return f'cumsum({self.ref(args[0])})'
        if op == 'diff':      return f'diff({self.ref(args[0])})'
        self.err(f'unknown operator: {op}')

    # -- statements
    def stmt(self, node, indent='    '):
        if isinstance(node, list) and node and isinstance(node[0], str):
            op = node[0]
            if op == 'let':
                name, val = node[1], node[2]
                t = self.infer(val)
                self.lines.append(f'{indent}let mut {name} = {self.expr(val)};')
                self.env[name] = t
                return
            if op == 'set':
                name, val = node[1], node[2]
                if name not in self.env: self.err(f'set of undefined variable: {name}')
                self.lines.append(f'{indent}{name} = {self.expr(val)};')
                return
            if op == 'set-idx':
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
    a = ap.parse_args()

    with open(a.source) as f:
        stmts = parse(f.read())
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
