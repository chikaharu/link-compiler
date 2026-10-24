"""Boundary representation and boundary reduction (2026-10-24 session).

Two integrated pieces:

1. Stream codec. A stream is stored as its boundary form: the initial
   value plus the list of switch points (position, jump). Extraction is
   the discrete difference (diff), restoration is the cumulative sum
   (cumsum); the constant part lives in ker(diff) and is carried as the
   initial value. Roundtrip is exact.

2. Boundary reduction of compose chains. A chain of linear Links
   F_k ... F_1 (F_1 applied first) has prefix products
   Lambda_j = F_1 ... F_j ("boundary accumulations"). A segment of the
   chain is determined by two boundaries alone:
       F_a ... F_b = Lambda_{a-1}^{-1} @ Lambda_b
   in this codebase's row-vector convention (the inverse sits on the
   other side in function-composition notation), verified exactly over
   Q(i) in the tests. At a seam, an adjacent
   pair whose product is exactly the identity cancels and the two
   boundaries join directly; this rescues values that stepwise
   floating-point evaluation destroys (e.g. (1+1e16)-1e16).
   Cancellation is accepted only when the product is *exactly* the
   identity in float arithmetic -- the same strict rule as the session's
   boundary mode.
"""
from fractions import Fraction
from .syntax import LinkError
from .vm import multiply
from .nullspace import _rat, _mul, _add, _zero, _float

# --- 1. Stream codec ---------------------------------------------------------
def extract(values):
    """Boundary form of a stream: (initial value, [(position, jump)])."""
    values = list(values)
    if not values: return None, []
    switches = [(i, values[i] - values[i - 1])
                for i in range(1, len(values)) if values[i] != values[i - 1]]
    return values[0], switches

def restore(initial, switches, length):
    """Rebuild a stream from its boundary form. Exact inverse of extract."""
    if length == 0: return []
    jumps = dict(switches)
    out = [initial]
    for i in range(1, length):
        out.append(out[-1] + jumps.get(i, 0))
    return out

# --- 2. Compose chains -------------------------------------------------------
def evaluate_registers(program):
    """Evaluate every register of a bytecode program (mirrors vm.run)."""
    regs = {}
    for ins in program.get('registers', []):
        op, idx = ins['op'], ins['dst']
        m, n = ins['shape']
        if op == 'identity':
            a = [[complex(i == j) for j in range(n)] for i in range(m)]
        elif op == 'matrix':
            a = [[complex(*item) for item in row] for row in ins['values']]
        elif op == 'phase2':
            a = [[(1, 1j, -1, -1j)[ins['q']]]]
        elif op == 'adjoint':
            x = regs[ins['arg']]
            a = [[x[j][i].conjugate() for j in range(len(x))] for i in range(len(x[0]))]
        elif op == 'compose':
            a = multiply(regs[ins['right']], regs[ins['left']])
        elif op in ('add', 'sub'):
            left, right = regs[ins['left']], regs[ins['right']]
            sign = 1 if op == 'add' else -1
            a = [[left[i][j] + sign * right[i][j] for j in range(len(left[0]))]
                 for i in range(len(left))]
        else:
            raise LinkError(f'Unknown bytecode opcode {op!r}')
        regs[idx] = a
    return regs

def compose_factors(program):
    """Flatten the result's compose chain into application-ordered factors.

    Returns [F_1, ..., F_k] with F_1 applied first. Only pure compose
    chains of matrix/identity/phase2/adjoint leaves are supported.
    """
    by = {r['dst']: r for r in program['registers']}
    regs = evaluate_registers(program)
    def flatten(idx):
        ins = by[idx]
        if ins['op'] == 'compose':
            return flatten(ins['right']) + flatten(ins['left'])
        if ins['op'] in ('matrix', 'identity', 'phase2', 'adjoint'):
            return [regs[idx]]
        raise LinkError(f'Result is not a pure compose chain (saw {ins["op"]!r})')
    return flatten(program['result'])

def _is_identity(A):
    m, n = len(A), len(A[0])
    return m == n and all(A[i][j] == (1 if i == j else 0)
                           for i in range(m) for j in range(n))

def chain_product(factors):
    out = factors[0]
    for F in factors[1:]:
        out = multiply(out, F)
    return out

def prefix_products(factors):
    """Boundary accumulations Lambda_j = F_1 ... F_j, with Lambda_0 = I."""
    d = len(factors[0])
    lambdas = [[[complex(i == j) for j in range(d)] for i in range(d)]]
    for F in factors:
        lambdas.append(multiply(lambdas[-1], F))
    return lambdas

def reduce_chain(factors):
    """Boundary reduction: remove identity factors and cancel adjacent
    pairs whose product is exactly the identity (direct boundary join).

    Returns (reduced_factors, {'cancelled_pairs': c, 'removed_identities': r}).
    """
    factors = list(factors)
    removed = 0
    kept = []
    for F in factors:
        if _is_identity(F): removed += 1
        else: kept.append(F)
    factors = kept
    cancelled = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(factors) - 1):
            a, b = factors[i], factors[i + 1]
            if len(a[0]) == len(b) and _is_identity(multiply(a, b)):
                del factors[i:i + 2]
                cancelled += 1
                changed = True
                break
    return factors, {'cancelled_pairs': cancelled, 'removed_identities': removed}

# --- Boundary formula over Q(i) ----------------------------------------------
def _inverse_exact(M):
    """Inverse of a square rational matrix (Gauss-Jordan over Q(i))."""
    M = [[_rat(z) for z in row] for row in M]
    d = len(M)
    aug = [row + [(Fraction(1), Fraction(0)) if i == j else (Fraction(0), Fraction(0))
                  for j in range(d)] for i, row in enumerate(M)]
    from .nullspace import _rref
    rows, pivots = _rref(aug, d)
    if pivots != list(range(d)):
        raise LinkError('Boundary accumulation is not invertible')
    return [row[d:] for row in rows]

def segment_via_boundary(factors, a, b):
    """Product F_a ... F_b (1-based, inclusive) from boundaries only,
    computed exactly over Q(i).

    Orientation note: with this codebase's row-vector convention,
    Lambda_b = Lambda_{a-1} @ S for the segment S, so S = Lambda_{a-1}^{-1}
    @ Lambda_b. (In function-composition notation the same identity reads
    S = Lambda_b o Lambda_{a-1}^{-1}; the inverse changes side with the
    convention.)
    """
    lambdas = prefix_products(factors)
    Lb = [[_rat(z) for z in row] for row in lambdas[b]]
    inv = _inverse_exact(lambdas[a - 1])
    out = []
    for i in range(len(inv)):
        row = []
        for j in range(len(Lb[0])):
            acc = (Fraction(0), Fraction(0))
            for k in range(len(Lb)):
                acc = _add(acc, _mul(inv[i][k], Lb[k][j]))
            row.append(_float(acc))
        out.append(row)
    return out

# --- Program synthesis ---------------------------------------------------------
def _fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)

def synthesize_program(program):
    """Reduce a pure compose-chain program to an equivalent program with a
    single matrix (or identity) register, via boundary reduction.

    The equivalent program is produced by compiling generated source, so
    operator-Link certificates stay consistent by construction.
    Currently supports real-valued reduced products.
    """
    from .compiler import compile_source, optimize
    factors = compose_factors(program)
    reduced, _stats = reduce_chain(factors)
    m, n = program['result_shape']
    if not reduced:
        if m != n: raise LinkError('Chain reduced to identity but shape is not square')
        src = f'fn main() -> Link<C[{m}], C[{n}]> {{ return identity::<C[{m}]>(); }}'
        return optimize(compile_source(src))
    # Reduction preserves the total product, and the product is computed
    # only after cancellation -- that is where the exactness is rescued.
    F = reduced[0] if len(reduced) == 1 else chain_product(reduced)
    if any(z.imag != 0 for row in F for z in row):
        raise LinkError('Synthesis currently supports real-valued reduced products')
    rows = ', '.join('[' + ', '.join(_fmt(z.real) for z in row) + ']' for row in F)
    src = (f'fn main() -> Link<C[{m}], C[{n}]> '
           f'{{ return matrix::<C[{m}], C[{n}]>([{rows}]); }}')
    return optimize(compile_source(src))

def boundary_reduce(program):
    """Full report: factors, reduction stats, and (when possible) the
    synthesized equivalent program."""
    factors = compose_factors(program)
    reduced, stats = reduce_chain(factors)
    report = {'factors': len(factors), 'reduced_factors': len(reduced), **stats}
    try:
        report['program'] = synthesize_program(program)
    except LinkError:
        report['program'] = None
    return report
