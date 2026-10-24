"""Null-space carry for linear Links (integration of the 2026-10-24 session).

A linear Link is a matrix A (rows = inputs, columns = outputs); a row vector
x maps to x @ A. The (left) kernel ker A = {x : x @ A = 0} is the carry
destination: components of x that land in ker A are not destroyed by A,
they are stored undistorted and can be restored exactly ("recycle box",
not deletion).

All core arithmetic is exact over the Gaussian rationals Q(i), represented
as (Fraction, Fraction) pairs, so the ledger closes exactly: for the
decomposition x = x_range + x_kernel,
    x_kernel @ A == 0   and   restore == x   and   x_range @ A == x @ A
hold as exact rational identities, not approximations.
"""
from fractions import Fraction
from .syntax import LinkError

# --- Gaussian rationals: z = (re, im) of Fractions -------------------------
def _rat(z):
    if isinstance(z, tuple):
        return (Fraction(z[0]), Fraction(z[1]))
    if isinstance(z, Fraction):
        return (z, Fraction(0))
    if isinstance(z, int):
        return (Fraction(z), Fraction(0))
    z = complex(z)
    return (Fraction(z.real), Fraction(z.imag))

def _add(a, b): return (a[0] + b[0], a[1] + b[1])
def _sub(a, b): return (a[0] - b[0], a[1] - b[1])
def _mul(a, b): return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])
def _conj(a): return (a[0], -a[1])
def _div(a, b):
    d = b[0] * b[0] + b[1] * b[1]
    if d == 0: raise LinkError('Division by zero in rational kernel arithmetic')
    n = _mul(a, _conj(b))
    return (n[0] / d, n[1] / d)
def _zero(a): return a[0] == 0 and a[1] == 0
def _float(a): return complex(float(a[0]), float(a[1]))

def _rat_matrix(A): return [[_rat(z) for z in row] for row in A]
def _float_matrix(A): return [[_float(z) for z in row] for row in A]

def _rref(rows, ncols):
    """Reduced row echelon form over Q(i); returns (rows, pivot columns)."""
    rows = [list(r) for r in rows]
    pivots = []
    r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, len(rows)) if not _zero(rows[i][c])), None)
        if piv is None: continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = _div((Fraction(1), Fraction(0)), rows[r][c])
        rows[r] = [_mul(v, inv) for v in rows[r]]
        for i in range(len(rows)):
            if i != r and not _zero(rows[i][c]):
                f = rows[i][c]
                rows[i] = [_sub(v, _mul(f, w)) for v, w in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
        if r == len(rows): break
    return rows[:r], pivots

# --- Kernel -----------------------------------------------------------------
def kernel_basis_exact(A):
    """Exact basis of {x : x @ A = 0} as (Fraction, Fraction) vectors.

    A is m x n (rows = inputs). The conditions on x are A^T x^T = 0, i.e.
    one equation per output column: sum_i x_i * A[i][j] = 0.
    """
    A = _rat_matrix(A)
    m = len(A)
    if m == 0: return []
    n = len(A[0])
    equations = [[A[i][j] for i in range(m)] for j in range(n)]
    rows, pivots = _rref(equations, m)
    free = [c for c in range(m) if c not in pivots]
    basis = []
    for f in free:
        v = [(Fraction(0), Fraction(0))] * m
        v[f] = (Fraction(1), Fraction(0))
        for row, p in zip(rows, pivots):
            v[p] = _mul(row[f], (Fraction(-1), Fraction(0)))
        basis.append(v)
    return basis

def kernel_basis(A):
    """Kernel basis as complex-float vectors (exact values, converted)."""
    return [[_float(z) for z in v] for v in kernel_basis_exact(A)]

def nullity(A):
    return len(kernel_basis_exact(A))

def _apply_exact(A, x):
    """x @ A over Q(i): A is a rational matrix, x a rational vector."""
    n = len(A[0])
    out = []
    for j in range(n):
        acc = (Fraction(0), Fraction(0))
        for i, xi in enumerate(x):
            acc = _add(acc, _mul(xi, A[i][j]))
        out.append(acc)
    return out

def _inner(u, v):
    """Hermitian inner product sum conj(u_i) * v_i over Q(i)."""
    acc = (Fraction(0), Fraction(0))
    for a, b in zip(u, v):
        acc = _add(acc, _mul(_conj(a), b))
    return acc

# --- Carry ------------------------------------------------------------------
def carry_decompose(A, x):
    """Split x into range part + kernel part, exactly over Q(i).

    Returns a dict with complex-float views ('range', 'kernel',
    'kernel_coords', 'kernel_basis') and the exact payload under 'exact'.
    The kernel part is the orthogonal projection of x onto ker A
    (Gram-Schmidt over Q(i)); it is stored, not deleted.
    """
    Ar = _rat_matrix(A)
    xr = [_rat(z) for z in x]
    basis = kernel_basis_exact(A)
    # Gram-Schmidt (unnormalized) over Q(i)
    ortho = []
    for b in basis:
        v = list(b)
        for u in ortho:
            denom = _inner(u, u)
            if _zero(denom): continue
            coef = _div(_inner(u, v), denom)
            v = [_sub(vi, _mul(coef, ui)) for vi, ui in zip(v, u)]
        if any(not _zero(vi) for vi in v):
            ortho.append(v)
    coords = []
    xk = [(Fraction(0), Fraction(0))] * len(xr)
    for u in ortho:
        denom = _inner(u, u)
        coef = _div(_inner(u, xr), denom)
        coords.append(coef)
        xk = [_add(ki, _mul(coef, ui)) for ki, ui in zip(xk, u)]
    x_range = [_sub(a, b) for a, b in zip(xr, xk)]
    decomp = {
        'range': [_float(z) for z in x_range],
        'kernel': [_float(z) for z in xk],
        'kernel_coords': [_float(z) for z in coords],
        'kernel_basis': [[_float(z) for z in u] for u in ortho],
        'exact': {'range': x_range, 'kernel': xk, 'coords': coords, 'ortho': ortho},
    }
    # Ledger: the carried part must be exactly invisible to A.
    if any(not _zero(z) for z in _apply_exact(Ar, xk)):
        raise LinkError('Internal ledger failure: kernel part is not in ker A')
    return decomp

def restore(decomp):
    """Recover x from a decomposition: range part + carried kernel part."""
    return [a + b for a, b in zip(decomp['range'], decomp['kernel'])]

def apply_with_carry(A, x):
    """Apply A while carrying the kernel component aside.

    Returns (y, decomp) with y == x @ A computed from the range part only;
    the kernel component rides along in decomp, undistorted.
    """
    decomp = carry_decompose(A, x)
    Ar = _rat_matrix(A)
    y = [_float(z) for z in _apply_exact(Ar, decomp['exact']['range'])]
    return y, decomp
