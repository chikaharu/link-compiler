# link-compiler

Experimental typed Link compiler with Rust-like block/type syntax, Perl-style `$` variables, typed operator Links, inspectable JSON bytecode, a Python LinkVM, and native Rust/C lowering.

**Convention:** rows are inputs and columns are outputs. `A @ B` applies B, then A.

## Pipeline

```text
.link source
  -> typed Link IR
  -> operator Link: SourceOp -> TypedOperation
  -> Link bytecode (.lbc)
  -> backend Link
       -> Python LinkVM
       -> standalone Rust source / rustc native binary
```

The operator Links are sparse incidence maps between representation spaces. They do not assert that arbitrary programs are linear operators.

## Quick start

Requires Python 3.10+.

```bash
python -m linkc examples/identity.link -o identity.lbc --dump-lir
python -m linkc vm identity.lbc
python -m unittest discover -s tests -v
```

Rust lowering:

```bash
python -m linkc rust examples/rectangular.lbc --emit-rust rectangular.rs
rustc -C opt-level=2 rectangular.rs -o rectangular
```

The generated Rust has no external crate dependency.

## Implemented subset

- `identity::<C[n]>()`
- `matrix::<C[m], C[n]>(...)`
- `phase2(theta)`
- adjoint `†`
- composition `@`
- `+` and `-`
- static `Link<C[m], C[n]>` shape checking
- typed source -> semantic -> bytecode operator-Link certificates
- identity / exact 2-bit phase simplification and dead-register elimination
- Python LinkVM
- Rust source/native backend

## Integration: boundary reduction and null-space carry

This branch (`integration/boundary-nullspace`, based on v0.3.1) integrates
the two mechanisms that the 2026-10-24 session identified as the core:
**boundary reduction** and the **null space**. Both are dependency-free
library modules over the existing bytecode; neither changes the bytecode
format.

- `linkc/boundary.py`
  - Stream codec: boundary form (initial value + switch-point list);
    extract = diff, restore = cumsum, exact roundtrip.
  - Boundary reduction of compose chains: prefix products (boundary
    accumulations) determine any segment from two boundaries,
    `F_a ... F_b = Lambda_{a-1}^{-1} @ Lambda_b` in this codebase's
    row-vector convention, verified exactly over Q(i). Adjacent pairs
    whose product is *exactly* the identity cancel at the seam; a
    reduced chain can be synthesized back into an equivalent
    single-register program (compiled from generated source, so
    operator-Link certificates stay consistent).
- `linkc/nullspace.py`
  - Null-space carry for a linear Link `A`: exact kernel basis over the
    Gaussian rationals Q(i) (Fraction pairs), orthogonal decomposition
    `x = x_range + x_kernel`, application from the range part only, and
    exact restoration. The ledger closes exactly: `x_kernel @ A == 0`
    and `restore == x` are rational identities, not approximations.
    The kernel is a recycle box, not a trash bin.

Tests: `tests/test_boundary.py`, `tests/test_nullspace.py`
(15 tests total in the suite, all passing).

Host verification note: on a host with `rustc` 1.99.1, the Rust native
backend was built and executed for `examples/rectangular.link`; its
output matches the LinkVM result exactly. The v0.3.1 caveat below about
missing host verification is resolved on such hosts.

Still open (from the session's hypothesis ledger): wiring seam
cancellation into `optimize()` as an automatic pass, affine (not only
linear) chains, and float (non-exact) kernel tolerance policy.

## Status

This is an experimental compiler prototype, not yet a general implementation of crosstalk, carry, residual projection, or arbitrary interval compression. Rust source generation is tested; the development environment used for v0.3.1 did not contain `rustc`, so genuine Rust-native execution still needs CI/host verification. The optional C/GCC fallback remains in the local prototype and is not included in this pull request.

See `docs/` for the mathematical and compiler architecture.
