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

## Status

This is an experimental compiler prototype, not yet a general implementation of crosstalk, carry, residual projection, or arbitrary interval compression. Rust source generation is tested; the development environment used for v0.3.1 did not contain `rustc`, so genuine Rust-native execution still needs CI/host verification. The optional C/GCC fallback remains in the local prototype and is not included in this pull request.

See `docs/` for the mathematical and compiler architecture.
