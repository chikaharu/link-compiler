# Link Compiler v0.3.1

Rust-like block/type syntax, Perl-style `$` variables and `†` adjoint. **Rows represent input, columns represent output.** `A @ B` means apply B, then A; stored matrix multiplication is `B.matrix × A.matrix`.

Requires Python >=3.10, no runtime dependencies.

```bash
cd linkc
python -m linkc examples/identity.link -o examples/identity.lbc --dump-lir
python -m linkc vm examples/identity.lbc
python -m unittest discover -s tests -v
```

Optional installation: `pip install -e .` to obtain `linkc` and `linkvm` commands.

Supported constructors and operations:
- `identity::<C[n]>()`, `matrix::<C[m], C[n]>([[...], ...])`
- `phase2(theta)`: nearest of {1,i,-1,-i} (radians, ties round upward)
- `†`, `@`, `+`, `-`, explicit `Link<C[m], C[n]>` annotations

A compiler-created identity is the only algebraically general simplification. Phase2 operations are exact on their four-state alphabet. The compiler never assumes a user matrix is unitary. This implementation does not claim generic crosstalk, carry, residual projection, or arbitrary interval compression; these require independently specified semantics and proof obligations.

`.lbc` is inspectable versioned JSON bytecode, not Java JVM `.class` format. VM is a register machine with shape checks. The compiler's `--dump-lir` prints typed registers before and after dead-code elimination.

## v0.2: language ↔ bytecode operator Links

The compiler now uses two **typed sparse operator-incidence Links**, with input
symbols as rows and output symbols as columns:

`LinkSourceOp -> TypedOperation -> LinkVMOpcode`

For instance `@ -> compose -> compose` and `† -> adjoint -> adjoint`.
The intermediate type must match for the Links to compose. The composed map
is an inspectable sparse 0/1 matrix (see `linkc/bridge.py`); lowering executes
through these maps instead of directly naming VM opcodes. Each live bytecode
register carries a verifiable source/semantic/opcode/shape correspondence.
Dead-code elimination preserves and renumbers its live correspondence.

```sh
python -m linkc examples/rectangular.link -o examples/rectangular.lbc --dump-links
python -m linkc links examples/rectangular.lbc
python -m linkc vm examples/rectangular.lbc
python -m unittest discover -s tests -v
```

**Scope:** the Link here maps *operator symbols*, not arbitrary program source
strings or their runtime matrices. It is a typed incidence linear operator,
not an assertion that all programming languages or arbitrary optimized programs
can be reconstructed from bytecode. Because optimization discards source
operations, reverse lookup is only defined for the retained instructions.
The original expression-level typing and numeric semantics remain enforced by
the compiler and VM; a metadata certificate alone is not a safety proof.

## v0.3: Native Rust output Link

The new third Link maps validated LinkVM opcode symbols to concrete Rust
operations: `SourceOp -> TypedOperation -> LinkVMOpcode -> RustLowering`.
This is a sparse symbolic operator mapping, **not** a theorem that arbitrary
programs are linear operators. The typed bytecode validator independently
checks operand dimensions and supported operations before code generation.

```sh
# From Rust/Perl-style .link source, emit both .lbc and dependency-free .rs:
python -m linkc examples/rectangular.link \
  -o examples/rectangular.lbc --emit-rust examples/rectangular.rs

# From an existing .lbc, generate Rust source:
python -m linkc rust examples/rectangular.lbc --emit-rust examples/rectangular.rs

# When rustc is installed, build a standalone native executable directly:
python -m linkc rust examples/rectangular.lbc \
  --emit-rust examples/rectangular.rs --rust-bin examples/rectangular
./examples/rectangular

# Or build an emitted .rs file manually (no external crates):
rustc -C opt-level=2 examples/rectangular.rs -o examples/rectangular
```

Output is JSON with `shape` and `matrix`, identical in format to the Python
LinkVM. Currently supported operations: identity, matrix literals (including
complex numbers), phase2, adjoint, composition, addition and subtraction.
The convention remains **rows=input, columns=output**, so `A @ B` produces
numeric `B*A`.

The generated Rust is independent of the Python compiler at execution time.
It uses a simple dependency-free `Complex` and `Matrix` runtime embedded in
the generated source. This first backend uses dense matrices, not BLAS, SIMD,
or Link-specific native optimization. The compiler already folds identities,
constant 2bit phase compositions and dead registers before Rust lowering.

Run tests with `python -m unittest discover -s tests -v`.  The native-binary
parity test is automatically skipped if `rustc` is unavailable. Rust source
emission and typed validation are tested without Rust installed. `--rust-bin`
reports a nonzero exit and keeps `.rs` if `rustc` is unavailable.

## Native binary delivery (v0.3.1)

A genuine Rust executable is built from the emitted `.rs` source with
`./build_native.sh` or `python -m linkc rust examples/rectangular.lbc --rust-bin examples/rectangular-rust` **when rustc is installed**.

The build environment used for this delivery has no `rustc`, and an attempted
package installation could not complete. Therefore the bundled working Linux
x86-64 executables in `native-linux-x86_64/` were compiled with **GCC from
independently generated C** using the same validated Link bytecode, NOT Rust.
They are supplied as a verified native-execution alternative; do not treat them
as proof that the Rust backend has compiled successfully.

Rebuild C fallback for the current host (requires GCC):

```sh
python - <<'PY'
import json
from pathlib import Path
from linkc.c_native_backend import compile_c
for name in ('identity', 'rectangular', 'phases'):
    p=json.loads(Path(f'examples/{name}.lbc').read_text())
    compile_c(p, Path(f'native-linux-x86_64/{name}-c'))
PY
```

The native binaries output a JSON `shape` and `matrix` result in LinkVM format.
The generated Rust source is standalone and does not require third-party Rust
crates, but genuine Rust-native verification remains outstanding on a host
with `rustc`.