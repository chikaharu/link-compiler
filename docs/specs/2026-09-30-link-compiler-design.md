# Link compiler v0.1 design

Source: Rust-style `fn`, `let`, type annotations and braces; Perl-style `$` sigils; adjoint `†`; composition `@`. Matrix storage uses **input rows, output columns**: `Link<C[m],C[n]>` stores an m-by-n matrix and acts on input row vectors from the right. `A @ B` means A after B, with `B.matrix × A.matrix`.

Pipeline: tokenizer -> recursive-descent parser -> typed register LIR -> conservative optimization -> JSON `.lbc` bytecode -> dependency-free Python LinkVM. First release uses a single `fn main()` and eager values. Parser accepts `identity::<C[n]>()`, `matrix::<C[m],C[n]>([[...],...])`, `phase2(theta)`, adjoint `†`, composition `@`, and additions/subtractions on matching Link types. Link metadata describing unitarity is only attached to compiler-created identity and phase2 links. Optimizer folds adjoints of identity, compositions with identity, and adjacent phase2 states as arithmetic mod 4 without dropping operand effects. User matrices receive no unitary assumption.

The emitted JSON bytecode is a versioned register program; both `linkc` and `linkvm` use it. Runtime performs dimensional checks again, rejects malformed bytecode, and prints complex results in JSON `{re,im}` form. No arbitrary user code execution or Python eval. This is a small subset, not a proof or implementation of generalized crosstalk/carry semantics.
