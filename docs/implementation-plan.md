# v0.1 implementation plan
1. Lexer/parser: define source AST for typed `main`, `let`, `return`, matrix/identity/phase2 constructors, `†`, `@`, `+`, `-`; reject unknown tokens and malformed syntax with line/column diagnostics. Test parsing and syntax rejection.
2. Typed lowering: enforce `Link<C[m],C[n]>`; encode input-rows/output-columns composition (`A @ B` means A after B); emit register LIR and fail dimension mismatch; test non-square operator composition.
3. Optimization and bytecode: eliminate identity/adjoint identity and fold phase2 modular compositions; generate `.lbc` with format/version; ensure `--dump-lir` before/after is inspectable. Test no unsound unitary assumptions.
4. VM and CLI: run bytecode without third-party dependencies; JSON complex output; reject malformed opcode, invalid shape and version; integrate `linkc`, `linkvm`, examples and unittest suite. Exit 0 on success, 1 on compile/runtime errors, 2 on CLI usage errors; explicit output path can be overwritten.
