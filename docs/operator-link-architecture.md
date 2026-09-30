# Typed operator-Links between source language and bytecode

## What is a Link here?
Let \(S,M,B\) be finite-dimensional vector spaces whose distinguished bases
are respectively source-language operator forms, type-checked mathematical
operations, and LinkVM opcodes. These are **operator-symbol spaces**, not the
numeric spaces acted on by a runtime Link.

Two sparse incidence operators, with **input rows and output columns**, connect
them:

\[
L_{SM}: S \longrightarrow M,\quad
L_{MB}: M \longrightarrow B,\quad
L_{SB}=L_{SM} L_{MB}.
\]

In the project's row-oriented representation, a basis vector for the source
operator `@` is mapped first to `compose` and then to the same opcode. A
validated type match at the intermediate space permits Link composition.
Every currently supported source operator has one outgoing edge. This is a
restricted deterministic operator map, not a generic or invertible compiler.

The emitted register bytecode also includes `operator_links` recording the
surviving source-form, semantic operation, opcode and Link value shape. The VM
rejects inconsistent certificates. This does not replace runtime checks or
prove general semantic equivalence by itself.

## Compile and inspect

```shell
python -m linkc examples/rectangular.link --dump-links -o /tmp/example.lbc
python -m linkc links /tmp/example.lbc
python -m linkc vm /tmp/example.lbc
```

## Constraints and future extension

- Compiler folds some identities and 2-bit phase compositions before emitting
  instructions. Folded and dead source expressions are intentionally absent
  from the **live-register** certificate; this is not lossless decompilation.
- For a true reversible source-to-bytecode correspondence, retain the source
  AST and optimizer rewrites in a separate provenance graph, and define a
  restricted inverse on an explicitly selected normal form.
- A further extension would lift control-flow structures and whole typed
  expression graphs, not just individual operator symbols, to a typed
  graph-rewrite language. Proof obligations would cover type preservation and
  evaluation equivalence for every rewrite.