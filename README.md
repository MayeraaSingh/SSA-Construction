# Implementation of Static Single Assignment (SSA) Construction Using the Lazy, Sealed-Block Algorithm

## Project Description

This project implements Static Single Assignment (SSA) form construction for a
simple imperative language. SSA is an intermediate representation in which every
variable is assigned exactly once, with phi (`φ`) functions inserted at control
flow merge points to reconcile multiple reaching definitions. The implementation
uses the lazy, sealed-block algorithm, which defers phi-function insertion until
a block can no longer receive new predecessors, keeping construction simple and
efficient.

The work is based on the research paper:

> **Simple and Efficient Construction of Static Single Assignment Form**
> Ron Cytron, Jeanne Ferrante, Barry K. Rosen, Mark N. Wegman, and F. Kenneth Zadeck
> *(with the lazy variant described by Braun et al.)*

## Project Objectives

- Build a working front end (lexer and parser) for a small imperative language.
- Construct a Control Flow Graph (CFG) from the parsed abstract syntax tree.
- Implement SSA construction using the lazy, sealed-block algorithm.
- Insert and manage phi functions at merge points.
- Eliminate trivial phi functions through simplification.
- Visualize the CFG and the resulting SSA form (text-based and graphical).

## Planned Core Features

- **Lexer and parser** — tokenize and parse source code into an AST.
- **Control Flow Graph construction** — build the CFG from the AST.
- **Lazy SSA construction** — place definitions and phi functions on demand.
- **Phi-function insertion** — insert `φ` functions at control flow merges.
- **Sealed-block handling** — seal blocks once all predecessors are known.
- **Trivial phi-function elimination** — remove `φ` functions with identical
  operands.
- **CFG visualization** — render the CFG textually and graphically.

## Technology Stack

- **Language:** Python 3.11+
- **Testing:** pytest
- **Graph libraries:** NetworkX, Matplotlib
- **Visualization:** Text-based output and graphical CFG rendering

## Project Structure

```text
ssa-construction/
├── src/
│   ├── frontend/        # Lexer, parser, and AST nodes
│   ├── ir/              # Instructions, basic blocks, and CFG
│   ├── ssa/             # SSA builder, phi functions, simplifier, dominance frontier
│   ├── visualization/   # CFG visualizer
│   └── main.py          # Command-line entry point
├── examples/            # Example source programs (.c)
├── tests/               # pytest test suite
├── results/             # Generated output (visualizations, dumps)
├── .gitignore
├── requirements.txt
└── README.md
```

## Development Status

**Day 3 complete** — SSA correctness hardening, validation strengthening, and
semantic equivalence testing implemented.  The implementation has been
systematically tested for phi-function correctness, trivial phi cascading
simplification, alias propagation, complex control flow, and semantic
equivalence between source-level and SSA-level evaluation.  All 185 tests
pass (86 Day 1 + 50 Day 2 + 49 Day 3).

## Day 3 — Correctness Hardening, Validation & Semantic Testing

### Additional Correctness Checks

- **Phi operand-to-predecessor mapping**: verified that each phi operand
  carries the value from its associated predecessor block's outgoing state,
  not merely the correct count.
- **Incomplete phi completion**: verified that incomplete phis created in
  unsealed blocks (loop headers) are correctly resolved with one operand per
  predecessor upon sealing.
- **Cascading trivial phi elimination**: verified that eliminating one phi can
  make dependent phis trivial, resolved by the `_finalize()` fixpoint pass.
- **Alias chain resolution**: verified that multi-level alias chains resolve
  completely, that cycles are safely handled, and that `_finalize()` is
  idempotent.

### Edge Cases Tested

- Nested if-else with three-way merge.
- Nested loops with both loop-carried variables.
- Loop containing an if-else where a variable is modified in only one branch.
- Variable used after a merge but not assigned in one branch (empty branch).
- Variable modified before and after a loop (post-loop uses post-loop value,
  not the loop header phi).
- Multiple variables requiring phi functions at the same merge block.
- Sequential if-else blocks with cascading trivial phi elimination.
- Empty true/false branches.

### SSA Validation

`validate_ssa()` now checks:

1. Every assignment result and non-eliminated phi result is unique.
2. Every SSA value use resolves (through aliases) to a valid definition.
3. Every non-eliminated phi has exactly one operand per predecessor.
4. Phi operand block IDs are valid predecessors of the phi's block.
5. No incomplete (empty-operand) phis remain in sealed blocks.
6. No eliminated phi result is directly referenced in a non-eliminated phi's
   operands (must be resolved through aliases).
7. All alias targets resolve to defined values.
8. All blocks are sealed.
9. Phi `block_id` matches the block storing it.
10. All branch conditions use valid SSA values.

Error messages include the block ID, phi result name, and source variable name
for actionable debugging.

### Semantic Equivalence Testing

A source-level interpreter (`SourceInterpreter`) executes the AST directly,
maintaining a `{variable: int}` environment.  An SSA-level evaluator
(`SSAEvaluator`) executes the CFG/SSA IR by following edges, evaluating phi
functions using the actual predecessor block ID, and evaluating SSA
instructions using an SSA value store.

Both interpreters are compared on 17 test programs covering:

- Straight-line assignments and arithmetic precedence.
- Conditionals (true branch, false branch, unmodified variables).
- Loops (simple, accumulation, unmodified, modified before/after).
- Nested loops.
- If-else inside loops.
- Variables modified in one branch of a loop.
- Complex nested programs with loops and conditionals.
- All six relational operators.
- Unary minus.

All 17 semantic tests pass, confirming that the SSA construction preserves the
original program's behaviour.

### Known Limitations

- **Variables defined only inside a loop body**: if a variable is assigned
  only inside a loop body and used after the loop, the SSA builder raises an
  undefined-variable error.  This is correct behaviour — the variable may be
  undefined if the loop never executes.  The source interpreter does not
  detect this because it only runs the actual execution path.  Test programs
  that use variables after loops always define them before the loop.
- **No constant folding**: trivial phi elimination only checks SSA value
  identity, not value equality.  `phi(x_1=1, x_2=1)` is not eliminated because
  x_1 and x_2 are different SSA values, even though they hold the same value.
  This matches the paper's algorithm.
- **Post-construction alias propagation**: the `_finalize()` pass uses a
  fixpoint iteration to propagate aliases after construction.  The paper
  describes doing this eagerly during construction.  Our approach is simpler
  and produces identical results; the deviation is justified by the idempotency
  tests confirming correctness.

### Deviations from the Paper

- **Post-construction `_finalize()`**: the paper performs trivial phi
  elimination during construction and immediately updates dependent phis.  Our
  implementation performs elimination during construction AND a post-pass
  fixpoint iteration.  This handles cascading simplification that the paper
  handles by tracking phi users.  The post-pass is simpler and produces the
  same result, as verified by the idempotency tests.
