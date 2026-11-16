# Implementation of Static Single Assignment (SSA) Construction Using the Lazy, Sealed-Block Algorithm

A complete Python implementation of SSA form construction for a simple imperative language,
based on the lazy sealed-block algorithm from:

> Braun, M., Buchwald, S., Hack, S., Leißa, R., Mallon, C., & Zwinkau, A. (2013).
> **Simple and Efficient Construction of Static Single Assignment Form.**
> *Compiler Construction (CC 2013)*, Lecture Notes in Computer Science, vol. 7791.
> Springer, Berlin, Heidelberg.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Algorithm](#algorithm)
3. [Architecture](#architecture)
4. [Installation and Execution](#installation-and-execution)
5. [Demonstration](#demonstration)
6. [Testing and Results](#testing-and-results)
7. [Known Limitations](#known-limitations)
8. [Project Structure](#project-structure)
9. [References](#references)

---

## Project Overview

**What is Static Single Assignment form?**

Static Single Assignment (SSA) form is an intermediate representation (IR) used in
optimising compilers. In SSA form, every variable is assigned exactly once. When a
variable is assigned multiple times in the source program, each assignment produces a
uniquely versioned SSA value (e.g. `x_1`, `x_2`, `x_3`).

**Why are phi functions required?**

At control-flow merge points — such as after an if-else statement or at a loop header —
multiple versions of the same variable may flow in from different predecessors. A phi
function (`phi(x_1 [B1], x_2 [B2])`) selects the correct version based on which
predecessor block was actually executed at runtime. The phi function is a formal device,
not executable code; a backend would implement it via register allocation or
copy-insertion.

**What does the lazy sealed-block algorithm achieve?**

Conventional SSA construction algorithms (e.g. Cytron et al.) compute dominance frontiers
over the whole CFG before inserting phi functions. The lazy sealed-block algorithm,
described by Braun et al., avoids this two-pass structure entirely. It inserts phi
functions on demand as variables are read, and defers phi operand resolution until a
block is *sealed* — that is, until all its predecessor edges are known. This makes SSA
construction incremental and naturally handles forward references without requiring a
pre-computed dominator tree.

---

## Algorithm

The core algorithm is implemented in `src/ssa/ssa_builder.py` as the `SSABuilder` class.

### Recursive variable lookup — `read_variable(name, block)`

This is the central operation. When a variable is read in a block:

1. **Local definition exists** → return the most recent SSA version for that variable in
   that block.
2. **Block is unsealed** (not all predecessors known yet, typical of loop headers) →
   create an *incomplete* phi function. Record its result as the current definition
   immediately. Return the phi result. Operands will be filled in later.
3. **Sealed, single predecessor** → recursively look up the variable in the predecessor,
   cache the result locally, return it.
4. **Sealed, multiple predecessors** → create a phi, record its result *before* reading
   predecessor operands (this breaks cyclic recursion on loop backedges), read the
   variable from each predecessor, populate operands, attempt trivial phi elimination,
   return the result.
5. **Sealed, no predecessors, no definition** → raise an undefined-variable error.

### Incomplete phi functions

When a variable is read from an unsealed block (step 2 above), the phi is created with
empty operands. The phi result is recorded as the block's current definition so
subsequent reads return it without creating a second phi.

### Sealed blocks — `seal_block(block)`

A block is sealed once all its CFG predecessors have been registered. For if-else
constructs, the merge block is sealed after both branches are built. For while loops,
the header is sealed *after* the body is built and the backedge has been added. Sealing
triggers resolution of all incomplete phis: for each incomplete phi, the variable is
read from every predecessor and the operands are populated.

### Trivial phi elimination — `try_remove_trivial_phi(phi)`

A phi is trivial when every non-self operand resolves (through the alias map) to the
same SSA value. For example:

- `phi(x_1 [B0], x_1 [B2])` → trivial, replaced by `x_1`
- `phi(x_2 [B0], x_2 [B2])` (self-reference in a loop where the variable is unmodified)
  → trivial, replaced by the entry operand
- `phi(x_2 [B1], x_3 [B2])` → non-trivial, kept

When a phi is eliminated, its result is aliased to the surviving value in the builder's
alias map. All future lookups resolve through this map.

### Alias resolution and finalization — `_finalize()`

After construction completes, a fixpoint pass iterates over all non-eliminated phis,
resolves their operands through the current alias map, and re-checks triviality. This
handles cascading simplification: eliminating one phi may make a dependent phi trivial.
The pass repeats until no more changes occur. It is idempotent: running it twice
produces the same result.

---

## Architecture

```
Source program (.c file)
        │
        ▼
  ┌─────────────┐
  │    Lexer    │  src/frontend/lexer.py
  │             │  Character-by-character tokenizer
  └──────┬──────┘
         │  Token stream
         ▼
  ┌─────────────┐
  │   Parser    │  src/frontend/parser.py
  │             │  Recursive-descent parser
  └──────┬──────┘
         │  AST (Program, Assignment, IfStatement, WhileStatement, ...)
         ▼
  ┌─────────────────────────┐
  │     CFG Construction    │  src/ir/cfg_builder.py
  │  (Day 1 — pre-SSA CFG)  │  Builds BasicBlock graph with Day 1 IR instructions
  └───────────┬─────────────┘
              │  ControlFlowGraph
              ▼
  ┌─────────────────────────┐
  │   Lazy SSA Construction │  src/ssa/ssa_builder.py
  │   (sealed-block algo)   │  read_variable / write_variable / seal_block
  │                         │  add_phi / try_remove_trivial_phi / _finalize
  └───────────┬─────────────┘
              │  SSA-form CFG (BasicBlock with ssa_instructions + phis)
              ▼
  ┌─────────────────────────┐
  │    SSA Validation       │  validate_ssa() — 10 structural checks
  └───────────┬─────────────┘
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
  Textual output   Graphical output
  (to console)     src/visualization/cfg_visualizer.py
                   PNG files via NetworkX + Matplotlib
```

---

## Installation and Execution

**Requirements:** Python 3.11+

### Install dependencies

```powershell
pip install -r requirements.txt
```

`requirements.txt` includes: `pytest`, `networkx`, `matplotlib`

### Run the pipeline (textual output only)

```powershell
python -m src.main examples/conditional.c
```

This prints the AST, original CFG, SSA IR, and SSA validation result.

### Run with graphical visualization

```powershell
python -m src.main examples/conditional.c --visualize
```

Saves `outputs/conditional_cfg.png` and `outputs/conditional_ssa.png`.

### Specify a custom output directory

```powershell
python -m src.main examples/while_loop.c --visualize --output-dir my_outputs/
```

### Generate only the original CFG diagram

```powershell
python -m src.main examples/conditional.c --visualize --cfg-only
```

### Generate only the SSA CFG diagram

```powershell
python -m src.main examples/conditional.c --visualize --ssa-only
```

### Run the full test suite

```powershell
python -m pytest tests/ -v
```

### Generate diagrams for all four examples at once

```powershell
foreach ($f in @("assignments.c","conditional.c","nested_if.c","while_loop.c")) {
    python -m src.main "examples/$f" --visualize --output-dir outputs
}
```

---

## Demonstration

### Input program — `examples/conditional.c`

```c
x = 10;

if (x > 5) {
    y = 20;
} else {
    y = 30;
}

z = y + 5;
```

### Original CFG (textual)

```
--- Block 0 ---   ASSIGN x = 10          BRANCH if (x > 5) -> B1 else -> B2
--- Block 1 ---   ASSIGN y = 20          JUMP -> B3
--- Block 2 ---   ASSIGN y = 30          JUMP -> B3
--- Block 3 ---   ASSIGN z = (y + 5)     JUMP -> B4
--- Block 4 ---   (exit — no instructions)
```

### SSA IR (textual)

```
--- Block 0 ---   x_1 = 10
                  BRANCH if (x_1 > 5) -> B1 else -> B2

--- Block 1 ---   y_1 = 20
                  JUMP -> B3

--- Block 2 ---   y_2 = 30
                  JUMP -> B3

--- Block 3 ---   y_3 = phi(y_1 [B1], y_2 [B2])
                  z_1 = (y_3 + 5)
                  JUMP -> B4

--- Block 4 ---   (empty)
```

### Explanation of phi functions

`y_3 = phi(y_1 [B1], y_2 [B2])` at the merge block (Block 3) reconciles the two
definitions of `y`: `y_1 = 20` from the true branch and `y_2 = 30` from the false
branch. The subsequent `z_1 = (y_3 + 5)` correctly uses the merged value regardless
of which branch was taken.

`x` is not modified in either branch, so no phi is inserted for it. This is the
*lazy* property of the algorithm: phi functions are only created when a variable is
actually read at a merge point with differing incoming definitions.

### Generated SSA CFG diagram

Diagram saved at `outputs/conditional_ssa.png`:

- Block 0 (green = entry): contains `x_1 = 10` and the conditional branch.
- Blocks 1 and 2 (blue): true and false branches with `y_1` and `y_2` assignments.
- Block 3 (blue): merge block with the `phi` function and `z_1` computation.
- Block 4 (coral = exit): empty exit block.
- Edges labelled "True" and "False" indicate the conditional branch directions.

### While-loop SSA — `examples/while_loop.c`

```c
x = 0;
while (x < 5) { x = x + 1; }
y = x;
```

SSA output:
```
Block 0:  x_1 = 0           JUMP -> B1
Block 1:  x_2 = phi(x_1 [B0], x_3 [B2])
          BRANCH if (x_2 < 5) -> B2 else -> B3
Block 2:  x_3 = (x_2 + 1)   JUMP -> B1   [backedge]
Block 3:  y_1 = x_2          JUMP -> B4
Block 4:  (exit)
```

The phi function `x_2 = phi(x_1 [B0], x_3 [B2])` at the loop header merges the initial
value `x_1 = 0` with the loop-body increment `x_3 = x_2 + 1`. The loop condition and
body both correctly use `x_2`. Post-loop, `y_1 = x_2` also uses the phi result.

The backedge (Block 2 → Block 1) is rendered as a dashed red arrow in the generated
diagram `outputs/while_loop_ssa.png`.

---

## Testing and Results

### Test count by day

| Day | Test file(s) | Tests | Coverage |
|-----|-------------|-------|----------|
| Day 1 | `test_parser.py`, `test_cfg.py` | 86 | Lexer, parser, AST, CFG construction, edge relationships, validation |
| Day 2 | `test_ssa.py`, `test_loops.py`, `test_phi.py` | 50 | Straight-line SSA, conditionals, loops, phi functions, incomplete phis |
| Day 3 | `test_hardening.py`, `test_semantic.py` | 49 | Phi operand correctness, cascading elimination, alias propagation, semantic equivalence |
| Day 4 | `test_visualization.py` | 13 | PNG generation, data immutability, complex structures, all examples |
| **Total** | | **198** | |

### Final test run

```
198 passed in ~11s
```

All 185 previously passing tests (Days 1–3) continue to pass. No test was weakened or
deleted.

### Semantic equivalence testing

A `SourceInterpreter` executes the AST directly. An `SSAEvaluator` executes the SSA
CFG by following edges, evaluating phi functions using the actual predecessor block ID.
Both are compared on 22 test programs including:

- Straight-line code and arithmetic precedence
- Conditionals (true/false paths, unmodified variables, multiple variables at merge)
- Simple loops, accumulating loops, nested loops
- If-else inside loops
- All 6 relational operators (`<`, `>`, `<=`, `>=`, `==`, `!=`)
- Unary minus
- Complex nested programs

All 22 semantic equivalence tests pass, confirming that the SSA construction preserves
the original program's behaviour.

---

## Known Limitations

| Limitation | Detail |
|-----------|--------|
| Variables defined only inside a loop body | If a variable is first assigned inside a loop and read after the loop, the SSA builder raises an undefined-variable error. This is correct: the variable is genuinely undefined if the loop never executes. Programs in this repository always define such variables before the loop. |
| No constant folding | Trivial phi elimination checks SSA value *identity*, not value *equality*. `phi(x_1=1, x_2=1)` is not eliminated because `x_1` and `x_2` are different SSA versions. This matches the paper's algorithm. |
| Explicit else required | The parser requires an explicit `else` block for every `if`. Optional-else is not supported. |
| Arithmetic operators | Only `+`, `-`, `*` are supported. Division and bitwise operators are not in the grammar. |
| No dominance-frontier baseline | `src/ssa/dominance_frontier.py` is a placeholder. Dominance-frontier-based SSA construction is outside the scope of this project. |
| No graphical comparison | Dominance-frontier comparison visualization is not implemented. |
| Post-construction `_finalize()` | The paper performs trivial phi elimination eagerly during construction. This implementation also runs a post-construction fixpoint pass for cascading simplification. The two approaches produce identical results (verified by idempotency tests). |

---

## Project Structure

```text
SSA Construction/
├── src/
│   ├── __init__.py
│   ├── main.py                      # CLI entry point (argparse, pipeline)
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── lexer.py                 # Hand-written character-by-character lexer
│   │   ├── parser.py                # Recursive-descent parser
│   │   └── ast_nodes.py             # AST dataclass nodes + pretty-printer
│   ├── ir/
│   │   ├── __init__.py
│   │   ├── instructions.py          # Day 1 IR instruction types
│   │   ├── basic_block.py           # BasicBlock (Day 1 + SSA lifecycle fields)
│   │   ├── control_flow_graph.py    # CFG data structure, validation, to_text()
│   │   └── cfg_builder.py           # AST → Day 1 CFG transformation
│   ├── ssa/
│   │   ├── __init__.py
│   │   ├── ssa_builder.py           # Core SSA engine: read/write/seal/phi/finalize
│   │   ├── phi_functions.py         # PhiFunction dataclass
│   │   ├── ssa_simplifier.py        # Trivial phi elimination, alias resolution
│   │   └── dominance_frontier.py    # Placeholder (not implemented)
│   └── visualization/
│       ├── __init__.py
│       └── cfg_visualizer.py        # NetworkX + Matplotlib CFG/SSA rendering
├── examples/
│   ├── assignments.c                # Sequential assignments
│   ├── conditional.c                # Simple if-else
│   ├── nested_if.c                  # Nested conditionals
│   └── while_loop.c                 # While loop with backedge
├── tests/
│   ├── __init__.py
│   ├── test_parser.py               # 49 parser tests
│   ├── test_cfg.py                  # 37 CFG tests
│   ├── test_ssa.py                  # 22 SSA construction tests
│   ├── test_loops.py                # 16 loop SSA tests
│   ├── test_phi.py                  # 12 phi function tests
│   ├── test_hardening.py            # 27 correctness hardening tests
│   ├── test_semantic.py             # 22 semantic equivalence tests
│   └── test_visualization.py        # 13 visualization tests
├── outputs/                         # Generated PNG diagrams (git-ignored)
├── results/
│   └── .gitkeep
├── conftest.py                      # Adds project root to sys.path for pytest
├── .gitignore
├── requirements.txt                 # pytest, networkx, matplotlib
└── README.md
```

---

## References

Braun, M., Buchwald, S., Hack, S., Leißa, R., Mallon, C., & Zwinkau, A. (2013).
Simple and Efficient Construction of Static Single Assignment Form.
In *Compiler Construction: 22nd International Conference, CC 2013*,
Lecture Notes in Computer Science, vol. 7791, pp. 102–122.
Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-642-37051-9_6

Cytron, R., Ferrante, J., Rosen, B. K., Wegman, M. N., & Zadeck, F. K. (1991).
Efficiently computing static single assignment form and the control dependence graph.
*ACM Transactions on Programming Languages and Systems (TOPLAS)*, 13(4), 451–490.
https://doi.org/10.1145/115372.115320
