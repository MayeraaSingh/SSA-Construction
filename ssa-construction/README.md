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

**Initial scaffolding completed; implementation not yet started.**
