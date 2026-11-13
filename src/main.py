"""Command-line entry point for the SSA construction tool.

Drives the full pipeline: source → lexer → parser → AST → CFG → SSA
construction → SSA output → SSA validation.

Usage::

    python -m src.main <source-file>
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.frontend.ast_nodes import Program, pretty_print
from src.frontend.lexer import LexError, Lexer
from src.frontend.parser import ParseError, Parser
from src.ir.cfg_builder import CFGBuilder
from src.ir.control_flow_graph import CFGError
from src.ssa.ssa_builder import SSAError, SSABuilder


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline on a single source file.

    Args:
        argv: Command-line arguments (excluding the program name).  If
            ``None``, ``sys.argv[1:]`` is used.

    Returns:
        Process exit code (0 on success, 1 on error).
    """
    args = argv if argv is not None else sys.argv[1:]

    if len(args) < 1:
        print("Usage: python -m src.main <source-file>")
        return 1

    source_path = Path(args[0])
    if not source_path.exists():
        print(f"Error: file not found: {source_path}")
        return 1

    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: cannot read file: {exc}")
        return 1

    # 1-2. Lex
    try:
        tokens = Lexer(source).tokenize()
    except LexError as exc:
        print(f"Lex error: {exc}")
        return 1

    # 3-4. Parse
    try:
        ast = Parser(tokens).parse()
    except ParseError as exc:
        print(f"Parse error: {exc}")
        return 1

    # 5. Print AST
    print("========== ABSTRACT SYNTAX TREE ==========")
    print()
    print(pretty_print(ast))
    print()

    # 6. Build and print original CFG (Day 1)
    try:
        cfg = CFGBuilder().build(ast)
    except CFGError as exc:
        print(f"CFG construction error: {exc}")
        return 1

    cfg_errors = cfg.validate()
    if cfg_errors:
        print("CFG validation failed:")
        for err in cfg_errors:
            print(f"  - {err}")
        return 1

    print("========== ORIGINAL CONTROL FLOW GRAPH ==========")
    print()
    print(cfg.to_text())

    # 7. Build SSA
    try:
        ssa_builder = SSABuilder()
        ssa_cfg = ssa_builder.build(ast)
    except SSAError as exc:
        print(f"SSA construction error: {exc}")
        return 1

    # 8. Print SSA IR
    print("========== SSA INTERMEDIATE REPRESENTATION ==========")
    print()
    print(ssa_builder.to_ssa_text())

    # 9. Validate SSA
    print("========== SSA VALIDATION ==========")
    print()
    ssa_errors = ssa_builder.validate_ssa()
    if ssa_errors:
        print("SSA validation failed:")
        for err in ssa_errors:
            print(f"  - {err}")
        return 1
    print("SSA validation successful.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
