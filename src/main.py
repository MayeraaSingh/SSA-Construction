"""Command-line entry point for the SSA construction tool.

Drives the pipeline from source parsing through CFG construction.  For Day 1
the pipeline stops after CFG construction and textual output — SSA conversion
is not yet implemented.

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


def main(argv: list[str] | None = None) -> int:
    """Run the full Day 1 pipeline on a single source file.

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
    print("=== AST ===")
    print(pretty_print(ast))
    print()

    # 6. Build CFG
    try:
        cfg = CFGBuilder().build(ast)
    except CFGError as exc:
        print(f"CFG construction error: {exc}")
        return 1

    # 7. Validate CFG
    errors = cfg.validate()
    if errors:
        print("CFG validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    # 8. Print CFG
    print(cfg.to_text())

    return 0


if __name__ == "__main__":
    sys.exit(main())
