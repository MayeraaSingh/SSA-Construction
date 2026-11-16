"""Command-line entry point for the SSA construction tool.

Drives the full pipeline: source -> lexer -> parser -> AST -> CFG -> SSA
construction -> SSA output -> SSA validation, with optional graphical
CFG/SSA visualization.

Usage::

    python -m src.main <source-file>
    python -m src.main <source-file> --visualize
    python -m src.main <source-file> --visualize --output-dir outputs/
    python -m src.main <source-file> --visualize --cfg-only
    python -m src.main <source-file> --visualize --ssa-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.frontend.ast_nodes import Program, pretty_print
from src.frontend.lexer import LexError, Lexer
from src.frontend.parser import ParseError, Parser
from src.ir.cfg_builder import CFGBuilder
from src.ir.control_flow_graph import CFGError
from src.ssa.ssa_builder import SSAError, SSABuilder


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="SSA Construction Tool — lazy sealed-block algorithm",
    )
    parser.add_argument("source_file", help="Path to the source program file")
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Generate graphical CFG and SSA CFG diagrams as PNG files",
    )
    parser.add_argument(
        "--output-dir", "-o", default="outputs",
        help="Directory for generated diagram files (default: outputs)",
    )
    parser.add_argument(
        "--cfg-only", action="store_true",
        help="With --visualize, generate only the original CFG diagram",
    )
    parser.add_argument(
        "--ssa-only", action="store_true",
        help="With --visualize, generate only the SSA CFG diagram",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline on a single source file.

    Returns:
        Process exit code (0 on success, 1 on error).
    """
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    source_path = Path(args.source_file)
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
        ssa_builder.build(ast)
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

    # 10. Optional visualization
    if args.visualize:
        _generate_visualizations(
            cfg, ssa_builder, source_path.stem, args.output_dir,
            args.cfg_only, args.ssa_only,
        )

    return 0


def _generate_visualizations(
    cfg,
    ssa_builder,
    stem: str,
    output_dir: str,
    cfg_only: bool,
    ssa_only: bool,
) -> None:
    """Generate and save CFG/SSA diagram PNGs."""
    from src.visualization.cfg_visualizer import (
        VisualizationError,
        visualize_cfg,
        visualize_ssa,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not ssa_only:
        cfg_path = out / f"{stem}_cfg.png"
        try:
            visualize_cfg(cfg, cfg_path, title=f"Original CFG — {stem}")
            print(f"  Saved: {cfg_path}")
        except VisualizationError as exc:
            print(f"  Visualization error (CFG): {exc}")

    if not cfg_only:
        ssa_path = out / f"{stem}_ssa.png"
        try:
            visualize_ssa(ssa_builder, ssa_path, title=f"SSA CFG — {stem}")
            print(f"  Saved: {ssa_path}")
        except VisualizationError as exc:
            print(f"  Visualization error (SSA): {exc}")

    print()
    print("========== VISUALIZATION ==========")
    print()
    print(f"Diagrams saved to: {out.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
