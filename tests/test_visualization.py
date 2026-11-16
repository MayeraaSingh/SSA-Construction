"""Tests for graphical CFG visualization.

Covers rendering of original CFG and SSA CFG, saving to disk, data
immutability, and rendering of complex control-flow structures.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.ir.cfg_builder import CFGBuilder
from src.ir.control_flow_graph import ControlFlowGraph
from src.ssa.ssa_builder import SSABuilder
from src.visualization.cfg_visualizer import (
    VisualizationError,
    _VIZ_AVAILABLE,
    visualize_cfg,
    visualize_ssa,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_cfg(source: str) -> ControlFlowGraph:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return CFGBuilder().build(ast)


def build_ssa(source: str) -> SSABuilder:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    builder = SSABuilder()
    builder.build(ast)
    return builder


# Skip all tests if matplotlib/networkx not available.
pytestmark = pytest.mark.skipif(
    not _VIZ_AVAILABLE,
    reason="matplotlib or networkx not installed",
)


# ---------------------------------------------------------------------------
# File generation tests
# ---------------------------------------------------------------------------


class TestFileGeneration:
    def test_cfg_renders_png(self, tmp_path):
        cfg = build_cfg("x = 1;\ny = 2;")
        out = tmp_path / "test_cfg.png"
        result = visualize_cfg(cfg, out)
        assert result.exists()
        assert result.suffix == ".png"
        assert result.stat().st_size > 0

    def test_ssa_renders_png(self, tmp_path):
        builder = build_ssa("x = 1;\ny = 2;")
        out = tmp_path / "test_ssa.png"
        result = visualize_ssa(builder, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_creates_output_directory(self, tmp_path):
        cfg = build_cfg("x = 1;")
        out = tmp_path / "subdir" / "nested" / "test.png"
        result = visualize_cfg(cfg, out)
        assert result.exists()
        assert out.parent.exists()


# ---------------------------------------------------------------------------
# Data immutability tests
# ---------------------------------------------------------------------------


class TestDataImmutability:
    def test_cfg_not_modified(self, tmp_path):
        source = "x = 1;\nif (x > 0) { y = 2; } else { y = 3; }\nz = y;"
        cfg = build_cfg(source)
        # Snapshot block data.
        snapshot = {
            bid: (
                list(b.instructions),
                list(b.predecessors),
                list(b.successors),
                list(b.ssa_instructions),
                list(b.phis),
            )
            for bid, b in cfg.blocks.items()
        }
        visualize_cfg(cfg, tmp_path / "cfg.png")
        # Verify nothing changed.
        for bid, b in cfg.blocks.items():
            assert list(b.instructions) == snapshot[bid][0]
            assert list(b.predecessors) == snapshot[bid][1]
            assert list(b.successors) == snapshot[bid][2]
            assert list(b.ssa_instructions) == snapshot[bid][3]
            assert list(b.phis) == snapshot[bid][4]

    def test_ssa_not_modified(self, tmp_path):
        source = "x = 1;\nif (x > 0) { y = 2; } else { y = 3; }\nz = y;"
        builder = build_ssa(source)
        cfg = builder._cfg
        snapshot = {
            bid: (
                list(b.ssa_instructions),
                list(b.phis),
                dict(b.current_defs),
            )
            for bid, b in cfg.blocks.items()
        }
        aliases_before = dict(builder._aliases)
        visualize_ssa(builder, tmp_path / "ssa.png")
        for bid, b in cfg.blocks.items():
            assert list(b.ssa_instructions) == snapshot[bid][0]
            assert list(b.phis) == snapshot[bid][1]
            assert dict(b.current_defs) == snapshot[bid][2]
        assert dict(builder._aliases) == aliases_before


# ---------------------------------------------------------------------------
# Complex control flow rendering tests
# ---------------------------------------------------------------------------


class TestComplexRendering:
    def test_conditional(self, tmp_path):
        source = "x = 10;\nif (x > 5) { y = 20; } else { y = 30; }\nz = y + 5;"
        cfg = build_cfg(source)
        builder = build_ssa(source)
        assert visualize_cfg(cfg, tmp_path / "cond_cfg.png").exists()
        assert visualize_ssa(builder, tmp_path / "cond_ssa.png").exists()

    def test_while_loop(self, tmp_path):
        source = "x = 0;\nwhile (x < 5) { x = x + 1; }\ny = x;"
        cfg = build_cfg(source)
        builder = build_ssa(source)
        assert visualize_cfg(cfg, tmp_path / "loop_cfg.png").exists()
        assert visualize_ssa(builder, tmp_path / "loop_ssa.png").exists()

    def test_nested_if(self, tmp_path):
        source = """x = 1;
if (x > 0) {
    if (x > 5) { y = 10; } else { y = 20; }
} else {
    y = 30;
}
z = y + 1;"""
        cfg = build_cfg(source)
        builder = build_ssa(source)
        assert visualize_cfg(cfg, tmp_path / "nested_cfg.png").exists()
        assert visualize_ssa(builder, tmp_path / "nested_ssa.png").exists()

    def test_nested_loops(self, tmp_path):
        source = """x = 0;
y = 0;
while (x < 3) {
    y = 0;
    while (y < 2) {
        y = y + 1;
    }
    x = x + 1;
}
z = x;"""
        cfg = build_cfg(source)
        builder = build_ssa(source)
        assert visualize_cfg(cfg, tmp_path / "nloop_cfg.png").exists()
        assert visualize_ssa(builder, tmp_path / "nloop_ssa.png").exists()

    def test_loop_with_if_else(self, tmp_path):
        source = """x = 0;
y = 0;
while (x < 10) {
    if (x > 5) { y = y + 10; } else { y = y + 1; }
    x = x + 1;
}
result = y;"""
        cfg = build_cfg(source)
        builder = build_ssa(source)
        assert visualize_cfg(cfg, tmp_path / "loopif_cfg.png").exists()
        assert visualize_ssa(builder, tmp_path / "loopif_ssa.png").exists()

    def test_empty_program(self, tmp_path):
        cfg = build_cfg("")
        builder = build_ssa("")
        assert visualize_cfg(cfg, tmp_path / "empty_cfg.png").exists()
        assert visualize_ssa(builder, tmp_path / "empty_ssa.png").exists()

    def test_sequential_if_else(self, tmp_path):
        source = """a = 1;
x = 10;
y = 20;
if (a > 0) { x = 30; } else { x = 40; }
if (a > 0) { y = 50; } else { y = 60; }
z = x + y;"""
        cfg = build_cfg(source)
        builder = build_ssa(source)
        assert visualize_cfg(cfg, tmp_path / "seq_cfg.png").exists()
        assert visualize_ssa(builder, tmp_path / "seq_ssa.png").exists()


# ---------------------------------------------------------------------------
# All examples render test
# ---------------------------------------------------------------------------


class TestAllExamples:
    def test_all_examples_render(self, tmp_path):
        examples_dir = Path(__file__).parent.parent / "examples"
        for example_file in sorted(examples_dir.glob("*.c")):
            source = example_file.read_text(encoding="utf-8")
            cfg = build_cfg(source)
            builder = build_ssa(source)
            stem = example_file.stem
            assert visualize_cfg(cfg, tmp_path / f"{stem}_cfg.png").exists()
            assert visualize_ssa(builder, tmp_path / f"{stem}_ssa.png").exists()
