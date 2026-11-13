"""Tests for phi-function insertion and handling.

Covers trivial phi elimination, incomplete phi creation and resolution, and
phi operand correctness.
"""

from __future__ import annotations

import pytest

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.ir.basic_block import BasicBlock
from src.ir.control_flow_graph import ControlFlowGraph
from src.ssa.phi_functions import PhiFunction
from src.ssa.ssa_builder import SSABuilder, SSAError, SSAValue


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def build_ssa(source: str):
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    builder = SSABuilder()
    builder.build(ast)
    return builder


# ---------------------------------------------------------------------------
# Test 4: Trivial phi function
# ---------------------------------------------------------------------------


class TestTrivialPhi:
    def test_identical_operands_eliminated(self):
        """When both branches define x to the same SSA value, the phi is trivial."""
        # x = 5; if (a > 0) { } else { } z = x;
        # x is not modified in either branch, so the merge phi has
        # operands (x_1, x_1) → trivial, eliminated.
        source = "x = 5;\nif (x > 0) {\n    y = 1;\n} else {\n    y = 2;\n}\nz = x;"
        builder = build_ssa(source)
        x_phis = [
            p for b in builder._cfg.blocks.values()
            for p in b.phis if p.variable == "x"
        ]
        # The phi for x should have been created and then eliminated.
        assert len(x_phis) >= 1
        eliminated = [p for p in x_phis if p.eliminated]
        assert len(eliminated) >= 1

    def test_eliminated_phi_redirects_uses(self):
        """Uses of an eliminated phi should resolve to the surviving value."""
        source = "x = 5;\nif (x > 0) {\n    y = 1;\n} else {\n    y = 2;\n}\nz = x;"
        builder = build_ssa(source)
        # z should use x_1 (the original), not the phi result.
        z_assigns = [
            inst
            for b in builder._cfg.blocks.values()
            for inst in b.ssa_instructions
            if hasattr(inst, "result") and inst.result == "z_1"
        ]
        assert len(z_assigns) == 1
        z_expr = z_assigns[0].expr
        assert isinstance(z_expr, SSAValue)
        assert builder._resolve_alias(z_expr.name) == "x_1"

    def test_different_operands_not_eliminated(self):
        """When operands differ, the phi must NOT be eliminated."""
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        x_phis = [
            p for b in builder._cfg.blocks.values()
            for p in b.phis if p.variable == "x"
        ]
        non_eliminated = [p for p in x_phis if not p.eliminated]
        assert len(non_eliminated) == 1

    def test_self_reference_eliminated(self):
        """phi(x_3, x_1) where x_3 is self → trivial, replaced by x_1.

        This happens when a variable is not modified in one branch of a loop.
        The backedge operand is the phi itself (self-reference), which is
        skipped, leaving only the entry operand → trivial.
        """
        source = """x = 0;
y = 10;

while (x < 5) {
    x = x + 1;
}

z = y;"""
        builder = build_ssa(source)
        # y is not modified in the loop.  A phi is created for y at the header
        # (because the header is unsealed when y is read), but the backedge
        # operand is the phi itself (self-reference).  So the phi is trivial
        # and eliminated.
        y_phis = [
            p for b in builder._cfg.blocks.values()
            for p in b.phis if p.variable == "y"
        ]
        # At least one y phi should be eliminated (trivial).
        eliminated = [p for p in y_phis if p.eliminated]
        assert len(eliminated) >= 1


# ---------------------------------------------------------------------------
# Test 5: Incomplete phi function
# ---------------------------------------------------------------------------


class TestIncompletePhi:
    def test_incomplete_phi_created_for_unsealed_block(self):
        """Reading a variable from an unsealed block creates an incomplete phi."""
        # Build a simple CFG manually with a loop-like structure.
        cfg = ControlFlowGraph()
        builder = SSABuilder(cfg)

        b0 = cfg.create_block()  # entry
        b1 = cfg.create_block()  # header (will stay unsealed)
        b2 = cfg.create_block()  # body

        cfg.add_edge(b0, b1)
        cfg.add_edge(b1, b2)
        cfg.add_edge(b2, b1)  # backedge (not yet — we'll test before sealing)

        # Seal entry, define x.
        builder.seal_block(b0)
        x1 = builder.new_ssa_value("x")
        builder.write_variable("x", b0, x1)

        # Seal body (only pred is header).
        builder.seal_block(b2)

        # Read x from unsealed header → should create incomplete phi.
        result = builder.read_variable("x", b1)

        # Verify: incomplete phi exists.
        assert len(b1.phis) == 1
        phi = b1.phis[0]
        assert phi.variable == "x"
        assert len(phi.operands) == 0  # Incomplete
        assert phi.result == result

    def test_incomplete_phi_resolved_on_seal(self):
        """Sealing the block resolves the incomplete phi's operands."""
        cfg = ControlFlowGraph()
        builder = SSABuilder(cfg)

        b0 = cfg.create_block()
        b1 = cfg.create_block()
        b2 = cfg.create_block()

        cfg.add_edge(b0, b1)
        cfg.add_edge(b1, b2)
        cfg.add_edge(b2, b1)

        builder.seal_block(b0)
        x1 = builder.new_ssa_value("x")
        builder.write_variable("x", b0, x1)

        builder.seal_block(b2)
        x2 = builder.new_ssa_value("x")
        builder.write_variable("x", b2, x2)

        # Read x from unsealed header.
        builder.read_variable("x", b1)
        phi = b1.phis[0]
        assert len(phi.operands) == 0  # Still incomplete

        # Seal header → resolves operands.
        builder.seal_block(b1)

        assert len(phi.operands) == 2
        assert phi.operands[b0.block_id] == x1
        assert phi.operands[b2.block_id] == x2
        assert b1.sealed

    def test_result_recorded_before_operand_resolution(self):
        """The phi result must be the current definition before operands are read.

        This prevents infinite recursion on backedges.
        """
        cfg = ControlFlowGraph()
        builder = SSABuilder(cfg)

        b0 = cfg.create_block()
        b1 = cfg.create_block()
        b2 = cfg.create_block()

        cfg.add_edge(b0, b1)
        cfg.add_edge(b1, b2)
        cfg.add_edge(b2, b1)

        builder.seal_block(b0)
        builder.write_variable("x", b0, builder.new_ssa_value("x"))

        builder.seal_block(b2)
        builder.write_variable("x", b2, builder.new_ssa_value("x"))

        # Read from unsealed header.
        phi_result = builder.read_variable("x", b1)

        # The phi result should be recorded as current_defs before sealing.
        assert b1.current_defs["x"] == phi_result

        # Seal and verify no infinite recursion.
        builder.seal_block(b1)

        # The phi should be resolved with 2 operands.
        phi = b1.phis[0]
        assert len(phi.operands) == 2

    def test_incomplete_phi_in_actual_loop(self):
        """In a real loop, the header's phi starts incomplete and is resolved on seal."""
        source = """x = 0;

while (x < 5) {
    x = x + 1;
}

y = x;"""
        builder = build_ssa(source)

        # The header should have a phi for x that is now complete.
        from src.ssa.ssa_builder import SSAConditionalBranch
        headers = [
            b for b in builder._cfg.blocks.values()
            if any(isinstance(i, SSAConditionalBranch) for i in b.ssa_instructions)
        ]
        header = headers[0]
        x_phis = [p for p in header.phis if p.variable == "x" and not p.eliminated]
        assert len(x_phis) == 1
        assert len(x_phis[0].operands) == 2  # Complete


# ---------------------------------------------------------------------------
# Phi function structure tests
# ---------------------------------------------------------------------------


class TestPhiStructure:
    def test_phi_has_one_operand_per_predecessor(self):
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        for block in builder._cfg.blocks.values():
            for phi in block.phis:
                if not phi.eliminated:
                    assert len(phi.operands) == len(block.predecessors)

    def test_phi_operands_associated_with_valid_predecessors(self):
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        for block in builder._cfg.blocks.values():
            for phi in block.phis:
                if not phi.eliminated:
                    for pred_id in phi.operands:
                        assert pred_id in block.predecessors

    def test_phi_result_is_unique_ssa_value(self):
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        all_results = set()
        for block in builder._cfg.blocks.values():
            for phi in block.phis:
                if not phi.eliminated:
                    assert phi.result not in all_results
                    all_results.add(phi.result)

    def test_all_phis_resolved_after_build(self):
        """No incomplete phis should remain after build() completes."""
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        for block in builder._cfg.blocks.values():
            for phi in block.phis:
                if not phi.eliminated:
                    assert len(phi.operands) > 0
