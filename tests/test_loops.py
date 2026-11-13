"""Tests for SSA construction over loop structures.

Covers loop header phi functions, backedge handling, loop-carried variables,
and correct post-loop variable resolution.
"""

from __future__ import annotations

import pytest

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.ssa.ssa_builder import (
    SSAAssignment,
    SSABinaryExpression,
    SSABuilder,
    SSAConditionalBranch,
    SSAValue,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def build_ssa(source: str):
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    builder = SSABuilder()
    builder.build(ast)
    return builder


def get_ssa_assignments(builder):
    result = []
    for block in builder._cfg.blocks.values():
        for inst in block.ssa_instructions:
            if isinstance(inst, SSAAssignment):
                result.append(inst)
    return result


# ---------------------------------------------------------------------------
# Test 6: Loop header
# ---------------------------------------------------------------------------


class TestLoopHeader:
    SOURCE = """x = 0;

while (x < 5) {
    x = x + 1;
}

y = x;"""

    def test_phi_at_loop_header(self):
        """The loop header must have a phi function for x."""
        builder = build_ssa(self.SOURCE)
        # Find the conditional block (header)
        headers = [
            b for b in builder._cfg.blocks.values()
            if any(
                isinstance(i, SSAConditionalBranch)
                for i in b.ssa_instructions
            )
        ]
        assert len(headers) == 1
        header = headers[0]
        x_phis = [p for p in header.phis if p.variable == "x" and not p.eliminated]
        assert len(x_phis) == 1

    def test_phi_has_two_operands(self):
        """The phi should have one operand from entry and one from backedge."""
        builder = build_ssa(self.SOURCE)
        headers = [
            b for b in builder._cfg.blocks.values()
            if any(
                isinstance(i, SSAConditionalBranch)
                for i in b.ssa_instructions
            )
        ]
        header = headers[0]
        phi = next(p for p in header.phis if p.variable == "x" and not p.eliminated)
        assert len(phi.operands) == 2

    def test_phi_operand_from_entry(self):
        """One operand should be x_1 (defined before the loop)."""
        builder = build_ssa(self.SOURCE)
        headers = [
            b for b in builder._cfg.blocks.values()
            if any(
                isinstance(i, SSAConditionalBranch)
                for i in b.ssa_instructions
            )
        ]
        header = headers[0]
        phi = next(p for p in header.phis if p.variable == "x" and not p.eliminated)
        operand_values = set(phi.operands.values())
        assert "x_1" in operand_values

    def test_phi_operand_from_backedge(self):
        """One operand should come from the loop body's definition.

        Versioning: x_1=0 (entry), x_2=phi (header), x_3=x_2+1 (body).
        The backedge operand is x_3.
        """
        builder = build_ssa(self.SOURCE)
        headers = [
            b for b in builder._cfg.blocks.values()
            if any(
                isinstance(i, SSAConditionalBranch)
                for i in b.ssa_instructions
            )
        ]
        header = headers[0]
        phi = next(p for p in header.phis if p.variable == "x" and not p.eliminated)
        operand_values = set(phi.operands.values())
        # x_1 from entry, x_3 from body backedge
        assert "x_1" in operand_values
        assert "x_3" in operand_values

    def test_body_uses_phi_result(self):
        """The loop body's x = x + 1 should use the phi result.

        Versioning: x_2 is the phi result.  The body assigns x_3 = x_2 + 1.
        """
        builder = build_ssa(self.SOURCE)
        assignments = get_ssa_assignments(builder)
        # Find the body assignment: x_3 = x_2 + 1
        x3_assign = next(a for a in assignments if a.result == "x_3")
        assert isinstance(x3_assign.expr, SSABinaryExpression)
        x_ref = x3_assign.expr.left
        assert isinstance(x_ref, SSAValue)
        # x_2 is the phi result
        assert x_ref.name == "x_2"

    def test_condition_uses_phi_result(self):
        """The loop condition should reference the phi result for x.

        Versioning: x_2 is the phi result at the header.  The condition
        uses x_2, which is NOT the entry definition (x_1) and NOT the body
        definition (x_3).
        """
        builder = build_ssa(self.SOURCE)
        headers = [
            b for b in builder._cfg.blocks.values()
            if any(
                isinstance(i, SSAConditionalBranch)
                for i in b.ssa_instructions
            )
        ]
        header = headers[0]
        cond = next(
            i for i in header.ssa_instructions if isinstance(i, SSAConditionalBranch)
        )
        from src.ssa.ssa_builder import SSAComparisonExpression
        assert isinstance(cond.condition, SSAComparisonExpression)
        left = cond.condition.left
        assert isinstance(left, SSAValue)
        assert left.name == "x_2"  # The phi result
        assert left.name != "x_1"  # Not the original definition
        assert left.name != "x_3"  # Not the body's definition

    def test_post_loop_uses_phi_result(self):
        """y = x after the loop should use the phi result."""
        builder = build_ssa(self.SOURCE)
        assignments = get_ssa_assignments(builder)
        y_assign = next(a for a in assignments if a.result == "y_1")
        assert isinstance(y_assign.expr, SSAValue)
        assert y_assign.expr.name.startswith("x_")

    def test_ssa_validates(self):
        builder = build_ssa(self.SOURCE)
        errors = builder.validate_ssa()
        assert errors == []

    def test_all_blocks_sealed(self):
        builder = build_ssa(self.SOURCE)
        for block in builder._cfg.blocks.values():
            assert block.sealed


# ---------------------------------------------------------------------------
# Loop with unmodified variable
# ---------------------------------------------------------------------------


class TestLoopUnmodifiedVariable:
    SOURCE = """x = 0;
y = 10;

while (x < 5) {
    x = x + 1;
}

z = y;"""

    def test_no_phi_for_y(self):
        """y is not modified in the loop, so no phi should survive for y."""
        builder = build_ssa(self.SOURCE)
        y_phis = [
            p for b in builder._cfg.blocks.values()
            for p in b.phis if p.variable == "y" and not p.eliminated
        ]
        assert len(y_phis) == 0

    def test_z_uses_original_y(self):
        builder = build_ssa(self.SOURCE)
        assignments = get_ssa_assignments(builder)
        z_assign = next(a for a in assignments if a.result == "z_1")
        assert isinstance(z_assign.expr, SSAValue)
        assert builder._resolve_alias(z_assign.expr.name) == "y_1"

    def test_ssa_validates(self):
        builder = build_ssa(self.SOURCE)
        errors = builder.validate_ssa()
        assert errors == []


# ---------------------------------------------------------------------------
# Nested loop
# ---------------------------------------------------------------------------


class TestNestedLoop:
    SOURCE = """x = 0;

while (x < 3) {
    y = 0;
    while (y < 2) {
        y = y + 1;
    }
    x = x + 1;
}

z = x;"""

    def test_ssa_validates(self):
        builder = build_ssa(self.SOURCE)
        errors = builder.validate_ssa()
        assert errors == []

    def test_all_blocks_sealed(self):
        builder = build_ssa(self.SOURCE)
        for block in builder._cfg.blocks.values():
            assert block.sealed

    def test_no_incomplete_phis(self):
        builder = build_ssa(self.SOURCE)
        for block in builder._cfg.blocks.values():
            for phi in block.phis:
                if not phi.eliminated:
                    assert len(phi.operands) > 0

    def test_two_loop_header_phis(self):
        """Both inner and outer loop headers should have phi functions for their loop variables."""
        builder = build_ssa(self.SOURCE)
        x_phis = [
            p for b in builder._cfg.blocks.values()
            for p in b.phis if p.variable == "x" and not p.eliminated
        ]
        y_phis = [
            p for b in builder._cfg.blocks.values()
            for p in b.phis if p.variable == "y" and not p.eliminated
        ]
        assert len(x_phis) >= 1
        assert len(y_phis) >= 1
