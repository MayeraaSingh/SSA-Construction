"""Tests for SSA construction.

Covers straight-line assignments, simple conditionals, unmodified variables
across branches, nested conditionals, and undefined-variable detection.
"""

from __future__ import annotations

import pytest

from src.frontend.ast_nodes import (
    Assignment,
    BinaryExpression,
    ComparisonExpression,
    Identifier,
    IfStatement,
    IntegerLiteral,
    Program,
    WhileStatement,
)
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.ssa.phi_functions import PhiFunction
from src.ssa.ssa_builder import (
    SSAAssignment,
    SSABinaryExpression,
    SSABuilder,
    SSAComparisonExpression,
    SSAConditionalBranch,
    SSAError,
    SSAIntegerLiteral,
    SSAUnconditionalJump,
    SSAValue,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def build_ssa(source: str):
    """Lex, parse, and build SSA form from *source*.

    Returns the SSABuilder instance (which holds the CFG).
    """
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    builder = SSABuilder()
    builder.build(ast)
    return builder


def get_ssa_assignments(builder, block_id=None):
    """Return all SSAAssignment instructions, optionally filtered by block."""
    result = []
    for block in builder._cfg.blocks.values():
        if block_id is not None and block.block_id != block_id:
            continue
        for inst in block.ssa_instructions:
            if isinstance(inst, SSAAssignment):
                result.append(inst)
    return result


# ---------------------------------------------------------------------------
# Test 1: Straight-line assignments
# ---------------------------------------------------------------------------


class TestStraightLineAssignments:
    def test_unique_versions(self):
        """x = 10; x = 20; y = x + 5; → x_1, x_2, y_1."""
        builder = build_ssa("x = 10;\nx = 20;\ny = x + 5;")
        assignments = get_ssa_assignments(builder)
        results = [a.result for a in assignments]
        assert results == ["x_1", "x_2", "y_1"]

    def test_no_phi_functions(self):
        builder = build_ssa("x = 10;\nx = 20;\ny = x + 5;")
        for block in builder._cfg.blocks.values():
            assert len(block.phis) == 0

    def test_uses_correct_version(self):
        """y must use x_2 (the latest definition), not x_1."""
        builder = build_ssa("x = 10;\nx = 20;\ny = x + 5;")
        assignments = get_ssa_assignments(builder)
        # y_1 = x_2 + 5
        y_assign = assignments[2]
        assert y_assign.result == "y_1"
        assert isinstance(y_assign.expr, SSABinaryExpression)
        assert isinstance(y_assign.expr.left, SSAValue)
        assert y_assign.expr.left.name == "x_2"

    def test_values_preserved(self):
        builder = build_ssa("x = 10;\nx = 20;\ny = x + 5;")
        assignments = get_ssa_assignments(builder)
        assert assignments[0].result == "x_1"
        assert isinstance(assignments[0].expr, SSAIntegerLiteral)
        assert assignments[0].expr.value == 10
        assert assignments[1].result == "x_2"
        assert assignments[1].expr.value == 20


# ---------------------------------------------------------------------------
# Test 2: Simple conditional
# ---------------------------------------------------------------------------


class TestSimpleConditional:
    SOURCE = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

y = x + 5;"""

    def test_unique_versions(self):
        builder = build_ssa(self.SOURCE)
        assignments = get_ssa_assignments(builder)
        results = [a.result for a in assignments]
        assert "a_1" in results
        assert "x_1" in results
        assert "x_2" in results
        assert "x_3" in results
        assert "y_1" in results

    def test_phi_at_merge_for_x(self):
        """x is modified in both branches, so a phi is needed at the merge."""
        builder = build_ssa(self.SOURCE)
        phi_blocks = [
            b for b in builder._cfg.blocks.values()
            if any(p.variable == "x" and not p.eliminated for p in b.phis)
        ]
        assert len(phi_blocks) == 1
        phi = next(p for p in phi_blocks[0].phis if p.variable == "x" and not p.eliminated)
        # phi should have 2 operands (one per predecessor branch)
        assert len(phi.operands) == 2

    def test_y_uses_merged_x(self):
        """y must use the phi result for x, not a branch-specific version."""
        builder = build_ssa(self.SOURCE)
        assignments = get_ssa_assignments(builder)
        y_assign = next(a for a in assignments if a.result == "y_1")
        # y_1 = phi_x + 5
        assert isinstance(y_assign.expr, SSABinaryExpression)
        x_ref = y_assign.expr.left
        assert isinstance(x_ref, SSAValue)
        # The phi result should be an x_* version (not x_1, x_2, or x_3)
        assert x_ref.name.startswith("x_")
        assert x_ref.name != "x_1"
        assert x_ref.name != "x_2"
        assert x_ref.name != "x_3"

    def test_phi_operands_from_branches(self):
        builder = build_ssa(self.SOURCE)
        phi_blocks = [
            b for b in builder._cfg.blocks.values()
            if any(p.variable == "x" and not p.eliminated for p in b.phis)
        ]
        phi = next(p for p in phi_blocks[0].phis if p.variable == "x" and not p.eliminated)
        operand_values = set(phi.operands.values())
        # One operand should be x_2 (from true branch), one x_3 (from false branch)
        assert "x_2" in operand_values
        assert "x_3" in operand_values

    def test_ssa_validates(self):
        builder = build_ssa(self.SOURCE)
        errors = builder.validate_ssa()
        assert errors == []


# ---------------------------------------------------------------------------
# Test 3: Unmodified variable across branches
# ---------------------------------------------------------------------------


class TestUnmodifiedAcrossBranches:
    SOURCE = """a = 1;
x = 10;

if (a > 0) {
    y = 20;
} else {
    y = 30;
}

z = x + 5;"""

    def test_no_phi_for_x(self):
        """x is not reassigned in either branch, so no phi is needed for x."""
        builder = build_ssa(self.SOURCE)
        x_phis = [
            p for b in builder._cfg.blocks.values()
            for p in b.phis if p.variable == "x" and not p.eliminated
        ]
        assert len(x_phis) == 0

    def test_no_phi_for_y(self):
        """y is not subsequently read after the if-else, so the lazy algorithm
        creates no phi for y.  Only z = x + 5 uses x (which also needs no phi
        since x is not reassigned)."""
        builder = build_ssa(self.SOURCE)
        y_phis = [
            p for b in builder._cfg.blocks.values()
            for p in b.phis if p.variable == "y" and not p.eliminated
        ]
        assert len(y_phis) == 0

    def test_z_uses_original_x(self):
        """z must use x_1 (the original, unmodified definition)."""
        builder = build_ssa(self.SOURCE)
        assignments = get_ssa_assignments(builder)
        z_assign = next(a for a in assignments if a.result == "z_1")
        assert isinstance(z_assign.expr, SSABinaryExpression)
        x_ref = z_assign.expr.left
        assert isinstance(x_ref, SSAValue)
        assert builder._resolve_alias(x_ref.name) == "x_1"

    def test_ssa_validates(self):
        builder = build_ssa(self.SOURCE)
        errors = builder.validate_ssa()
        assert errors == []


# ---------------------------------------------------------------------------
# Test 7: Nested conditional
# ---------------------------------------------------------------------------


class TestNestedConditional:
    SOURCE = """a = 1;
b = 2;
x = 0;

if (a > 0) {
    if (b > 0) {
        x = 10;
    } else {
        x = 20;
    }
} else {
    x = 30;
}

y = x;"""

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

    def test_y_uses_merged_x(self):
        """y should use a phi that merges all three x definitions."""
        builder = build_ssa(self.SOURCE)
        assignments = get_ssa_assignments(builder)
        y_assign = next(a for a in assignments if a.result == "y_1")
        assert isinstance(y_assign.expr, SSAValue)
        assert y_assign.expr.name.startswith("x_")

    def test_ssa_validates(self):
        builder = build_ssa(self.SOURCE)
        errors = builder.validate_ssa()
        assert errors == []

    def test_correct_x_definitions(self):
        builder = build_ssa(self.SOURCE)
        assignments = get_ssa_assignments(builder)
        x_assigns = [a for a in assignments if a.result.startswith("x_")]
        # x_1=0, x_2=10, x_3=20, x_4=30, plus phi result(s)
        x_literal_values = [
            a.expr.value for a in x_assigns
            if isinstance(a.expr, SSAIntegerLiteral)
        ]
        assert 0 in x_literal_values
        assert 10 in x_literal_values
        assert 20 in x_literal_values
        assert 30 in x_literal_values


# ---------------------------------------------------------------------------
# Test 8: Undefined variable
# ---------------------------------------------------------------------------


class TestUndefinedVariable:
    def test_undefined_raises_error(self):
        """x = y + 5; where y is never defined should raise SSAError."""
        with pytest.raises(SSAError, match="undefined variable 'y'"):
            build_ssa("x = y + 5;")

    def test_undefined_in_condition(self):
        with pytest.raises(SSAError, match="undefined variable 'z'"):
            build_ssa("if (z > 5) { x = 1; } else { x = 2; }")

    def test_undefined_in_loop_body(self):
        """Variable used in loop but never defined before the loop."""
        with pytest.raises(SSAError, match="undefined variable 'w'"):
            build_ssa("while (w < 5) { w = w + 1; }")

    def test_no_silent_creation(self):
        """Ensure the error is about the undefined variable, not a generic error."""
        with pytest.raises(SSAError) as exc_info:
            build_ssa("x = y + 5;")
        assert "y" in str(exc_info.value)
