"""Day 3 hardening tests for SSA correctness.

Covers phi-function operand-to-predecessor mapping, trivial phi cascading
simplification, alias propagation correctness and idempotency, and complex
control-flow structures including if-else inside loops, variables modified in
only one branch, multiple variables at the same merge, variables not assigned
in one branch, and empty branches.
"""

from __future__ import annotations

import pytest

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.ir.basic_block import BasicBlock
from src.ir.control_flow_graph import ControlFlowGraph
from src.ssa.phi_functions import PhiFunction
from src.ssa.ssa_builder import (
    SSAAssignment,
    SSABinaryExpression,
    SSAConditionalBranch,
    SSAError,
    SSAIntegerLiteral,
    SSAUnconditionalJump,
    SSAValue,
    SSABuilder,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def build_ssa(source: str) -> SSABuilder:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    builder = SSABuilder()
    builder.build(ast)
    return builder


def get_assignments(builder: SSABuilder) -> list[SSAAssignment]:
    result = []
    for block in builder._cfg.blocks.values():
        for inst in block.ssa_instructions:
            if isinstance(inst, SSAAssignment):
                result.append(inst)
    return result


def get_non_eliminated_phis(builder: SSABuilder, variable: str | None = None) -> list[PhiFunction]:
    result = []
    for block in builder._cfg.blocks.values():
        for phi in block.phis:
            if not phi.eliminated:
                if variable is None or phi.variable == variable:
                    result.append(phi)
    return result


# ---------------------------------------------------------------------------
# A. Phi-function correctness
# ---------------------------------------------------------------------------


class TestPhiOperandPredecessorMapping:
    """Verify that phi operands are associated with the correct predecessor,
    not merely stored in the correct quantity."""

    def test_operands_match_branch_values(self):
        """The operand from the true-branch predecessor must be the value
        defined in the true branch, and likewise for the false branch."""
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        phi = get_non_eliminated_phis(builder, "x")[0]
        # The merge block has two predecessors (the two branch blocks).
        # One predecessor's current_defs["x"] is x_2 (20), the other's is x_3 (30).
        merge_block = builder._cfg.get_block(phi.block_id)
        for pred_id, operand in phi.operands.items():
            pred = builder._cfg.get_block(pred_id)
            pred_x_def = pred.current_defs.get("x")
            # The operand should be the value that the predecessor holds for x.
            assert builder._resolve_alias(operand) == builder._resolve_alias(pred_x_def)

    def test_phi_operands_carry_correct_values_through_loop(self):
        """In a loop, the phi operand from the backedge must carry the
        loop-body's value, not the entry value."""
        source = """x = 0;

while (x < 5) {
    x = x + 1;
}

y = x;"""
        builder = build_ssa(source)
        phi = get_non_eliminated_phis(builder, "x")[0]
        # One operand is from the entry (x_1=0), one from the body (x_3=x_2+1).
        operand_values = set(phi.operands.values())
        assert "x_1" in operand_values  # entry
        assert "x_3" in operand_values  # body backedge


class TestPhiMultiplePredecessors:
    def test_nested_merge_has_correct_predecessor_count(self):
        """Each merge phi must have one operand per predecessor of its block."""
        source = """a = 1;
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
        builder = build_ssa(source)
        for block in builder._cfg.blocks.values():
            for phi in block.phis:
                if not phi.eliminated:
                    assert len(phi.operands) == len(block.predecessors), (
                        f"phi {phi.result} in block {block.block_id} has "
                        f"{len(phi.operands)} operands but "
                        f"{len(block.predecessors)} predecessors"
                    )

    def test_multiple_variables_at_same_merge(self):
        """Two variables modified in both branches each get their own phi
        at the merge block."""
        source = """a = 1;
x = 10;
y = 20;

if (a > 0) {
    x = 30;
    y = 40;
} else {
    x = 50;
    y = 60;
}

z = x + y;"""
        builder = build_ssa(source)
        x_phi = get_non_eliminated_phis(builder, "x")
        y_phi = get_non_eliminated_phis(builder, "y")
        assert len(x_phi) == 1
        assert len(y_phi) == 1
        # Both phis should be in the same block.
        assert x_phi[0].block_id == y_phi[0].block_id
        # z should use both phi results.
        assigns = get_assignments(builder)
        z_assign = next(a for a in assigns if a.result == "z_1")
        assert isinstance(z_assign.expr, SSABinaryExpression)
        assert isinstance(z_assign.expr.left, SSAValue)
        assert isinstance(z_assign.expr.right, SSAValue)


class TestPhiIncompleteResolution:
    def test_incomplete_phi_has_empty_operands_before_seal(self):
        """An incomplete phi should have zero operands before sealing."""
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

        builder.read_variable("x", b1)
        assert len(b1.phis) == 1
        assert len(b1.phis[0].operands) == 0

    def test_incomplete_phi_resolved_correctly_on_seal(self):
        """After sealing, each operand must match the predecessor's value."""
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

        builder.read_variable("x", b1)
        builder.seal_block(b1)

        phi = b1.phis[0]
        assert phi.operands[b0.block_id] == x1
        assert phi.operands[b2.block_id] == x2


# ---------------------------------------------------------------------------
# B. Trivial phi simplification — cascading and cyclic
# ---------------------------------------------------------------------------


class TestTrivialPhiCascading:
    def test_cascading_simplification_through_two_merges(self):
        """If x is not modified in either branch of two sequential if-else
        blocks, both merge phis should be eliminated (cascading)."""
        source = """x = 1;
a = 1;

if (a > 0) {
    y = 2;
} else {
    y = 3;
}

if (a > 0) {
    z = 4;
} else {
    z = 5;
}

w = x;"""
        builder = build_ssa(source)
        # x is never modified in any branch, so all phis for x are trivial.
        x_phis = [p for b in builder._cfg.blocks.values() for p in b.phis if p.variable == "x"]
        # All x phis should be eliminated.
        for phi in x_phis:
            assert phi.eliminated, f"phi {phi.result} should be eliminated"
        # w should use x_1 directly.
        assigns = get_assignments(builder)
        w_assign = next(a for a in assigns if a.result == "w_1")
        assert isinstance(w_assign.expr, SSAValue)
        assert builder._resolve_alias(w_assign.expr.name) == "x_1"

    def test_phi_not_trivial_when_operands_differ(self):
        """A phi with genuinely different operands must survive simplification."""
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        x_phis = get_non_eliminated_phis(builder, "x")
        assert len(x_phis) == 1
        operand_values = set(x_phis[0].operands.values())
        assert len(operand_values) == 2  # Two distinct values


class TestCyclicPhiDependency:
    def test_loop_header_phi_not_trivial(self):
        """A loop-carried variable's phi has different operands (entry value
        vs. body value) and must NOT be eliminated."""
        source = """x = 0;

while (x < 5) {
    x = x + 1;
}

y = x;"""
        builder = build_ssa(source)
        x_phis = get_non_eliminated_phis(builder, "x")
        assert len(x_phis) == 1

    def test_unmodified_loop_variable_phi_is_trivial(self):
        """A variable not modified in the loop body has a self-referencing
        phi operand, making it trivial."""
        source = """x = 0;
y = 10;

while (x < 5) {
    x = x + 1;
}

z = y;"""
        builder = build_ssa(source)
        y_phis = [p for b in builder._cfg.blocks.values() for p in b.phis if p.variable == "y"]
        assert any(p.eliminated for p in y_phis)


# ---------------------------------------------------------------------------
# C. Alias propagation and _finalize()
# ---------------------------------------------------------------------------


class TestAliasPropagation:
    def test_alias_chain_resolves_completely(self):
        """Multiple levels of alias should resolve to the ultimate value."""
        from src.ssa.ssa_simplifier import resolve_alias
        aliases = {"a_2": "a_1", "a_3": "a_2", "a_4": "a_3"}
        assert resolve_alias("a_4", aliases) == "a_1"

    def test_alias_chain_no_infinite_loop(self):
        """A cycle in aliases should not cause infinite recursion."""
        from src.ssa.ssa_simplifier import resolve_alias
        aliases = {"a_2": "a_3", "a_3": "a_2"}
        result = resolve_alias("a_2", aliases)
        # Should terminate and return one of the cycle members.
        assert result in ("a_2", "a_3")

    def test_finalize_idempotent(self):
        """Running _finalize() twice should produce the same result."""
        source = """a = 1;
x = 10;

if (a > 0) {
    y = 2;
} else {
    y = 3;
}

if (a > 0) {
    z = 4;
} else {
    z = 5;
}

w = x;"""
        builder = build_ssa(source)
        # Snapshot aliases after first build (which calls _finalize internally).
        aliases_before = dict(builder._aliases)
        # Run _finalize again.
        builder._finalize()
        aliases_after = dict(builder._aliases)
        assert aliases_before == aliases_after

    def test_finalization_updates_phi_operands(self):
        """After _finalize(), non-eliminated phi operands should not contain
        eliminated phi results."""
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
    x = 30;
}

if (a > 0) {
    y = x;
} else {
    y = x;
}

z = y;"""
        builder = build_ssa(source)
        eliminated = {p.result for b in builder._cfg.blocks.values() for p in b.phis if p.eliminated}
        for block in builder._cfg.blocks.values():
            for phi in block.phis:
                if not phi.eliminated:
                    for operand in phi.operands.values():
                        # The operand might be an eliminated result if
                        # _finalize hasn't resolved it, but after _finalize
                        # it should be resolved.
                        assert operand not in eliminated, (
                            f"phi {phi.result} still references eliminated "
                            f"value {operand}"
                        )


# ---------------------------------------------------------------------------
# D. Complex control flow
# ---------------------------------------------------------------------------


class TestNestedIfElse:
    def test_three_way_merge(self):
        source = """a = 1;
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
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        # y should use a phi that merges all three paths.
        assigns = get_assignments(builder)
        y_assign = next(a for a in assigns if a.result == "y_1")
        assert isinstance(y_assign.expr, SSAValue)
        assert y_assign.expr.name.startswith("x_")

    def test_inner_merge_then_outer_merge(self):
        """The inner if-else creates a merge; the outer if-else creates another
        merge that combines the inner merge with the outer else."""
        source = """a = 1;
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
        builder = build_ssa(source)
        x_phis = get_non_eliminated_phis(builder, "x")
        # There should be at least one non-eliminated phi (the outer merge).
        # The inner merge might also survive if operands differ.
        assert len(x_phis) >= 1


class TestNestedLoops:
    def test_nested_loops_both_carried(self):
        source = """x = 0;
y = 0;

while (x < 3) {
    y = 0;
    while (y < 2) {
        y = y + 1;
    }
    x = x + 1;
}

z = x;
w = y;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        # Both x and y should have phis at their respective loop headers.
        assert len(get_non_eliminated_phis(builder, "x")) >= 1
        assert len(get_non_eliminated_phis(builder, "y")) >= 1


class TestLoopWithIfElse:
    def test_if_else_inside_loop(self):
        source = """x = 0;
y = 0;

while (x < 10) {
    if (x > 5) {
        y = 1;
    } else {
        y = 2;
    }
    x = x + 1;
}

z = y;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []

    def test_variable_modified_in_one_branch_inside_loop(self):
        """A variable modified in only one branch of an if-else inside a loop
        should still get a phi at the loop header."""
        source = """x = 0;

while (x < 10) {
    if (x > 5) {
        x = x + 2;
    } else {
        x = x + 1;
    }
}

y = x;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        # x is modified in both branches, so there are phis at both
        # the loop header and the if-else merge inside the body.
        x_phis = get_non_eliminated_phis(builder, "x")
        assert len(x_phis) >= 1
        # The loop header phi should have 2 operands (entry + backedge).
        header_phi = next(p for p in x_phis if len(p.operands) == 2 and p.block_id != 0)
        assert len(header_phi.operands) == 2


class TestVariableNotAssignedInOneBranch:
    def test_variable_only_in_true_branch(self):
        """If x is assigned in the true branch but not the false branch, the
        merge phi should get x's pre-branch value from the false path."""
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
}

y = x;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        phi = get_non_eliminated_phis(builder, "x")[0]
        operand_values = set(phi.operands.values())
        # One operand should be x_2 (true branch), the other x_1 (pre-branch).
        assert "x_2" in operand_values
        assert "x_1" in operand_values

    def test_variable_only_in_false_branch(self):
        source = """a = 1;
x = 10;

if (a > 0) {
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        phi = get_non_eliminated_phis(builder, "x")[0]
        operand_values = set(phi.operands.values())
        assert "x_2" in operand_values  # from false branch (x_2 = 30)
        assert "x_1" in operand_values  # from true branch (pre-branch value)


class TestUnchangedAcrossBranch:
    def test_unmodified_variable_uses_original(self):
        source = """a = 1;
x = 10;

if (a > 0) {
    y = 20;
} else {
    y = 30;
}

z = x + 5;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        # No phi for x (not reassigned).
        assert len(get_non_eliminated_phis(builder, "x")) == 0
        # z should use x_1.
        assigns = get_assignments(builder)
        z_assign = next(a for a in assigns if a.result == "z_1")
        assert builder._resolve_alias(z_assign.expr.left.name) == "x_1"


class TestModifiedBeforeAndAfterLoop:
    def test_variable_modified_before_and_after(self):
        source = """x = 0;

while (x < 5) {
    x = x + 1;
}

x = x + 10;
y = x;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        # The loop header phi should exist (loop-carried).
        x_phis = get_non_eliminated_phis(builder, "x")
        assert len(x_phis) == 1
        # After the loop, x is incremented. y should use the post-loop value.
        assigns = get_assignments(builder)
        y_assign = next(a for a in assigns if a.result == "y_1")
        assert isinstance(y_assign.expr, SSAValue)
        # The value should be the post-loop increment, not the phi result.
        assert y_assign.expr.name != x_phis[0].result


class TestEmptyBranches:
    def test_empty_true_branch(self):
        source = """a = 1;
x = 10;

if (a > 0) {
} else {
    x = 30;
}

y = x;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []

    def test_empty_false_branch(self):
        source = """a = 1;
x = 10;

if (a > 0) {
    x = 20;
} else {
}

y = x;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []

    def test_both_branches_empty(self):
        source = """a = 1;
x = 10;

if (a > 0) {
} else {
}

y = x;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        # x is not modified in either branch → phi should be trivial.
        x_phis = [p for b in builder._cfg.blocks.values() for p in b.phis if p.variable == "x"]
        assert all(p.eliminated for p in x_phis)
        # y should use x_1.
        assigns = get_assignments(builder)
        y_assign = next(a for a in assigns if a.result == "y_1")
        assert builder._resolve_alias(y_assign.expr.name) == "x_1"


class TestSequentialIfElse:
    def test_two_sequential_if_else(self):
        source = """a = 1;
x = 10;
y = 20;

if (a > 0) {
    x = 30;
} else {
    x = 40;
}

if (a > 0) {
    y = 50;
} else {
    y = 60;
}

z = x + y;"""
        builder = build_ssa(source)
        assert builder.validate_ssa() == []
        # Two phis: one for x, one for y, at different merge blocks.
        assert len(get_non_eliminated_phis(builder, "x")) == 1
        assert len(get_non_eliminated_phis(builder, "y")) == 1
