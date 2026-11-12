"""Tests for control flow graph construction.

Covers straight-line programs, entry/exit blocks, conditional branch and
merge construction, while-loop backedges, nested control flow,
predecessor/successor consistency, and CFG validation.
"""

from __future__ import annotations

import pytest

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.ir.basic_block import BasicBlock, TerminationType
from src.ir.cfg_builder import CFGBuilder
from src.ir.control_flow_graph import ControlFlowGraph
from src.ir.instructions import (
    AssignmentInstruction,
    ConditionalBranchInstruction,
    UnconditionalJumpInstruction,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def build_cfg(source: str) -> ControlFlowGraph:
    """Lex, parse, and build a CFG from *source*."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return CFGBuilder().build(ast)


# ---------------------------------------------------------------------------
# Straight-line programs
# ---------------------------------------------------------------------------


class TestStraightLine:
    def test_assignments_in_single_block(self):
        """Sequential assignments must not create a new block for each one."""
        cfg = build_cfg("x = 10;\ny = x + 5;\nz = x * y - 2;")
        entry = cfg.entry_block
        assignments = [
            i for i in entry.instructions if isinstance(i, AssignmentInstruction)
        ]
        assert len(assignments) == 3
        # The only other instruction should be the jump to exit.
        assert len(entry.instructions) == 4

    def test_entry_and_exit_exist(self):
        cfg = build_cfg("x = 1;")
        assert cfg.entry_block is not None
        assert cfg.exit_block is not None
        assert cfg.entry_block.block_id != cfg.exit_block.block_id

    def test_entry_connected_to_exit(self):
        cfg = build_cfg("x = 1;")
        entry = cfg.entry_block
        exit_block = cfg.exit_block
        assert exit_block.block_id in entry.successors
        assert entry.block_id in exit_block.predecessors
        assert exit_block.termination_type == TerminationType.EXIT

    def test_exit_has_no_successors(self):
        cfg = build_cfg("x = 1;\ny = 2;")
        assert len(cfg.exit_block.successors) == 0

    def test_empty_program(self):
        cfg = build_cfg("")
        assert cfg.entry_block is not None
        assert cfg.exit_block is not None
        assert cfg.entry_block.termination_type == TerminationType.JUMP
        assert cfg.exit_block.block_id in cfg.entry_block.successors


# ---------------------------------------------------------------------------
# Conditional branches
# ---------------------------------------------------------------------------


class TestConditionalBranch:
    def test_if_else_creates_five_blocks(self):
        """Entry (cond), true, false, merge, exit."""
        cfg = build_cfg("if (x > 5) { y = 20; } else { y = 30; }")
        assert len(cfg.blocks) == 5

    def test_entry_has_conditional_branch(self):
        cfg = build_cfg("if (x > 5) { y = 20; } else { y = 30; }")
        entry = cfg.entry_block
        assert entry.termination_type == TerminationType.CONDITIONAL
        assert len(entry.successors) == 2

    def test_both_branches_merge_to_same_block(self):
        cfg = build_cfg("if (x > 5) { y = 20; } else { y = 30; }")
        entry = cfg.entry_block
        true_block = cfg.get_block(entry.successors[0])
        false_block = cfg.get_block(entry.successors[1])
        assert len(true_block.successors) == 1
        assert len(false_block.successors) == 1
        assert true_block.successors[0] == false_block.successors[0]

    def test_merge_block_has_two_predecessors(self):
        cfg = build_cfg("if (x > 5) { y = 20; } else { y = 30; }")
        entry = cfg.entry_block
        true_block = cfg.get_block(entry.successors[0])
        merge_id = true_block.successors[0]
        merge = cfg.get_block(merge_id)
        assert len(merge.predecessors) == 2

    def test_merge_is_continuation(self):
        """Statements after an if-else should go into the merge block."""
        source = "if (x > 5) { y = 20; } else { y = 30; }\nz = y + 5;"
        cfg = build_cfg(source)
        merge_candidates = [
            b
            for b in cfg.blocks.values()
            if len(b.predecessors) == 2
            and any(
                isinstance(i, AssignmentInstruction) and i.target == "z"
                for i in b.instructions
            )
        ]
        assert len(merge_candidates) == 1
        merge = merge_candidates[0]
        assert cfg.exit_block.block_id in merge.successors

    def test_branch_destinations_match_instruction(self):
        cfg = build_cfg("if (x > 5) { y = 1; } else { y = 2; }")
        entry = cfg.entry_block
        cond = next(
            i for i in entry.instructions if isinstance(i, ConditionalBranchInstruction)
        )
        assert cond.true_dest in entry.successors
        assert cond.false_dest in entry.successors
        assert cond.true_dest != cond.false_dest

    def test_true_branch_contains_assignment(self):
        cfg = build_cfg("if (x > 5) { y = 20; } else { y = 30; }")
        entry = cfg.entry_block
        true_block = cfg.get_block(entry.successors[0])
        assigns = [i for i in true_block.instructions if isinstance(i, AssignmentInstruction)]
        assert len(assigns) == 1
        assert assigns[0].target == "y"
        assert assigns[0].expr.value == 20

    def test_false_branch_contains_assignment(self):
        cfg = build_cfg("if (x > 5) { y = 20; } else { y = 30; }")
        entry = cfg.entry_block
        false_block = cfg.get_block(entry.successors[1])
        assigns = [i for i in false_block.instructions if isinstance(i, AssignmentInstruction)]
        assert len(assigns) == 1
        assert assigns[0].expr.value == 30


# ---------------------------------------------------------------------------
# While loops
# ---------------------------------------------------------------------------


class TestWhileLoop:
    def test_while_creates_five_blocks(self):
        """Entry, header, body, after, exit."""
        cfg = build_cfg("while (x < 5) { x = x + 1; }")
        assert len(cfg.blocks) == 5

    def test_header_is_conditional(self):
        cfg = build_cfg("while (x < 5) { x = x + 1; }")
        headers = [
            b for b in cfg.blocks.values()
            if b.termination_type == TerminationType.CONDITIONAL
        ]
        assert len(headers) == 1
        header = headers[0]
        assert len(header.successors) == 2

    def test_header_has_two_predecessors(self):
        """The header receives an edge from the entry and a backedge from the body."""
        cfg = build_cfg("while (x < 5) { x = x + 1; }")
        header = next(
            b for b in cfg.blocks.values()
            if b.termination_type == TerminationType.CONDITIONAL
        )
        assert len(header.predecessors) == 2

    def test_backedge_exists(self):
        """One of the header's predecessors must also be a successor (the backedge)."""
        cfg = build_cfg("while (x < 5) { x = x + 1; }")
        header = next(
            b for b in cfg.blocks.values()
            if b.termination_type == TerminationType.CONDITIONAL
        )
        backedge_blocks = [p for p in header.predecessors if p in header.successors]
        assert len(backedge_blocks) == 1

    def test_body_jumps_back_to_header(self):
        cfg = build_cfg("while (x < 5) { x = x + 1; }")
        header = next(
            b for b in cfg.blocks.values()
            if b.termination_type == TerminationType.CONDITIONAL
        )
        body_candidates = [
            b
            for b in cfg.blocks.values()
            if header.block_id in b.successors and b.block_id != cfg.entry_block.block_id
        ]
        assert len(body_candidates) == 1
        body = body_candidates[0]
        assert body.termination_type == TerminationType.JUMP

    def test_body_contains_assignment(self):
        cfg = build_cfg("while (x < 5) { x = x + 1; }")
        header = next(
            b for b in cfg.blocks.values()
            if b.termination_type == TerminationType.CONDITIONAL
        )
        # The body is both a successor of the header (true edge) and has the
        # header as a successor (backedge).  The entry block jumps to the
        # header too, but is NOT a successor of the header.
        body_candidates = [
            b for b in cfg.blocks.values()
            if header.block_id in b.successors and b.block_id in header.successors
        ]
        assert len(body_candidates) == 1
        body = body_candidates[0]
        assigns = [i for i in body.instructions if isinstance(i, AssignmentInstruction)]
        assert len(assigns) == 1

    def test_after_block_is_continuation(self):
        source = "while (x < 5) { x = x + 1; }\ny = x;"
        cfg = build_cfg(source)
        header = next(
            b for b in cfg.blocks.values()
            if b.termination_type == TerminationType.CONDITIONAL
        )
        # The "after" block is the successor that does NOT loop back.
        after_candidates = [
            cfg.get_block(s) for s in header.successors
            if header.block_id not in cfg.get_block(s).successors
        ]
        assert len(after_candidates) == 1
        after = after_candidates[0]
        # It should contain the y = x assignment.
        assigns = [i for i in after.instructions if isinstance(i, AssignmentInstruction)]
        assert len(assigns) == 1
        assert assigns[0].target == "y"


# ---------------------------------------------------------------------------
# Nested control flow
# ---------------------------------------------------------------------------


class TestNestedControlFlow:
    def test_nested_if_validates(self):
        source = """if (x > 0) {
    if (x > 5) { y = 10; } else { y = 20; }
} else {
    y = 30;
}
z = y + 1;"""
        cfg = build_cfg(source)
        errors = cfg.validate()
        assert errors == []

    def test_if_inside_while_validates(self):
        source = """while (x < 10) {
    if (x > 5) { y = 1; } else { y = 2; }
    x = x + 1;
}"""
        cfg = build_cfg(source)
        errors = cfg.validate()
        assert errors == []

    def test_while_inside_if_validates(self):
        source = """if (x > 0) {
    while (x < 10) { x = x + 1; }
} else {
    y = 1;
}"""
        cfg = build_cfg(source)
        errors = cfg.validate()
        assert errors == []

    def test_nested_while_validates(self):
        source = "while (x < 5) { while (y < 3) { y = y + 1; } x = x + 1; }"
        cfg = build_cfg(source)
        errors = cfg.validate()
        assert errors == []

    def test_sequential_if_else_validates(self):
        source = """if (x > 1) { y = 1; } else { y = 2; }
if (x > 2) { z = 1; } else { z = 2; }
w = y + z;"""
        cfg = build_cfg(source)
        errors = cfg.validate()
        assert errors == []

    def test_nested_if_creates_multiple_merge_points(self):
        source = """if (x > 0) {
    if (x > 5) { y = 10; } else { y = 20; }
} else {
    y = 30;
}"""
        cfg = build_cfg(source)
        merge_blocks = [b for b in cfg.blocks.values() if len(b.predecessors) == 2]
        # Inner if creates one merge; outer if creates another.
        assert len(merge_blocks) == 2

    def test_complex_program_validates(self):
        source = """x = 0;
while (x < 5) {
    if (x > 2) { y = 1; } else { y = 2; }
    x = x + 1;
}
z = y;"""
        cfg = build_cfg(source)
        errors = cfg.validate()
        assert errors == []


# ---------------------------------------------------------------------------
# Predecessor / successor consistency
# ---------------------------------------------------------------------------


class TestPredecessorSuccessorConsistency:
    def test_all_edges_bidirectional(self):
        source = """if (x > 5) { y = 1; } else { y = 2; }
while (x < 10) { x = x + 1; }
z = x + y;"""
        cfg = build_cfg(source)
        for block in cfg.blocks.values():
            for succ_id in block.successors:
                succ = cfg.get_block(succ_id)
                assert block.block_id in succ.predecessors
            for pred_id in block.predecessors:
                pred = cfg.get_block(pred_id)
                assert block.block_id in pred.successors

    def test_ordered_predecessor_lists(self):
        """Predecessors should be in insertion order (not sorted)."""
        source = "if (x > 5) { y = 1; } else { y = 2; }"
        cfg = build_cfg(source)
        entry = cfg.entry_block
        true_block = cfg.get_block(entry.successors[0])
        merge_id = true_block.successors[0]
        merge = cfg.get_block(merge_id)
        # Should have exactly 2 predecessors.
        assert len(merge.predecessors) == 2


# ---------------------------------------------------------------------------
# CFG validation
# ---------------------------------------------------------------------------


class TestCFGValidation:
    def test_valid_cfg_no_errors(self):
        cfg = build_cfg("x = 1;\ny = 2;")
        errors = cfg.validate()
        assert errors == []

    def test_conditional_example_validates(self):
        source = """x = 10;

if (x > 5) {
    y = 20;
} else {
    y = 30;
}

z = y + 5;"""
        cfg = build_cfg(source)
        assert cfg.validate() == []

    def test_while_example_validates(self):
        source = """x = 0;

while (x < 5) {
    x = x + 1;
}

y = x;"""
        cfg = build_cfg(source)
        assert cfg.validate() == []

    def test_nested_if_example_validates(self):
        source = """x = 1;

if (x > 0) {
    if (x > 5) {
        y = 10;
    } else {
        y = 20;
    }
} else {
    y = 30;
}

z = y + 1;"""
        cfg = build_cfg(source)
        assert cfg.validate() == []

    def test_all_examples_valid(self):
        """All four example programs should produce valid CFGs."""
        import pathlib

        examples_dir = pathlib.Path(__file__).parent.parent / "examples"
        for example_file in sorted(examples_dir.glob("*.c")):
            source = example_file.read_text(encoding="utf-8")
            cfg = build_cfg(source)
            errors = cfg.validate()
            assert errors == [], f"{example_file.name}: {errors}"


# ---------------------------------------------------------------------------
# SSA lifecycle flags
# ---------------------------------------------------------------------------


class TestSSALifecycleFlags:
    def test_filled_and_sealed_after_build(self):
        cfg = build_cfg("x = 1;")
        for block in cfg.blocks.values():
            assert block.filled is True
            assert block.sealed is True

    def test_filled_and_sealed_complex(self):
        source = """while (x < 5) {
    if (x > 2) { y = 1; } else { y = 2; }
    x = x + 1;
}
z = y;"""
        cfg = build_cfg(source)
        for block in cfg.blocks.values():
            assert block.filled is True
            assert block.sealed is True

    def test_predecessor_successor_no_duplicates(self):
        """add_edge should not create duplicate entries."""
        cfg = build_cfg("if (x > 5) { y = 1; } else { y = 2; }")
        for block in cfg.blocks.values():
            assert len(block.successors) == len(set(block.successors))
            assert len(block.predecessors) == len(set(block.predecessors))
