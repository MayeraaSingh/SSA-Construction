"""CFG builder.

Transforms an AST (:class:`~src.frontend.ast_nodes.Program`) into a
:class:`~src.ir.control_flow_graph.ControlFlowGraph`.

The builder processes statements sequentially within a *current* block.
Straight-line assignments accumulate in the same block.  When a control-flow
construct (if-else or while) is encountered, the builder creates new blocks,
wires the edges, and returns the *continuation* block for subsequent
statements.
"""

from __future__ import annotations

from src.frontend.ast_nodes import (
    Program,
    Block,
    Assignment,
    IfStatement,
    WhileStatement,
)
from src.ir.basic_block import BasicBlock, TerminationType
from src.ir.control_flow_graph import CFGError, ControlFlowGraph
from src.ir.instructions import (
    AssignmentInstruction,
    ConditionalBranchInstruction,
    UnconditionalJumpInstruction,
)


class CFGBuilder:
    """Builds a :class:`ControlFlowGraph` from a program AST.

    The builder is stateless apart from the CFG it constructs, so each call
    to :meth:`build` produces a fresh graph.
    """

    def __init__(self) -> None:
        self._cfg = ControlFlowGraph()

    def build(self, program: Program) -> ControlFlowGraph:
        """Build the complete CFG from a :class:`Program` AST node.

        The resulting CFG has an entry block, an exit block, and all
        predecessor/successor edges wired.  Every block is marked
        ``filled`` and ``sealed`` because construction is complete.
        """
        entry = self._cfg.create_block()
        self._cfg.entry_block = entry

        current = entry
        for stmt in program.statements:
            current = self._build_statement(stmt, current)

        # Connect the final continuation block to the exit block.
        exit_block = self._cfg.create_block()
        self._cfg.exit_block = exit_block
        self._add_jump(current, exit_block)
        exit_block.termination_type = TerminationType.EXIT

        # Mark all blocks as filled and sealed — construction is complete.
        for block in self._cfg.blocks.values():
            block.filled = True
            block.sealed = True

        return self._cfg

    # ------------------------------------------------------------------
    # Statement dispatch
    # ------------------------------------------------------------------

    def _build_statement(self, stmt: object, current: BasicBlock) -> BasicBlock:
        """Build CFG nodes for *stmt*, returning the continuation block.

        The *current* block is always open (no terminator) when passed in and
        the returned block is likewise open.
        """
        if isinstance(stmt, Assignment):
            current.add_instruction(AssignmentInstruction(stmt.target, stmt.expr))
            return current

        if isinstance(stmt, IfStatement):
            return self._build_if(stmt, current)

        if isinstance(stmt, WhileStatement):
            return self._build_while(stmt, current)

        if isinstance(stmt, Block):
            for s in stmt.statements:
                current = self._build_statement(s, current)
            return current

        raise CFGError(f"unknown statement type: {type(stmt).__name__}")

    # ------------------------------------------------------------------
    # If-else construction
    # ------------------------------------------------------------------

    def _build_if(self, stmt: IfStatement, current: BasicBlock) -> BasicBlock:
        """Build CFG for an if-else statement.

        Structure::

            current ──(cond)──► true_block ──► merge
                   └──────────► false_block ──► merge

        Returns the merge block as the continuation.
        """
        true_block = self._cfg.create_block()
        false_block = self._cfg.create_block()
        merge_block = self._cfg.create_block()

        # Current block evaluates the condition and branches.
        current.add_instruction(
            ConditionalBranchInstruction(
                stmt.condition, true_block.block_id, false_block.block_id
            )
        )
        current.termination_type = TerminationType.CONDITIONAL
        current.condition = stmt.condition
        self._cfg.add_edge(current, true_block)
        self._cfg.add_edge(current, false_block)

        # Build the then-branch, then jump to merge.
        t = true_block
        for s in stmt.then_block.statements:
            t = self._build_statement(s, t)
        self._add_jump(t, merge_block)

        # Build the else-branch, then jump to merge.
        f = false_block
        for s in stmt.else_block.statements:
            f = self._build_statement(s, f)
        self._add_jump(f, merge_block)

        return merge_block

    # ------------------------------------------------------------------
    # While-loop construction
    # ------------------------------------------------------------------

    def _build_while(self, stmt: WhileStatement, current: BasicBlock) -> BasicBlock:
        """Build CFG for a while loop.

        Structure::

            current ──► header ──(cond)──► body ──► header  (backedge)
                        header ──(cond)──► after

        The header is a distinct block so that it can later host phi
        functions.  Returns the *after* block as the continuation.
        """
        header = self._cfg.create_block()
        body = self._cfg.create_block()
        after = self._cfg.create_block()

        # Current block jumps unconditionally to the header.
        self._add_jump(current, header)

        # Header evaluates the condition and branches.
        header.add_instruction(
            ConditionalBranchInstruction(
                stmt.condition, body.block_id, after.block_id
            )
        )
        header.termination_type = TerminationType.CONDITIONAL
        header.condition = stmt.condition
        self._cfg.add_edge(header, body)
        self._cfg.add_edge(header, after)

        # Build the loop body, then add the backedge to the header.
        b = body
        for s in stmt.body.statements:
            b = self._build_statement(s, b)
        self._add_jump(b, header)

        return after

    # ------------------------------------------------------------------
    # Edge helper
    # ------------------------------------------------------------------

    def _add_jump(self, src: BasicBlock, dst: BasicBlock) -> None:
        """Add an unconditional jump from *src* to *dst*.

        Appends a :class:`UnconditionalJumpInstruction`, sets the termination
        type to ``JUMP``, and creates the CFG edge.
        """
        src.add_instruction(UnconditionalJumpInstruction(dst.block_id))
        src.termination_type = TerminationType.JUMP
        self._cfg.add_edge(src, dst)


def build_cfg(program: Program) -> ControlFlowGraph:
    """Convenience function: build a CFG from a program AST."""
    return CFGBuilder().build(program)
