"""Lazy SSA construction using sealed blocks.

Implements the core SSA construction algorithm described in:

    "Simple and Efficient Construction of Static Single Assignment Form"

The algorithm is *lazy* in the sense that phi functions are only inserted when
a variable is actually read at a merge point, and *sealed-block* in the sense
that a block's phi operands are only resolved once the block is sealed (all
predecessors known).

Architecture
============

The :class:`SSABuilder` integrates CFG construction with SSA conversion in a
single pass.  This is necessary because the lazy algorithm requires
incremental predecessor availability:

1.  A block is created and begins accumulating SSA instructions.
2.  While the block is being filled, ``read_variable`` may encounter a
    variable with no local definition.  If the block is unsealed (not all
    predecessors known yet — typically a loop header whose backedge is
    pending), an **incomplete phi** is created and its result is immediately
    recorded as the current definition.
3.  Once all predecessors are registered, the block is **sealed** via
    :meth:`seal_block`.  Incomplete phi functions are then resolved by reading
    the variable from each predecessor.

For if-else constructs, both branch bodies are fully processed (and sealed)
before the merge block is sealed.  For while loops, the loop header is
processed while unsealed (creating incomplete phis for loop-carried
variables), the body is processed and sealed, the backedge is added, and only
then is the header sealed — resolving the incomplete phis with operands from
both the entry path and the backedge.

This ordering is what makes the algorithm handle cyclic control flow
correctly without computing dominance frontiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.frontend.ast_nodes import (
    Assignment,
    BinaryExpression,
    Block,
    ComparisonExpression,
    Identifier,
    IfStatement,
    IntegerLiteral,
    Program,
    UnaryExpression,
    WhileStatement,
)
from src.ir.basic_block import BasicBlock, TerminationType
from src.ir.control_flow_graph import CFGError, ControlFlowGraph
from src.ir.instructions import (
    AssignmentInstruction,
    ConditionalBranchInstruction,
    UnconditionalJumpInstruction,
)
from src.ssa.phi_functions import PhiFunction
from src.ssa.ssa_simplifier import (
    eliminate_trivial_phi,
    find_trivial_replacement,
    resolve_alias,
)


class SSAError(Exception):
    """Raised when SSA construction or validation fails."""


# ---------------------------------------------------------------------------
# SSA expression and instruction types
# ---------------------------------------------------------------------------


@dataclass
class SSAValue:
    """A reference to an SSA-named variable (e.g. ``x_2``)."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class SSAIntegerLiteral:
    """An integer literal in SSA form."""

    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class SSABinaryExpression:
    """A binary arithmetic expression in SSA form."""

    op: str
    left: object
    right: object

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class SSAUnaryExpression:
    """A unary expression in SSA form."""

    op: str
    operand: object

    def __str__(self) -> str:
        return f"({self.op}{self.operand})"


@dataclass
class SSAComparisonExpression:
    """A comparison expression in SSA form."""

    op: str
    left: object
    right: object

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class SSAAssignment:
    """An SSA assignment instruction: ``result = expr``."""

    result: str
    expr: object


@dataclass
class SSAConditionalBranch:
    """An SSA conditional branch instruction."""

    condition: object  # SSAComparisonExpression
    true_dest: int
    false_dest: int


@dataclass
class SSAUnconditionalJump:
    """An SSA unconditional jump instruction."""

    dest: int


# ---------------------------------------------------------------------------
# SSA Builder
# ---------------------------------------------------------------------------


class SSABuilder:
    """Lazy SSA construction engine using the sealed-block algorithm.

    Call :meth:`build` to construct a CFG with SSA form from a program AST.
    For unit-testing individual methods, pass an existing CFG to the
    constructor and call :meth:`read_variable`, :meth:`write_variable`,
    :meth:`seal_block`, etc. directly.
    """

    def __init__(self, cfg: ControlFlowGraph | None = None) -> None:
        self._cfg = cfg if cfg is not None else ControlFlowGraph()
        self._counters: dict[str, int] = {}
        self._aliases: dict[str, str] = {}
        self._definitions: set[str] = set()

    # ------------------------------------------------------------------
    # Variable version management
    # ------------------------------------------------------------------

    def new_ssa_value(self, name: str) -> str:
        """Allocate and return a unique SSA value name for source variable *name*.

        Versioning is deterministic and monotonically increasing per variable.
        """
        count = self._counters.get(name, 0) + 1
        self._counters[name] = count
        ssa_name = f"{name}_{count}"
        self._definitions.add(ssa_name)
        return ssa_name

    # ------------------------------------------------------------------
    # Core read / write operations
    # ------------------------------------------------------------------

    def write_variable(self, name: str, block: BasicBlock, value: str) -> None:
        """Record a new definition of *name* in *block*.

        Subsequent reads of *name* in *block* will resolve to *value*.
        """
        block.current_defs[name] = value

    def read_variable(self, name: str, block: BasicBlock) -> str:
        """Resolve the current SSA value of *name* in *block*.

        This is the central operation of the lazy algorithm.

        1.  If *block* has a local definition, return it (resolved through
            the alias map).
        2.  If *block* is unsealed, create an incomplete phi, record its result
            as the current definition, and return it.  Operands will be
            resolved when the block is sealed.
        3.  If *block* is sealed with exactly one predecessor, recursively
            read from that predecessor and cache the result.
        4.  If *block* is sealed with multiple predecessors, create a phi,
            record its result **before** reading predecessors (to break
            cycles on backedges), read from each predecessor, populate
            operands, and attempt trivial-phi elimination.
        5.  If *block* is sealed with no predecessors and no local definition,
            raise an :class:`SSAError` for an undefined variable.
        """
        # 1. Local definition exists.
        if name in block.current_defs:
            return self._resolve_alias(block.current_defs[name])

        # 2. Unsealed block: create incomplete phi.
        if not block.sealed:
            phi = self.add_phi(name, block)
            return phi.result

        preds = block.predecessors

        # 5. No predecessors and no local definition: undefined variable.
        if len(preds) == 0:
            raise SSAError(
                f"undefined variable '{name}' read in block {block.block_id}"
            )

        # 3. Single predecessor: recursively read and cache.
        if len(preds) == 1:
            pred = self._cfg.get_block(preds[0])
            value = self.read_variable(name, pred)
            block.current_defs[name] = value
            return self._resolve_alias(value)

        # 4. Multiple predecessors: create phi, then read operands.
        phi = self.add_phi(name, block)
        # Record the phi result BEFORE reading predecessors — this breaks
        # infinite recursion when a predecessor reads back through this block
        # (loop backedge).
        for pred_id in preds:
            pred = self._cfg.get_block(pred_id)
            operand = self.read_variable(name, pred)
            phi.operands[pred_id] = operand
        return self.try_remove_trivial_phi(phi)

    def add_phi(self, name: str, block: BasicBlock) -> PhiFunction:
        """Create a phi function for *name* in *block*.

        Allocates a unique SSA version, creates the :class:`PhiFunction`,
        appends it to the block's phi list, and records the result as the
        current definition.  Operands are left empty (incomplete) — they are
        resolved later by :meth:`seal_block` or by :meth:`read_variable`
        directly.
        """
        result = self.new_ssa_value(name)
        phi = PhiFunction(result=result, variable=name, block_id=block.block_id)
        block.phis.append(phi)
        block.current_defs[name] = result
        return phi

    # ------------------------------------------------------------------
    # Block sealing
    # ------------------------------------------------------------------

    def seal_block(self, block: BasicBlock) -> None:
        """Seal *block* and resolve all incomplete phi functions.

        Once sealed, no more predecessors may be added.  Every incomplete phi
        (one with empty operands) is resolved by reading its source variable
        from each predecessor, then trivial-phi elimination is attempted.
        """
        if block.sealed:
            return
        block.sealed = True

        # Resolve each incomplete phi (those with empty operands).
        for phi in block.phis:
            if len(phi.operands) == 0:
                for pred_id in block.predecessors:
                    pred = self._cfg.get_block(pred_id)
                    operand = self.read_variable(phi.variable, pred)
                    phi.operands[pred_id] = operand
                self.try_remove_trivial_phi(phi)

    # ------------------------------------------------------------------
    # Trivial phi elimination
    # ------------------------------------------------------------------

    def try_remove_trivial_phi(self, phi: PhiFunction) -> str:
        """Attempt to eliminate *phi* if it is trivial.

        A phi is trivial when all non-self operands resolve to the same value.
        If trivial, the phi's result is aliased to the surviving value, the
        phi is marked eliminated, and the surviving value is returned.
        Otherwise the phi's (aliased) result is returned unchanged.
        """
        replacement = eliminate_trivial_phi(phi, self._aliases)
        if replacement is not None:
            return replacement
        return self._resolve_alias(phi.result)

    def _resolve_alias(self, value: str) -> str:
        """Follow the alias chain to find the ultimate replacement for *value*."""
        return resolve_alias(value, self._aliases)

    def _finalize(self) -> None:
        """Propagate aliases and simplify newly-trivial phis after construction.

        Some phi functions may become trivial only after another phi they
        reference is eliminated.  This pass iterates until no more changes
        occur, resolving all aliases in phi operands and re-checking for
        triviality.
        """
        changed = True
        while changed:
            changed = False
            for block in self._cfg.blocks.values():
                for phi in block.phis:
                    if phi.eliminated:
                        continue
                    # Resolve operands through the current alias map.
                    resolved_ops: dict[int, str] = {}
                    for pred_id, operand in phi.operands.items():
                        resolved_ops[pred_id] = self._resolve_alias(operand)
                    phi.operands = resolved_ops
                    # Re-check triviality.
                    replacement = eliminate_trivial_phi(phi, self._aliases)
                    if replacement is not None:
                        changed = True

    # ------------------------------------------------------------------
    # Expression conversion (AST → SSA)
    # ------------------------------------------------------------------

    def _convert_expr(self, expr: object, block: BasicBlock) -> object:
        """Convert an AST expression node to an SSA expression node.

        :class:`Identifier` nodes are resolved via :meth:`read_variable`.
        All other node types are recursively rewritten, preserving operator
        precedence and structure.
        """
        if isinstance(expr, IntegerLiteral):
            return SSAIntegerLiteral(expr.value)
        if isinstance(expr, Identifier):
            ssa_name = self.read_variable(expr.name, block)
            return SSAValue(ssa_name)
        if isinstance(expr, BinaryExpression):
            left = self._convert_expr(expr.left, block)
            right = self._convert_expr(expr.right, block)
            return SSABinaryExpression(expr.op, left, right)
        if isinstance(expr, UnaryExpression):
            operand = self._convert_expr(expr.operand, block)
            return SSAUnaryExpression(expr.op, operand)
        raise SSAError(f"cannot convert expression of type {type(expr).__name__}")

    def _convert_condition(
        self, condition: object, block: BasicBlock
    ) -> object:
        """Convert a comparison condition to SSA form."""
        if isinstance(condition, ComparisonExpression):
            left = self._convert_expr(condition.left, block)
            right = self._convert_expr(condition.right, block)
            return SSAComparisonExpression(condition.op, left, right)
        raise SSAError(
            f"expected comparison expression, got {type(condition).__name__}"
        )

    # ------------------------------------------------------------------
    # Integrated CFG + SSA construction
    # ------------------------------------------------------------------

    def build(self, program: Program) -> ControlFlowGraph:
        """Build a CFG with SSA form from a program AST.

        The construction is integrated: CFG blocks and edges are created at
        the same time as SSA instructions, so that block sealing happens at
        the correct points (particularly for loop headers).

        Returns:
            The :class:`ControlFlowGraph` with SSA instructions and phi
            functions populated.
        """
        # Always start with a fresh CFG.
        self._cfg = ControlFlowGraph()

        entry = self._cfg.create_block()
        self._cfg.entry_block = entry
        # Entry has no predecessors — seal immediately.
        self.seal_block(entry)

        current = entry
        for stmt in program.statements:
            current = self._build_statement(stmt, current)

        # Seal the final continuation block.
        self._seal_if_needed(current)

        # Create exit block and connect.
        exit_block = self._cfg.create_block()
        self._cfg.exit_block = exit_block
        self._add_ssa_jump(current, exit_block)
        exit_block.termination_type = TerminationType.EXIT
        exit_block.filled = True
        self.seal_block(exit_block)

        # Propagate aliases and simplify.
        self._finalize()

        return self._cfg

    def _build_statement(self, stmt: object, current: BasicBlock) -> BasicBlock:
        """Build SSA IR for a single statement, returning the continuation block."""
        # Seal current block before processing (all predecessors should be
        # known at this point, except for loop headers handled by _build_while).
        self._seal_if_needed(current)

        if isinstance(stmt, Assignment):
            return self._build_assignment(stmt, current)
        if isinstance(stmt, IfStatement):
            return self._build_if(stmt, current)
        if isinstance(stmt, WhileStatement):
            return self._build_while(stmt, current)
        if isinstance(stmt, Block):
            for s in stmt.statements:
                current = self._build_statement(s, current)
            return current
        raise SSAError(f"unknown statement type: {type(stmt).__name__}")

    def _build_assignment(self, stmt: Assignment, current: BasicBlock) -> BasicBlock:
        """Convert an assignment to SSA form and append it to *current*."""
        ssa_expr = self._convert_expr(stmt.expr, current)
        result = self.new_ssa_value(stmt.target)
        current.add_ssa_instruction(SSAAssignment(result, ssa_expr))
        # Also add Day 1 IR instruction for structural validation.
        current.add_instruction(AssignmentInstruction(stmt.target, stmt.expr))
        self.write_variable(stmt.target, current, result)
        return current

    def _build_if(self, stmt: IfStatement, current: BasicBlock) -> BasicBlock:
        """Build SSA IR for an if-else statement.

        Creates true/false/merge blocks.  Both branch bodies are processed
        and sealed before the merge block is returned (unsealed — it will be
        sealed by the next statement that uses it).
        """
        true_block = self._cfg.create_block()
        false_block = self._cfg.create_block()
        merge_block = self._cfg.create_block()

        # Convert condition in the (already sealed) current block.
        ssa_condition = self._convert_condition(stmt.condition, current)
        current.add_ssa_instruction(
            SSAConditionalBranch(ssa_condition, true_block.block_id, false_block.block_id)
        )
        current.add_instruction(
            ConditionalBranchInstruction(
                stmt.condition, true_block.block_id, false_block.block_id
            )
        )
        current.termination_type = TerminationType.CONDITIONAL
        current.condition = stmt.condition
        current.filled = True
        self._cfg.add_edge(current, true_block)
        self._cfg.add_edge(current, false_block)

        # Build then-branch.
        t = self._build_statements(stmt.then_block.statements, true_block)
        self._seal_if_needed(t)
        self._add_ssa_jump(t, merge_block)

        # Build else-branch.
        f = self._build_statements(stmt.else_block.statements, false_block)
        self._seal_if_needed(f)
        self._add_ssa_jump(f, merge_block)

        # Merge block is returned unsealed — sealed by the next _build_statement.
        return merge_block

    def _build_while(self, stmt: WhileStatement, current: BasicBlock) -> BasicBlock:
        """Build SSA IR for a while loop.

        The loop header is processed while **unsealed** (the backedge from
        the body has not been added yet).  This causes loop-carried variable
        reads to create incomplete phi functions.  After the body is
        processed and the backedge is added, the header is sealed, resolving
        the incomplete phis with operands from both the entry path and the
        backedge.
        """
        header = self._cfg.create_block()
        body = self._cfg.create_block()
        after = self._cfg.create_block()

        # Current block jumps to the header.
        self._add_ssa_jump(current, header)

        # Process header's condition while it is UNSEALED.
        # This creates incomplete phis for loop-carried variables.
        ssa_condition = self._convert_condition(stmt.condition, header)
        header.add_ssa_instruction(
            SSAConditionalBranch(ssa_condition, body.block_id, after.block_id)
        )
        header.add_instruction(
            ConditionalBranchInstruction(
                stmt.condition, body.block_id, after.block_id
            )
        )
        header.termination_type = TerminationType.CONDITIONAL
        header.condition = stmt.condition
        header.filled = True
        self._cfg.add_edge(header, body)
        self._cfg.add_edge(header, after)

        # Build the loop body.
        b = self._build_statements(stmt.body.statements, body)
        self._seal_if_needed(b)

        # Add backedge: body (or its continuation) jumps to header.
        self._add_ssa_jump(b, header)

        # NOW seal the header — all predecessors are known.
        self.seal_block(header)

        # 'after' is returned unsealed — sealed by the next _build_statement.
        return after

    def _build_statements(
        self, statements: list, block: BasicBlock
    ) -> BasicBlock:
        """Process a list of statements starting from *block*.

        Seals *block* via the first :meth:`_build_statement` call (or
        explicitly if the list is empty).  Returns the final continuation
        block.
        """
        current = block
        for stmt in statements:
            current = self._build_statement(stmt, current)
        return current

    def _seal_if_needed(self, block: BasicBlock) -> None:
        """Seal *block* if it has not been sealed yet."""
        if not block.sealed:
            self.seal_block(block)

    def _add_ssa_jump(self, src: BasicBlock, dst: BasicBlock) -> None:
        """Add an unconditional SSA jump from *src* to *dst*.

        Appends both an SSA jump instruction and a Day 1 IR jump instruction
        (for structural validation), sets the termination type, and creates
        the CFG edge.
        """
        src.add_ssa_instruction(SSAUnconditionalJump(dst.block_id))
        src.add_instruction(UnconditionalJumpInstruction(dst.block_id))
        src.termination_type = TerminationType.JUMP
        src.filled = True
        self._cfg.add_edge(src, dst)

    # ------------------------------------------------------------------
    # SSA text rendering
    # ------------------------------------------------------------------

    def to_ssa_text(self) -> str:
        """Produce a human-readable text representation of the SSA IR.

        Phi functions are listed first (non-eliminated ones), followed by
        SSA instructions.  All SSA value names are resolved through the alias
        map so that eliminated trivial phis are transparent.
        """
        lines: list[str] = []
        lines.append("=== SSA Intermediate Representation ===")

        for block_id in sorted(self._cfg.blocks.keys()):
            block = self._cfg.blocks[block_id]
            lines.append(f"--- Block {block_id} ---")
            lines.append(
                f"  Preds: {block.predecessors}  Succs: {block.successors}"
            )

            # Render non-eliminated phi functions.
            has_content = False
            for phi in block.phis:
                if not phi.eliminated:
                    ops = ", ".join(
                        f"{self._resolve_alias(v)} [B{k}]"
                        for k, v in phi.operands.items()
                    )
                    lines.append(f"  {phi.result} = \u03c6({ops})")
                    has_content = True

            # Render SSA instructions.
            for inst in block.ssa_instructions:
                if isinstance(inst, SSAAssignment):
                    lines.append(f"  {inst.result} = {self._render_ssa_expr(inst.expr)}")
                    has_content = True
                elif isinstance(inst, SSAConditionalBranch):
                    lines.append(
                        f"  BRANCH if {self._render_ssa_expr(inst.condition)}"
                        f" -> B{inst.true_dest} else -> B{inst.false_dest}"
                    )
                    has_content = True
                elif isinstance(inst, SSAUnconditionalJump):
                    lines.append(f"  JUMP -> B{inst.dest}")
                    has_content = True

            if not has_content:
                lines.append("  (no instructions)")

            lines.append("")

        return "\n".join(lines)

    def _render_ssa_expr(self, expr: object) -> str:
        """Render an SSA expression node to a string, resolving aliases."""
        if isinstance(expr, SSAValue):
            return self._resolve_alias(expr.name)
        if isinstance(expr, SSAIntegerLiteral):
            return str(expr.value)
        if isinstance(expr, SSABinaryExpression):
            return f"({self._render_ssa_expr(expr.left)} {expr.op} {self._render_ssa_expr(expr.right)})"
        if isinstance(expr, SSAUnaryExpression):
            return f"({expr.op}{self._render_ssa_expr(expr.operand)})"
        if isinstance(expr, SSAComparisonExpression):
            return f"({self._render_ssa_expr(expr.left)} {expr.op} {self._render_ssa_expr(expr.right)})"
        return str(expr)

    # ------------------------------------------------------------------
    # SSA validation
    # ------------------------------------------------------------------

    def validate_ssa(self) -> list[str]:
        """Validate the generated SSA representation.

        Returns:
            A list of human-readable error messages.  An empty list means
            the SSA is valid.
        """
        errors: list[str] = []

        # Collect all defined SSA values (from assignments and non-eliminated phis).
        defined: set[str] = set()
        for block in self._cfg.blocks.values():
            for inst in block.ssa_instructions:
                if isinstance(inst, SSAAssignment):
                    defined.add(inst.result)
            for phi in block.phis:
                if not phi.eliminated:
                    defined.add(phi.result)
        # Eliminated phi results are aliased — their targets must be defined.
        for alias, target in self._aliases.items():
            resolved = self._resolve_alias(target)
            if resolved not in defined:
                errors.append(
                    f"alias {alias} -> {resolved} resolves to undefined value"
                )

        for block in self._cfg.blocks.values():
            bid = block.block_id

            # 8. All blocks must be sealed.
            if not block.sealed:
                errors.append(f"block {bid} is not sealed")

            # Check all SSA value uses in instructions.
            for inst in block.ssa_instructions:
                if isinstance(inst, SSAAssignment):
                    for u in self._collect_uses(inst.expr):
                        resolved = self._resolve_alias(u)
                        if resolved not in defined:
                            errors.append(
                                f"block {bid}: use of undefined SSA value {u}"
                            )
                elif isinstance(inst, SSAConditionalBranch):
                    for u in self._collect_uses(inst.condition):
                        resolved = self._resolve_alias(u)
                        if resolved not in defined:
                            errors.append(
                                f"block {bid}: branch uses undefined SSA value {u}"
                            )

            # Check phi functions.
            for phi in block.phis:
                if phi.eliminated:
                    continue

                # 3. One operand per predecessor.
                if len(phi.operands) != len(block.predecessors):
                    errors.append(
                        f"block {bid}: phi {phi.result} has {len(phi.operands)} "
                        f"operands but {len(block.predecessors)} predecessors"
                    )

                # 4. Operand block IDs must be valid predecessors.
                for pred_id in phi.operands:
                    if pred_id not in block.predecessors:
                        errors.append(
                            f"block {bid}: phi {phi.result} has operand from "
                            f"non-predecessor block {pred_id}"
                        )

                # 5. No incomplete phis.
                if len(phi.operands) == 0:
                    errors.append(
                        f"block {bid}: phi {phi.result} is incomplete (no operands)"
                    )

                # Check that operands are defined.
                for operand in phi.operands.values():
                    resolved = self._resolve_alias(operand)
                    if resolved not in defined:
                        errors.append(
                            f"block {bid}: phi {phi.result} uses undefined value {operand}"
                        )

        return errors

    def _collect_uses(self, expr: object) -> set[str]:
        """Collect all SSA value names used in an SSA expression."""
        uses: set[str] = set()
        if isinstance(expr, SSAValue):
            uses.add(expr.name)
        elif isinstance(expr, SSABinaryExpression):
            uses |= self._collect_uses(expr.left)
            uses |= self._collect_uses(expr.right)
        elif isinstance(expr, SSAUnaryExpression):
            uses |= self._collect_uses(expr.operand)
        elif isinstance(expr, SSAComparisonExpression):
            uses |= self._collect_uses(expr.left)
            uses |= self._collect_uses(expr.right)
        return uses


def build_ssa(program: Program) -> tuple[ControlFlowGraph, SSABuilder]:
    """Convenience: build SSA form from a program AST.

    Returns:
        A tuple of (CFG, SSABuilder) so callers can access the SSA text and
        validation methods.
    """
    builder = SSABuilder()
    cfg = builder.build(program)
    return cfg, builder
