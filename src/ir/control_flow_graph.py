"""Control Flow Graph (CFG).

Defines the :class:`ControlFlowGraph` data structure that maintains a
collection of basic blocks, unique block-ID generation, edge management with
predecessor/successor consistency, validation, and a text renderer.

The AST-to-CFG transformation logic lives in :mod:`src.ir.cfg_builder`.
"""

from __future__ import annotations

from typing import Any, Optional

from src.ir.basic_block import BasicBlock, TerminationType
from src.ir.instructions import (
    AssignmentInstruction,
    ConditionalBranchInstruction,
    UnconditionalJumpInstruction,
)


class CFGError(Exception):
    """Raised when CFG construction or validation fails."""


class ControlFlowGraph:
    """A control flow graph.

    Maintains a collection of basic blocks keyed by ID, along with entry and
    exit block references.  Block IDs are generated monotonically starting
    from 0.
    """

    def __init__(self) -> None:
        self._blocks: dict[int, BasicBlock] = {}
        self._next_id: int = 0
        self.entry_block: Optional[BasicBlock] = None
        self.exit_block: Optional[BasicBlock] = None

    # ------------------------------------------------------------------
    # Block / edge management
    # ------------------------------------------------------------------

    def create_block(self) -> BasicBlock:
        """Create and register a new basic block with a unique ID."""
        block = BasicBlock(block_id=self._next_id)
        self._blocks[self._next_id] = block
        self._next_id += 1
        return block

    def add_edge(self, src: BasicBlock, dst: BasicBlock) -> None:
        """Add a directed edge ``src -> dst``, maintaining predecessor/successor lists."""
        src.add_successor(dst.block_id)
        dst.add_predecessor(src.block_id)

    def get_block(self, block_id: int) -> BasicBlock:
        """Retrieve a block by ID.

        Raises:
            CFGError: If no block with *block_id* exists.
        """
        if block_id not in self._blocks:
            raise CFGError(f"block {block_id} does not exist")
        return self._blocks[block_id]

    @property
    def blocks(self) -> dict[int, BasicBlock]:
        """All blocks in the CFG, keyed by block ID."""
        return self._blocks

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the CFG structure.

        Returns:
            A list of human-readable error messages.  An empty list means the
            CFG is valid.
        """
        errors: list[str] = []

        # 6. Entry and exit blocks exist.
        if self.entry_block is None:
            errors.append("entry block is not set")
        if self.exit_block is None:
            errors.append("exit block is not set")

        # 4. Block IDs are unique (guaranteed by creation, but verify).
        ids = list(self._blocks.keys())
        if len(ids) != len(set(ids)):
            errors.append("duplicate block IDs detected")

        for block in self._blocks.values():
            bid = block.block_id

            # 3. All referenced blocks belong to the CFG.
            for succ_id in block.successors:
                if succ_id not in self._blocks:
                    errors.append(f"block {bid}: successor {succ_id} does not exist")
            for pred_id in block.predecessors:
                if pred_id not in self._blocks:
                    errors.append(f"block {bid}: predecessor {pred_id} does not exist")

            # 1. Every successor relationship has a corresponding predecessor.
            for succ_id in block.successors:
                succ = self._blocks.get(succ_id)
                if succ is not None and bid not in succ.predecessors:
                    errors.append(
                        f"block {bid}: successor {succ_id} does not list "
                        f"{bid} as a predecessor"
                    )

            # 2. Every predecessor relationship has a corresponding successor.
            for pred_id in block.predecessors:
                pred = self._blocks.get(pred_id)
                if pred is not None and bid not in pred.successors:
                    errors.append(
                        f"block {bid}: predecessor {pred_id} does not list "
                        f"{bid} as a successor"
                    )

            # 5. Conditional branches have valid outgoing destinations.
            cond_inst = next(
                (i for i in block.instructions if isinstance(i, ConditionalBranchInstruction)),
                None,
            )
            if cond_inst is not None:
                if len(block.successors) != 2:
                    errors.append(
                        f"block {bid}: conditional branch must have exactly "
                        f"2 successors, found {len(block.successors)}"
                    )
                for dest_id in (cond_inst.true_dest, cond_inst.false_dest):
                    if dest_id not in self._blocks:
                        errors.append(
                            f"block {bid}: branch destination {dest_id} "
                            "does not exist"
                        )
                    elif dest_id not in block.successors:
                        errors.append(
                            f"block {bid}: branch destination {dest_id} "
                            "not in successors list"
                        )

        return errors

    # ------------------------------------------------------------------
    # Text rendering
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        """Produce a human-readable text representation of the CFG."""
        lines: list[str] = []
        lines.append("=== Control Flow Graph ===")
        entry_id = self.entry_block.block_id if self.entry_block is not None else "N/A"
        exit_id = self.exit_block.block_id if self.exit_block is not None else "N/A"
        lines.append(f"Entry: Block {entry_id}")
        lines.append(f"Exit:  Block {exit_id}")
        lines.append(f"Total blocks: {len(self._blocks)}")
        lines.append("")

        for block_id in sorted(self._blocks.keys()):
            block = self._blocks[block_id]
            lines.append(f"--- Block {block_id} ---")
            lines.append(f"  Termination: {block.termination_type.name}")
            lines.append(f"  Predecessors: {block.predecessors}")
            lines.append(f"  Successors: {block.successors}")
            lines.append(f"  Filled: {block.filled}  Sealed: {block.sealed}")
            if block.instructions:
                lines.append("  Instructions:")
                for inst in block.instructions:
                    lines.append(f"    {_format_instruction(inst)}")
            else:
                lines.append("  Instructions: (none)")
            lines.append("")

        return "\n".join(lines)


def _format_instruction(inst: Any) -> str:
    """Format an IR instruction for display in :meth:`ControlFlowGraph.to_text`."""
    if isinstance(inst, AssignmentInstruction):
        return f"ASSIGN {inst.target} = {inst.expr}"
    if isinstance(inst, ConditionalBranchInstruction):
        return (
            f"BRANCH if {inst.condition} "
            f"-> B{inst.true_dest} else -> B{inst.false_dest}"
        )
    if isinstance(inst, UnconditionalJumpInstruction):
        return f"JUMP -> B{inst.dest}"
    return str(inst)
