"""Basic block representation.

A :class:`BasicBlock` groups a sequence of straight-line IR instructions that
execute without branching.  Each block tracks its predecessors and successors
so that the CFG (and later SSA passes) can walk the graph in either direction.

The block also carries two lifecycle flags for future SSA integration:

``filled``
    All instructions in the block have been processed.  Set to ``True`` once
    the block is complete (has a terminator).

``sealed``
    No additional predecessors will be added to this block.  Set to ``True``
    once the CFG is fully constructed.  The lazy SSA algorithm uses this to
    decide whether phi functions can be finalised, but no SSA behaviour is
    implemented on Day 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TerminationType(Enum):
    """How a basic block terminates."""

    NONE = auto()         # Block is still open; no terminator yet.
    CONDITIONAL = auto()  # Block ends with a conditional branch (two successors).
    JUMP = auto()         # Block ends with an unconditional jump (one successor).
    EXIT = auto()         # Block is the CFG exit (no successors).


@dataclass
class BasicBlock:
    """A basic block in the control flow graph.

    Attributes:
        block_id: Unique identifier assigned by the CFG.
        instructions: Ordered list of IR instructions in this block.
        predecessors: List of predecessor block IDs (insertion order).
        successors: List of successor block IDs (insertion order).
        termination_type: How this block terminates.
        condition: Optional branch condition (set for conditional blocks).
        filled: Whether all instructions have been processed (SSA lifecycle).
        sealed: Whether no more predecessors will be added (SSA lifecycle).
    """

    block_id: int
    instructions: list[Any] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)
    termination_type: TerminationType = TerminationType.NONE
    condition: Any = None
    filled: bool = False
    sealed: bool = False

    def add_instruction(self, instruction: Any) -> None:
        """Append an instruction to this block."""
        self.instructions.append(instruction)

    def add_predecessor(self, pred_id: int) -> None:
        """Add a predecessor block ID, ignoring duplicates."""
        if pred_id not in self.predecessors:
            self.predecessors.append(pred_id)

    def add_successor(self, succ_id: int) -> None:
        """Add a successor block ID, ignoring duplicates."""
        if succ_id not in self.successors:
            self.successors.append(succ_id)

    def remove_predecessor(self, pred_id: int) -> None:
        """Remove a predecessor block ID if present."""
        if pred_id in self.predecessors:
            self.predecessors.remove(pred_id)

    def remove_successor(self, succ_id: int) -> None:
        """Remove a successor block ID if present."""
        if succ_id in self.successors:
            self.successors.remove(succ_id)
