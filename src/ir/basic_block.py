"""Basic block representation.

A :class:`BasicBlock` groups a sequence of straight-line IR instructions that
execute without branching.  Each block tracks its predecessors and successors
so that the CFG (and later SSA passes) can walk the graph in either direction.

The block carries two lifecycle flags for SSA integration:

``filled``
    All instructions in the block have been processed.  Set to ``True`` once
    the block has its terminator (conditional branch or jump).  A block can
    be filled while still unsealed (e.g. a loop header whose backedge has not
    yet been added).

``sealed``
    No additional predecessors will be added to this block.  In the lazy
    sealed-block SSA algorithm, a block is sealed once all its predecessors
    are known.  Until then, variable reads may create incomplete phi
    functions; when the block is sealed, those incomplete phi functions are
    resolved by reading from each newly-available predecessor.

Day 2 extensions (added below the Day 1 fields):

``current_defs``
    Per-block mapping from source variable name to the current SSA value
    that holds at the end of the block (e.g. ``{"x": "x_3"}``).

``phis``
    Ordered list of :class:`~src.ssa.phi_functions.PhiFunction` objects
    inserted in this block.  Phi functions created before the block is
    sealed start with empty operands (incomplete) and are resolved when the
    block is sealed.

``ssa_instructions``
    Ordered list of SSA-form instructions (SSAAssignment, SSAConditionalBranch,
    SSAUnconditionalJump).  These are the SSA-converted counterparts of the
    Day 1 IR instructions stored in ``instructions``.
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
        instructions: Ordered list of Day 1 IR instructions (pre-SSA).
        predecessors: List of predecessor block IDs (insertion order).
        successors: List of successor block IDs (insertion order).
        termination_type: How this block terminates.
        condition: Optional branch condition (set for conditional blocks).
        filled: Whether all instructions have been processed (SSA lifecycle).
        sealed: Whether no more predecessors will be added (SSA lifecycle).
        current_defs: Per-block mapping of source variable name → current SSA value.
        phis: Ordered list of PhiFunction objects in this block.
        ssa_instructions: Ordered list of SSA-form instructions.
    """

    block_id: int
    instructions: list[Any] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)
    termination_type: TerminationType = TerminationType.NONE
    condition: Any = None
    filled: bool = False
    sealed: bool = False
    # Day 2 SSA fields
    current_defs: dict[str, str] = field(default_factory=dict)
    phis: list[Any] = field(default_factory=list)
    ssa_instructions: list[Any] = field(default_factory=list)

    def add_instruction(self, instruction: Any) -> None:
        """Append a Day 1 IR instruction to this block."""
        self.instructions.append(instruction)

    def add_ssa_instruction(self, instruction: Any) -> None:
        """Append an SSA-form instruction to this block."""
        self.ssa_instructions.append(instruction)

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
