"""Intermediate representation instructions.

Defines the instruction types used in the IR.  Each instruction is a dataclass
that stores structured data rather than raw strings.

Assignment instructions preserve the original expression tree (an AST node) so
that later SSA passes can walk it.  Branch instructions preserve their
condition and destination block IDs.

No SSA variable versions are introduced at this level.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class InstructionType(Enum):
    """Discriminator for IR instruction kinds."""

    ASSIGNMENT = auto()
    CONDITIONAL_BRANCH = auto()
    UNCONDITIONAL_JUMP = auto()


@dataclass
class AssignmentInstruction:
    """An assignment instruction: ``target = expression``.

    Attributes:
        target: The name of the variable being assigned.
        expr: The expression tree (an AST expression node) assigned to *target*.
    """

    target: str
    expr: Any  # AST expression node (IntegerLiteral, BinaryExpression, ...)

    @property
    def kind(self) -> InstructionType:
        return InstructionType.ASSIGNMENT


@dataclass
class ConditionalBranchInstruction:
    """A conditional branch instruction.

    Attributes:
        condition: The comparison expression (AST node) used as the branch test.
        true_dest: Block ID to jump to when the condition is true.
        false_dest: Block ID to jump to when the condition is false.
    """

    condition: Any  # AST ComparisonExpression node
    true_dest: int
    false_dest: int

    @property
    def kind(self) -> InstructionType:
        return InstructionType.CONDITIONAL_BRANCH


@dataclass
class UnconditionalJumpInstruction:
    """An unconditional jump instruction.

    Attributes:
        dest: Block ID to jump to.
    """

    dest: int

    @property
    def kind(self) -> InstructionType:
        return InstructionType.UNCONDITIONAL_JUMP
