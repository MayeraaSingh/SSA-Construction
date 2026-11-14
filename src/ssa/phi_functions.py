"""Phi-function representation and management.

Defines the :class:`PhiFunction` data structure used throughout SSA
construction.  A phi function merges multiple reaching definitions of the
same source variable at a control-flow merge point.

Key features:

* **Ordered operands** — operands are stored in a regular ``dict`` (insertion
  ordered in Python 3.7+) keyed by predecessor block ID, so each operand is
  associated with the edge it flows along.

* **Incomplete phi support** — a phi created in an unsealed block starts with
  empty operands.  :meth:`is_complete` reports whether all operands have been
  resolved.  This is the mechanism that lets the lazy algorithm defer operand
  resolution until a block is sealed.

* **Elimination tracking** — when a trivial phi is eliminated it is marked
  ``eliminated`` and its ``replacement`` field records the surviving SSA value.
  Callers should always resolve through the SSA builder's alias map rather
  than inspecting ``replacement`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhiFunction:
    """A phi (φ) function in the SSA intermediate representation.

    Attributes:
        result: The SSA value name produced by this phi (e.g. ``"x_3"``).
        variable: The source-level variable name this phi merges (e.g. ``"x"``).
        block_id: ID of the block that owns this phi.
        operands: Ordered mapping of predecessor block ID → SSA value name.
            Empty when the phi is incomplete (created in an unsealed block).
        eliminated: ``True`` once trivial-phi elimination removes this phi.
        replacement: The surviving SSA value name if the phi was eliminated.
    """

    result: str
    variable: str
    block_id: int
    operands: dict[int, str] = field(default_factory=dict)
    eliminated: bool = False
    replacement: str | None = None

    def is_complete(self) -> bool:
        """Return ``True`` when every predecessor has a resolved operand."""
        return len(self.operands) > 0

    def add_operand(self, pred_id: int, value: str) -> None:
        """Set or overwrite the operand coming from predecessor *pred_id*."""
        self.operands[pred_id] = value

    def __str__(self) -> str:
        """Render the phi in a human-readable form."""
        if self.eliminated:
            return f"{self.result} => {self.replacement}  [trivial, eliminated]"
        ops = ", ".join(
            f"{value} [B{pred}]" for pred, value in self.operands.items()
        )
        return f"{self.result} = phi({ops})"
