"""SSA simplification.

Implements trivial phi-function elimination, SSA value replacement, and
alias resolution for the lazy SSA construction algorithm.

A phi function is **trivial** when all of its non-self operands resolve to the
same SSA value.  For example::

    x_3 = φ(x_1, x_1)        →  x_3 is replaced by x_1 everywhere
    x_3 = φ(x_3, x_1)        →  x_3 is replaced by x_1 (self-reference ignored)
    x_3 = φ(x_1, x_2)        →  not trivial (operands differ)

When a trivial phi is found, its result is recorded as an alias for the
surviving value.  All subsequent lookups through :func:`resolve_alias`
return the surviving value transparently.
"""

from __future__ import annotations

from src.ssa.phi_functions import PhiFunction


def resolve_alias(value: str, aliases: dict[str, str]) -> str:
    """Follow the alias chain to find the ultimate replacement for *value*.

    If *value* has been aliased (because its defining phi was trivial and
    eliminated), follow the chain until a non-aliased value is found.  A
    cycle guard prevents infinite loops.
    """
    seen: set[str] = set()
    current = value
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


def find_trivial_replacement(
    phi: PhiFunction, aliases: dict[str, str]
) -> str | None:
    """Determine whether *phi* is trivial and return the surviving value.

    A phi is trivial when every non-self operand resolves (through *aliases*)
    to the same SSA value.  Self-references (operands equal to ``phi.result``
    or resolving to it) are skipped.

    Returns:
        The surviving SSA value name if the phi is trivial, or ``None`` if
        it is not trivial (or has no non-self operands).
    """
    same: str | None = None
    for operand in phi.operands.values():
        resolved = resolve_alias(operand, aliases)
        if resolved == phi.result:
            continue  # Self-reference — skip.
        if same is None:
            same = resolved
        elif same != resolved:
            return None  # Different operands — not trivial.
    return same


def eliminate_trivial_phi(
    phi: PhiFunction, aliases: dict[str, str]
) -> str | None:
    """Eliminate *phi* if it is trivial, recording the alias.

    If the phi is trivial, marks it as eliminated, sets ``replacement``, and
    records the alias in *aliases*.

    Returns:
        The surviving SSA value if eliminated, or ``None`` if not trivial.
    """
    replacement = find_trivial_replacement(phi, aliases)
    if replacement is not None:
        aliases[phi.result] = replacement
        phi.eliminated = True
        phi.replacement = replacement
        return replacement
    return None
