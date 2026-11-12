"""Abstract syntax tree node definitions.

Defines the node types used to represent parsed source code.  Every node is
implemented as a dataclass so that structural equality, ``repr`` and
``__hash__`` are provided automatically.

Expression nodes (``IntegerLiteral``, ``Identifier``, ``BinaryExpression``,
``UnaryExpression``, ``ComparisonExpression`` implement ``__str__`` for compact
inline rendering.

Statement and program nodes can be rendered hierarchically via the
:func:`pretty_print` function.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


@dataclass
class IntegerLiteral:
    """An integer literal expression."""

    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class Identifier:
    """A variable reference expression."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class BinaryExpression:
    """A binary arithmetic expression (``+``, ``-``, ``*``)."""

    op: str
    left: object  # Expression node
    right: object  # Expression node

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryExpression:
    """A unary expression (e.g. ``-x``)."""

    op: str
    operand: object  # Expression node

    def __str__(self) -> str:
        return f"({self.op}{self.operand})"


@dataclass
class ComparisonExpression:
    """A relational comparison expression (``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``)."""

    op: str
    left: object  # Expression node
    right: object  # Expression node

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


# ---------------------------------------------------------------------------
# Statement nodes
# ---------------------------------------------------------------------------


@dataclass
class Assignment:
    """An assignment statement: ``target = expression;``."""

    target: str
    expr: object  # Expression node

    def __str__(self) -> str:
        return f"{self.target} = {self.expr};"


@dataclass
class Block:
    """A sequence of statements enclosed in braces."""

    statements: list = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(str(s) for s in self.statements)


@dataclass
class IfStatement:
    """An if-else statement."""

    condition: object  # ComparisonExpression
    then_block: Block
    else_block: Block

    def __str__(self) -> str:
        lines = [f"if ({self.condition}) {{"]
        for s in self.then_block.statements:
            lines.append(f"  {s}")
        lines.append("} else {")
        for s in self.else_block.statements:
            lines.append(f"  {s}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class WhileStatement:
    """A while loop statement."""

    condition: object  # ComparisonExpression
    body: Block

    def __str__(self) -> str:
        lines = [f"while ({self.condition}) {{"]
        for s in self.body.statements:
            lines.append(f"  {s}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Program:
    """The root AST node representing a complete program."""

    statements: list = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(str(s) for s in self.statements)


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------


def _expr_str(expr: object) -> str:
    """Return a compact string for an expression node."""
    return str(expr)


def pretty_print(node: object, indent: int = 0) -> str:
    """Produce a human-readable, indented representation of an AST node.

    Args:
        node: The AST node to render (Program, statement, or expression).
        indent: Current indentation level (number of 2-space units).

    Returns:
        A multi-line string representation of the node.
    """
    pad = "  " * indent
    child_pad = "  " * (indent + 1)

    if isinstance(node, Program):
        lines = [f"{pad}Program:"]
        for s in node.statements:
            lines.append(pretty_print(s, indent + 1))
        return "\n".join(lines)

    if isinstance(node, Block):
        if not node.statements:
            return f"{pad}{{ }}"
        lines = [f"{pad}{{"]
        for s in node.statements:
            lines.append(pretty_print(s, indent + 1))
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    if isinstance(node, Assignment):
        return f"{pad}{node.target} = {_expr_str(node.expr)};"

    if isinstance(node, IfStatement):
        lines = [f"{pad}if ({_expr_str(node.condition)}) {{"]
        for s in node.then_block.statements:
            lines.append(pretty_print(s, indent + 1))
        lines.append(f"{pad}}} else {{")
        for s in node.else_block.statements:
            lines.append(pretty_print(s, indent + 1))
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    if isinstance(node, WhileStatement):
        lines = [f"{pad}while ({_expr_str(node.condition)}) {{"]
        for s in node.body.statements:
            lines.append(pretty_print(s, indent + 1))
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    # Expressions and anything else: inline representation.
    return f"{pad}{_expr_str(node)}"
