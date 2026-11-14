"""Semantic equivalence tests.

Implements a source-level interpreter (walks the AST) and an SSA-level
evaluator (walks the CFG/SSA IR), then compares their final variable values
for a collection of deterministic programs.  This provides evidence that the
SSA construction preserves the original program's behaviour.

The source interpreter executes the AST directly, maintaining a simple
``{variable: int}`` environment.  The SSA evaluator executes the CFG by
following edges, evaluating phi functions using the actual predecessor block,
and evaluating SSA instructions using the SSA value store.

Both interpreters use Python integers and the standard comparison/arithmetic
operators, so their results are directly comparable.
"""

from __future__ import annotations

import pytest

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
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.ssa.ssa_builder import (
    SSAAssignment,
    SSABinaryExpression,
    SSAComparisonExpression,
    SSAConditionalBranch,
    SSAError,
    SSAIntegerLiteral,
    SSAUnconditionalJump,
    SSAUnaryExpression,
    SSAValue,
    SSABuilder,
)


# ---------------------------------------------------------------------------
# Source interpreter
# ---------------------------------------------------------------------------


class SourceInterpreter:
    """Executes the AST directly, maintaining a variable-to-value environment."""

    def __init__(self) -> None:
        self.env: dict[str, int] = {}

    def run(self, program: Program) -> dict[str, int]:
        """Execute *program* and return the final variable environment."""
        for stmt in program.statements:
            self._exec_stmt(stmt)
        return dict(self.env)

    def _exec_stmt(self, stmt: object) -> None:
        if isinstance(stmt, Assignment):
            self.env[stmt.target] = self._eval_expr(stmt.expr)
        elif isinstance(stmt, IfStatement):
            if self._eval_condition(stmt.condition):
                for s in stmt.then_block.statements:
                    self._exec_stmt(s)
            else:
                for s in stmt.else_block.statements:
                    self._exec_stmt(s)
        elif isinstance(stmt, WhileStatement):
            while self._eval_condition(stmt.condition):
                for s in stmt.body.statements:
                    self._exec_stmt(s)
        elif isinstance(stmt, Block):
            for s in stmt.statements:
                self._exec_stmt(s)
        else:
            raise RuntimeError(f"cannot execute {type(stmt).__name__}")

    def _eval_expr(self, expr: object) -> int:
        if isinstance(expr, IntegerLiteral):
            return expr.value
        if isinstance(expr, Identifier):
            if expr.name not in self.env:
                raise NameError(f"undefined variable '{expr.name}'")
            return self.env[expr.name]
        if isinstance(expr, BinaryExpression):
            left = self._eval_expr(expr.left)
            right = self._eval_expr(expr.right)
            return self._apply_binop(expr.op, left, right)
        if isinstance(expr, UnaryExpression):
            operand = self._eval_expr(expr.operand)
            if expr.op == "-":
                return -operand
            raise RuntimeError(f"unknown unary op {expr.op}")
        raise RuntimeError(f"cannot evaluate {type(expr).__name__}")

    def _eval_condition(self, cond: object) -> bool:
        if isinstance(cond, ComparisonExpression):
            left = self._eval_expr(cond.left)
            right = self._eval_expr(cond.right)
            return self._apply_cmp(cond.op, left, right)
        raise RuntimeError(f"cannot evaluate condition {type(cond).__name__}")

    @staticmethod
    def _apply_binop(op: str, left: int, right: int) -> int:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        raise RuntimeError(f"unknown binary op {op}")

    @staticmethod
    def _apply_cmp(op: str, left: int, right: int) -> bool:
        if op == "<":
            return left < right
        if op == ">":
            return left > right
        if op == "<=":
            return left <= right
        if op == ">=":
            return left >= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        raise RuntimeError(f"unknown comparison op {op}")


# ---------------------------------------------------------------------------
# SSA evaluator
# ---------------------------------------------------------------------------


class SSAEvaluator:
    """Executes the CFG/SSA IR by following edges and evaluating instructions.

    Phi functions are evaluated using the actual predecessor block ID to
    select the correct operand.  SSA assignments and conditions are evaluated
    using an SSA value store.  A separate ``source_values`` dict tracks the
    current value of each source variable (updated on every assignment and
    phi evaluation), so the final result can be compared with the source
    interpreter.
    """

    MAX_STEPS = 100_000

    def __init__(self, builder: SSABuilder) -> None:
        self.builder = builder
        self.cfg = builder._cfg
        self.values: dict[str, int] = {}
        self.source_values: dict[str, int] = {}

    def run(self) -> dict[str, int]:
        """Execute the SSA IR and return the final source-variable values."""
        block = self.cfg.entry_block
        pred_id: int | None = None
        steps = 0

        while block is not None:
            steps += 1
            if steps > self.MAX_STEPS:
                raise RuntimeError("SSA evaluation exceeded step limit (infinite loop?)")

            # Evaluate phi functions using the actual predecessor.
            for phi in block.phis:
                if not phi.eliminated:
                    operand = phi.operands.get(pred_id)
                    if operand is not None:
                        resolved = self.builder._resolve_alias(operand)
                        val = self.values[resolved]
                        self.values[phi.result] = val
                        self.source_values[phi.variable] = val

            # Evaluate SSA instructions.
            next_block_id: int | None = None
            for inst in block.ssa_instructions:
                if isinstance(inst, SSAAssignment):
                    val = self._eval_expr(inst.expr)
                    self.values[inst.result] = val
                    var = self.builder.ssa_to_source_var[inst.result]
                    self.source_values[var] = val
                elif isinstance(inst, SSAConditionalBranch):
                    cond = self._eval_condition(inst.condition)
                    next_block_id = inst.true_dest if cond else inst.false_dest
                elif isinstance(inst, SSAUnconditionalJump):
                    next_block_id = inst.dest

            pred_id = block.block_id
            block = self.cfg.get_block(next_block_id) if next_block_id is not None else None

        return dict(self.source_values)

    def _eval_expr(self, expr: object) -> int:
        if isinstance(expr, SSAIntegerLiteral):
            return expr.value
        if isinstance(expr, SSAValue):
            resolved = self.builder._resolve_alias(expr.name)
            return self.values[resolved]
        if isinstance(expr, SSABinaryExpression):
            left = self._eval_expr(expr.left)
            right = self._eval_expr(expr.right)
            return SourceInterpreter._apply_binop(expr.op, left, right)
        if isinstance(expr, SSAUnaryExpression):
            operand = self._eval_expr(expr.operand)
            if expr.op == "-":
                return -operand
            raise RuntimeError(f"unknown unary op {expr.op}")
        raise RuntimeError(f"cannot evaluate SSA expr {type(expr).__name__}")

    def _eval_condition(self, cond: object) -> bool:
        if isinstance(cond, SSAComparisonExpression):
            left = self._eval_expr(cond.left)
            right = self._eval_expr(cond.right)
            return SourceInterpreter._apply_cmp(cond.op, left, right)
        raise RuntimeError(f"cannot evaluate SSA condition {type(cond).__name__}")


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


def run_both(source: str) -> tuple[dict[str, int], dict[str, int]]:
    """Run both interpreters on *source* and return (source_result, ssa_result)."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()

    source_result = SourceInterpreter().run(ast)

    builder = SSABuilder()
    builder.build(ast)
    ssa_result = SSAEvaluator(builder).run()

    return source_result, ssa_result


# ---------------------------------------------------------------------------
# Semantic equivalence tests
# ---------------------------------------------------------------------------


class TestSemanticStraightLine:
    def test_simple_assignments(self):
        source = "x = 10;\ny = x + 5;\nz = x * y - 2;"
        src, ssa = run_both(source)
        assert src == ssa
        assert src["x"] == 10
        assert src["y"] == 15
        assert src["z"] == 148

    def test_reassignment(self):
        source = "x = 10;\nx = 20;\nx = 30;\ny = x;"
        src, ssa = run_both(source)
        assert src == ssa
        assert src["x"] == 30
        assert src["y"] == 30

    def test_arithmetic_precedence(self):
        source = "x = 2 + 3 * 4;\ny = (2 + 3) * 4;\nz = x - y;"
        src, ssa = run_both(source)
        assert src == ssa
        assert src["x"] == 14
        assert src["y"] == 20
        assert src["z"] == -6


class TestSemanticConditional:
    def test_if_true_branch(self):
        source = "a = 1;\nx = 10;\nif (a > 0) {\n    x = 20;\n} else {\n    x = 30;\n}\ny = x;"
        src, ssa = run_both(source)
        assert src == ssa
        assert src["y"] == 20

    def test_if_false_branch(self):
        source = "a = 0;\nx = 10;\nif (a > 0) {\n    x = 20;\n} else {\n    x = 30;\n}\ny = x;"
        src, ssa = run_both(source)
        assert src == ssa
        assert src["y"] == 30

    def test_variable_not_modified_in_one_branch_true(self):
        source = "a = 1;\nx = 10;\nif (a > 0) {\n    x = 20;\n} else {\n}\ny = x;"
        src, ssa = run_both(source)
        assert src == ssa
        assert src["y"] == 20

    def test_variable_not_modified_in_one_branch_false(self):
        source = "a = 0;\nx = 10;\nif (a > 0) {\n    x = 20;\n} else {\n}\ny = x;"
        src, ssa = run_both(source)
        assert src == ssa
        assert src["y"] == 10

    def test_multiple_variables_at_merge(self):
        source = """a = 1;
x = 10;
y = 20;

if (a > 0) {
    x = 30;
    y = 40;
} else {
    x = 50;
    y = 60;
}

z = x + y;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["z"] == 70

    def test_nested_if_else(self):
        source = """a = 1;
b = 1;
x = 0;

if (a > 0) {
    if (b > 0) {
        x = 10;
    } else {
        x = 20;
    }
} else {
    x = 30;
}

y = x;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["y"] == 10

    def test_nested_if_else_false_path(self):
        source = """a = 0;
b = 1;
x = 0;

if (a > 0) {
    if (b > 0) {
        x = 10;
    } else {
        x = 20;
    }
} else {
    x = 30;
}

y = x;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["y"] == 30


class TestSemanticLoop:
    def test_simple_loop(self):
        source = """x = 0;

while (x < 5) {
    x = x + 1;
}

y = x;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["x"] == 5
        assert src["y"] == 5

    def test_loop_with_accumulation(self):
        source = """sum = 0;
i = 0;

while (i < 10) {
    sum = sum + i;
    i = i + 1;
}

result = sum;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["result"] == 45

    def test_loop_unmodified_variable(self):
        source = """x = 0;
y = 10;

while (x < 5) {
    x = x + 1;
}

z = y;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["z"] == 10

    def test_loop_modified_before_and_after(self):
        source = """x = 0;

while (x < 5) {
    x = x + 1;
}

x = x + 10;
y = x;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["x"] == 15
        assert src["y"] == 15

    def test_nested_loops(self):
        source = """x = 0;
count = 0;

while (x < 3) {
    y = 0;
    while (y < 2) {
        count = count + 1;
        y = y + 1;
    }
    x = x + 1;
}

result = count;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["result"] == 6


class TestSemanticLoopWithIfElse:
    def test_if_else_in_loop(self):
        source = """x = 0;
y = 0;

while (x < 10) {
    if (x > 5) {
        y = y + 10;
    } else {
        y = y + 1;
    }
    x = x + 1;
}

result = y;"""
        src, ssa = run_both(source)
        assert src == ssa
        # x=0..5: y += 1 each → 6 (x=5: 5>5 is false, goes to else)
        # x=6..9: y += 10 each → 40
        # Total: 6 + 40 = 46
        assert src["result"] == 46

    def test_variable_modified_in_one_branch_of_loop(self):
        source = """x = 0;

while (x < 10) {
    if (x > 5) {
        x = x + 2;
    } else {
        x = x + 1;
    }
}

y = x;"""
        src, ssa = run_both(source)
        assert src == ssa
        # x=0→1, 1→2, 2→3, 3→4, 4→5, 5→6 (5>5 is false),
        # 6→8, 8→10
        assert src["y"] == 10


class TestSemanticComplexPrograms:
    def test_two_sequential_if_else(self):
        source = """a = 1;
x = 10;
y = 20;

if (a > 0) {
    x = 30;
} else {
    x = 40;
}

if (a > 0) {
    y = 50;
} else {
    y = 60;
}

z = x + y;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["z"] == 80

    def test_empty_branches(self):
        source = """a = 1;
x = 10;

if (a > 0) {
} else {
}

y = x;"""
        src, ssa = run_both(source)
        assert src == ssa
        assert src["y"] == 10

    def test_all_relational_operators(self):
        for op, left_val, expected in [
            ("<", 3, 0), (">", 3, 0), ("<=", 3, 1), (">=", 3, 1),
            ("==", 3, 1), ("!=", 3, 0),
        ]:
            source = f"a = {left_val};\nif (a {op} 3) {{\n    r = 1;\n}} else {{\n    r = 0;\n}}"
            src, ssa = run_both(source)
            assert src == ssa, f"failed for operator {op}"
            assert src["r"] == expected

    def test_unary_minus(self):
        source = "x = -5;\ny = -x;\nz = x + y;"
        src, ssa = run_both(source)
        assert src == ssa
        assert src["x"] == -5
        assert src["y"] == 5
        assert src["z"] == 0

    def test_complex_nested_program(self):
        source = """a = 1;
b = 2;
result = 0;

while (a < 3) {
    if (b > 0) {
        result = result + a;
    } else {
        result = result + b;
    }
    a = a + 1;
}

final = result;"""
        src, ssa = run_both(source)
        assert src == ssa
        # a=1: result = 0 + 1 = 1
        # a=2: result = 1 + 2 = 3
        assert src["final"] == 3
