"""Tests for the parser.

Covers integer literals, identifier expressions, arithmetic precedence,
parenthesised expressions, assignments, if-else statements, while loops,
nested statements, and invalid-syntax error reporting.
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
from src.frontend.parser import ParseError, Parser


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def parse(source: str) -> Program:
    """Lex and parse *source* into a :class:`Program`."""
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def parse_stmt(source: str):
    """Parse a single-statement program and return that statement."""
    return parse(source).statements[0]


# ---------------------------------------------------------------------------
# Integer literals
# ---------------------------------------------------------------------------


class TestIntegerLiterals:
    def test_simple_integer(self):
        stmt = parse_stmt("x = 42;")
        assert isinstance(stmt, Assignment)
        assert isinstance(stmt.expr, IntegerLiteral)
        assert stmt.expr.value == 42

    def test_large_integer(self):
        stmt = parse_stmt("x = 12345;")
        assert stmt.expr.value == 12345

    def test_single_digit(self):
        stmt = parse_stmt("x = 0;")
        assert stmt.expr.value == 0


# ---------------------------------------------------------------------------
# Identifier expressions
# ---------------------------------------------------------------------------


class TestIdentifierExpressions:
    def test_identifier_assignment(self):
        stmt = parse_stmt("y = x;")
        assert isinstance(stmt.expr, Identifier)
        assert stmt.expr.name == "x"

    def test_underscore_identifier(self):
        stmt = parse_stmt("_var = _x;")
        assert stmt.target == "_var"
        assert isinstance(stmt.expr, Identifier)
        assert stmt.expr.name == "_x"

    def test_identifier_with_digits(self):
        stmt = parse_stmt("x1 = v2;")
        assert stmt.target == "x1"
        assert stmt.expr.name == "v2"


# ---------------------------------------------------------------------------
# Arithmetic precedence
# ---------------------------------------------------------------------------


class TestArithmeticPrecedence:
    def test_multiplication_before_addition(self):
        expr = parse_stmt("z = x + y * 2;").expr
        # Expected: x + (y * 2)
        assert isinstance(expr, BinaryExpression)
        assert expr.op == "+"
        assert isinstance(expr.left, Identifier)
        assert isinstance(expr.right, BinaryExpression)
        assert expr.right.op == "*"

    def test_multiplication_before_subtraction(self):
        expr = parse_stmt("z = x - y * 2;").expr
        # Expected: x - (y * 2)
        assert expr.op == "-"
        assert isinstance(expr.left, Identifier)
        assert isinstance(expr.right, BinaryExpression)
        assert expr.right.op == "*"

    def test_left_associative_subtraction(self):
        expr = parse_stmt("z = x - y - 2;").expr
        # Expected: (x - y) - 2
        assert expr.op == "-"
        assert isinstance(expr.left, BinaryExpression)
        assert expr.left.op == "-"
        assert isinstance(expr.right, IntegerLiteral)
        assert expr.right.value == 2

    def test_left_associative_addition(self):
        expr = parse_stmt("z = x + y + z;").expr
        # Expected: (x + y) + z
        assert expr.op == "+"
        assert isinstance(expr.left, BinaryExpression)
        assert expr.left.op == "+"

    def test_multiplication_left_associative(self):
        expr = parse_stmt("z = a * b * c;").expr
        # Expected: (a * b) * c
        assert expr.op == "*"
        assert isinstance(expr.left, BinaryExpression)
        assert expr.left.op == "*"

    def test_complex_precedence_from_example(self):
        expr = parse_stmt("z = x * y - 2;").expr
        # Expected: (x * y) - 2
        assert expr.op == "-"
        assert isinstance(expr.left, BinaryExpression)
        assert expr.left.op == "*"
        assert isinstance(expr.right, IntegerLiteral)
        assert expr.right.value == 2

    def test_chained_mixed_operators(self):
        expr = parse_stmt("z = a + b * c - d;").expr
        # Expected: ((a + (b * c)) - d)
        assert expr.op == "-"
        assert isinstance(expr.left, BinaryExpression)
        assert expr.left.op == "+"
        assert isinstance(expr.right, Identifier)


# ---------------------------------------------------------------------------
# Parenthesised expressions
# ---------------------------------------------------------------------------


class TestParenthesisedExpressions:
    def test_parens_override_precedence(self):
        expr = parse_stmt("z = (x + y) * 2;").expr
        # Expected: (x + y) * 2
        assert expr.op == "*"
        assert isinstance(expr.left, BinaryExpression)
        assert expr.left.op == "+"
        assert isinstance(expr.right, IntegerLiteral)

    def test_nested_parens(self):
        expr = parse_stmt("z = ((x + 1) * (y + 2)) - 3;").expr
        assert expr.op == "-"
        assert isinstance(expr.right, IntegerLiteral)
        assert expr.right.value == 3

    def test_redundant_parens(self):
        expr = parse_stmt("z = ((x + 5));").expr
        # Inner parens should unwrap to the binary expression.
        assert isinstance(expr, BinaryExpression)
        assert expr.op == "+"


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------


class TestAssignments:
    def test_simple_assignment(self):
        stmt = parse_stmt("x = 10;")
        assert isinstance(stmt, Assignment)
        assert stmt.target == "x"
        assert isinstance(stmt.expr, IntegerLiteral)
        assert stmt.expr.value == 10

    def test_assignment_with_binary_expr(self):
        stmt = parse_stmt("y = x + 5;")
        assert stmt.target == "y"
        assert isinstance(stmt.expr, BinaryExpression)

    def test_multiple_assignments(self):
        ast = parse("x = 1;\ny = 2;\nz = 3;")
        assert len(ast.statements) == 3
        assert all(isinstance(s, Assignment) for s in ast.statements)
        assert ast.statements[0].target == "x"
        assert ast.statements[1].target == "y"
        assert ast.statements[2].target == "z"

    def test_unary_minus(self):
        stmt = parse_stmt("x = -5;")
        assert isinstance(stmt.expr, UnaryExpression)
        assert stmt.expr.op == "-"

    def test_unary_minus_on_variable(self):
        stmt = parse_stmt("x = -y;")
        assert isinstance(stmt.expr, UnaryExpression)
        assert isinstance(stmt.expr.operand, Identifier)


# ---------------------------------------------------------------------------
# If-else statements
# ---------------------------------------------------------------------------


class TestIfElseStatements:
    def test_simple_if_else(self):
        stmt = parse_stmt("if (x > 10) { y = 20; } else { y = 30; }")
        assert isinstance(stmt, IfStatement)
        assert isinstance(stmt.condition, ComparisonExpression)
        assert stmt.condition.op == ">"
        assert len(stmt.then_block.statements) == 1
        assert len(stmt.else_block.statements) == 1

    def test_if_else_blocks_are_blocks(self):
        stmt = parse_stmt("if (x > 10) { y = 20; } else { y = 30; }")
        assert isinstance(stmt.then_block, Block)
        assert isinstance(stmt.else_block, Block)

    def test_nested_if(self):
        source = """if (x > 0) {
    if (x > 5) { y = 10; } else { y = 20; }
} else {
    y = 30;
}"""
        stmt = parse_stmt(source)
        assert isinstance(stmt, IfStatement)
        inner = stmt.then_block.statements[0]
        assert isinstance(inner, IfStatement)
        assert inner.then_block.statements[0].target == "y"
        assert inner.then_block.statements[0].expr.value == 10

    def test_all_relational_operators(self):
        for op in ["<", ">", "<=", ">=", "==", "!="]:
            stmt = parse_stmt(f"if (x {op} 5) {{ y = 1; }} else {{ y = 2; }}")
            assert stmt.condition.op == op

    def test_condition_with_expressions(self):
        stmt = parse_stmt("if (x + 1 > y * 2) { a = 1; } else { a = 2; }")
        assert isinstance(stmt.condition, ComparisonExpression)
        assert isinstance(stmt.condition.left, BinaryExpression)
        assert isinstance(stmt.condition.right, BinaryExpression)

    def test_if_in_then_branch(self):
        source = """if (x > 0) {
    if (y > 1) { z = 1; } else { z = 2; }
} else {
    z = 3;
}"""
        outer = parse_stmt(source)
        assert isinstance(outer, IfStatement)
        assert isinstance(outer.then_block.statements[0], IfStatement)


# ---------------------------------------------------------------------------
# While loops
# ---------------------------------------------------------------------------


class TestWhileLoops:
    def test_simple_while(self):
        stmt = parse_stmt("while (x < 5) { x = x + 1; }")
        assert isinstance(stmt, WhileStatement)
        assert isinstance(stmt.condition, ComparisonExpression)
        assert stmt.condition.op == "<"
        assert len(stmt.body.statements) == 1

    def test_while_body_is_block(self):
        stmt = parse_stmt("while (x < 5) { x = x + 1; }")
        assert isinstance(stmt.body, Block)

    def test_while_with_multiple_body_statements(self):
        source = "while (x < 5) { x = x + 1; y = y + 2; z = z + 3; }"
        stmt = parse_stmt(source)
        assert len(stmt.body.statements) == 3

    def test_while_with_nested_if(self):
        source = """while (x < 10) {
    if (x > 5) { y = 1; } else { y = 2; }
}"""
        stmt = parse_stmt(source)
        assert isinstance(stmt, WhileStatement)
        assert isinstance(stmt.body.statements[0], IfStatement)

    def test_nested_while(self):
        source = "while (x < 5) { while (y < 3) { y = y + 1; } x = x + 1; }"
        stmt = parse_stmt(source)
        assert isinstance(stmt, WhileStatement)
        inner = stmt.body.statements[0]
        assert isinstance(inner, WhileStatement)
        assert inner.condition.left.name == "y"


# ---------------------------------------------------------------------------
# Nested statements
# ---------------------------------------------------------------------------


class TestNestedStatements:
    def test_statements_after_if(self):
        source = "if (x > 5) { y = 1; } else { y = 2; }\nz = y + 5;"
        ast = parse(source)
        assert len(ast.statements) == 2
        assert isinstance(ast.statements[0], IfStatement)
        assert isinstance(ast.statements[1], Assignment)

    def test_statements_after_while(self):
        source = "while (x < 5) { x = x + 1; }\ny = x;"
        ast = parse(source)
        assert len(ast.statements) == 2
        assert isinstance(ast.statements[0], WhileStatement)
        assert isinstance(ast.statements[1], Assignment)

    def test_while_inside_if(self):
        source = """if (x > 0) {
    while (x < 10) { x = x + 1; }
} else {
    y = 1;
}"""
        if_stmt = parse_stmt(source)
        assert isinstance(if_stmt, IfStatement)
        assert isinstance(if_stmt.then_block.statements[0], WhileStatement)

    def test_sequential_if_statements(self):
        source = """if (x > 1) { y = 1; } else { y = 2; }
if (x > 2) { z = 1; } else { z = 2; }"""
        ast = parse(source)
        assert len(ast.statements) == 2
        assert all(isinstance(s, IfStatement) for s in ast.statements)

    def test_empty_program(self):
        ast = parse("")
        assert isinstance(ast, Program)
        assert len(ast.statements) == 0

    def test_block_statement(self):
        ast = parse("{ x = 1; y = 2; }")
        assert len(ast.statements) == 1
        assert isinstance(ast.statements[0], Block)
        assert len(ast.statements[0].statements) == 2

    def test_full_conditional_example(self):
        source = """x = 10;

if (x > 5) {
    y = 20;
} else {
    y = 30;
}

z = y + 5;"""
        ast = parse(source)
        assert len(ast.statements) == 3
        assert isinstance(ast.statements[0], Assignment)
        assert isinstance(ast.statements[1], IfStatement)
        assert isinstance(ast.statements[2], Assignment)


# ---------------------------------------------------------------------------
# Invalid syntax and error locations
# ---------------------------------------------------------------------------


class TestInvalidSyntax:
    def test_missing_semicolon(self):
        with pytest.raises(ParseError):
            parse("x = 10")

    def test_missing_brace(self):
        with pytest.raises(ParseError):
            parse("if (x > 5) { y = 1;")

    def test_missing_else_raises(self):
        with pytest.raises(ParseError):
            parse("if (x > 5) { y = 1; }")

    def test_missing_close_paren_in_if(self):
        with pytest.raises(ParseError):
            parse("if (x > 5 { y = 1; } else { y = 2; }")

    def test_unexpected_operator_at_statement_start(self):
        with pytest.raises(ParseError):
            parse("+ + 5;")

    def test_missing_condition_operator(self):
        with pytest.raises(ParseError):
            parse("if (x) { y = 1; } else { y = 2; }")

    def test_unterminated_parenthesis(self):
        with pytest.raises(ParseError):
            parse("x = (1 + 2;")

    def test_error_has_line_and_column(self):
        with pytest.raises(ParseError) as exc_info:
            parse("x = 10")
        assert exc_info.value.line >= 1
        assert exc_info.value.column >= 1

    def test_error_reports_correct_line(self):
        source = "x = 1;\ny = 2;\nz = +"
        with pytest.raises(ParseError) as exc_info:
            parse(source)
        assert exc_info.value.line == 3

    def test_empty_if_block_still_requires_else(self):
        with pytest.raises(ParseError):
            parse("if (x > 5) { }")
