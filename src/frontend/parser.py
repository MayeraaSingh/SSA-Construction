"""Parser.

Parses a token stream produced by the lexer into an abstract syntax tree
using a recursive-descent strategy.

Grammar (informal)::

    program        := statement*
    statement      := assignment | if_statement | while_statement | block
    assignment     := IDENTIFIER '=' expression ';'
    if_statement   := 'if' '(' condition ')' block 'else' block
    while_statement := 'while' '(' condition ')' block
    block          := '{' statement* '}'
    condition      := expression RELATIONAL_OPERATOR expression
    expression     := term (('+' | '-') term)*          # left-associative
    term           := factor ('*' factor)*              # left-associative
    factor         := INTEGER | IDENTIFIER | '(' expression ')' | '-' factor

The parser only produces AST nodes; it does not interact with the CFG or IR.
"""

from __future__ import annotations

from src.frontend.lexer import Token, TokenType
from src.frontend.ast_nodes import (
    Program,
    Block,
    Assignment,
    IfStatement,
    WhileStatement,
    BinaryExpression,
    UnaryExpression,
    Identifier,
    IntegerLiteral,
    ComparisonExpression,
)


class ParseError(Exception):
    """Raised when the parser encounters unexpected or missing tokens."""

    def __init__(self, message: str, token: Token) -> None:
        super().__init__(f"ParseError at line {token.line}, column {token.column}: {message}")
        self.token = token
        self.line = token.line
        self.column = token.column


class Parser:
    """Recursive-descent parser.

    Construct with a list of tokens (typically from :class:`~src.frontend.lexer.Lexer`)
    and call :meth:`parse` to obtain the AST.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> Program:
        """Parse the entire token stream into a :class:`Program` AST node."""
        return self.parse_program()

    # ------------------------------------------------------------------
    # Statement-level methods
    # ------------------------------------------------------------------

    def parse_program(self) -> Program:
        """Parse a program (zero or more statements)."""
        statements: list = []
        while self._peek().type != TokenType.EOF:
            statements.append(self.parse_statement())
        return Program(statements)

    def parse_statement(self) -> object:
        """Parse a single statement (assignment, if, while, or block)."""
        tok = self._peek()

        if tok.type == TokenType.KEYWORD:
            if tok.value == "if":
                return self.parse_if_statement()
            if tok.value == "while":
                return self.parse_while_statement()

        if tok.type == TokenType.LBRACE:
            return self.parse_block()

        if tok.type == TokenType.IDENTIFIER:
            return self.parse_assignment()

        raise ParseError(
            f"expected a statement but found {tok.type.name} {tok.value!r}", tok
        )

    def parse_block(self) -> Block:
        """Parse a brace-delimited block of statements."""
        self._expect(TokenType.LBRACE)
        statements: list = []
        while self._peek().type != TokenType.RBRACE:
            if self._peek().type == TokenType.EOF:
                raise ParseError("unexpected end of input, expected '}'", self._peek())
            statements.append(self.parse_statement())
        self._expect(TokenType.RBRACE)
        return Block(statements)

    def parse_assignment(self) -> Assignment:
        """Parse an assignment: ``IDENTIFIER = expression ;``."""
        target_tok = self._expect(TokenType.IDENTIFIER)

        eq_tok = self._peek()
        if eq_tok.type != TokenType.OPERATOR or eq_tok.value != "=":
            raise ParseError(
                f"expected '=' but found {eq_tok.type.name} {eq_tok.value!r}", eq_tok
            )
        self._advance()

        expr = self.parse_expression()
        self._expect(TokenType.SEMICOLON)
        return Assignment(target_tok.value, expr)

    def parse_if_statement(self) -> IfStatement:
        """Parse an if-else statement: ``if ( condition ) block else block``.

        An explicit ``else`` block is required.
        """
        self._expect_keyword("if")
        self._expect(TokenType.LPAREN)
        condition = self.parse_condition()
        self._expect(TokenType.RPAREN)
        then_block = self.parse_block()
        self._expect_keyword("else")
        else_block = self.parse_block()
        return IfStatement(condition, then_block, else_block)

    def parse_while_statement(self) -> WhileStatement:
        """Parse a while loop: ``while ( condition ) block``."""
        self._expect_keyword("while")
        self._expect(TokenType.LPAREN)
        condition = self.parse_condition()
        self._expect(TokenType.RPAREN)
        body = self.parse_block()
        return WhileStatement(condition, body)

    def parse_condition(self) -> ComparisonExpression:
        """Parse a relational condition: ``expression RELATIONAL_OPERATOR expression``."""
        left = self.parse_expression()
        op_tok = self._peek()
        if op_tok.type != TokenType.RELATIONAL_OPERATOR:
            raise ParseError(
                f"expected a relational operator but found {op_tok.type.name} {op_tok.value!r}",
                op_tok,
            )
        self._advance()
        right = self.parse_expression()
        return ComparisonExpression(op_tok.value, left, right)

    # ------------------------------------------------------------------
    # Expression-level methods (precedence climbing)
    # ------------------------------------------------------------------

    def parse_expression(self) -> object:
        """Parse an expression (addition / subtraction, left-associative)."""
        left = self.parse_term()
        while True:
            tok = self._peek()
            if tok.type == TokenType.OPERATOR and tok.value in ("+", "-"):
                self._advance()
                right = self.parse_term()
                left = BinaryExpression(tok.value, left, right)
            else:
                break
        return left

    def parse_term(self) -> object:
        """Parse a term (multiplication, left-associative)."""
        left = self.parse_factor()
        while True:
            tok = self._peek()
            if tok.type == TokenType.OPERATOR and tok.value == "*":
                self._advance()
                right = self.parse_factor()
                left = BinaryExpression(tok.value, left, right)
            else:
                break
        return left

    def parse_factor(self) -> object:
        """Parse a factor: ``INTEGER | IDENTIFIER | ( expression ) | - factor``."""
        tok = self._peek()

        if tok.type == TokenType.INTEGER:
            self._advance()
            return IntegerLiteral(int(tok.value))

        if tok.type == TokenType.IDENTIFIER:
            self._advance()
            return Identifier(tok.value)

        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self.parse_expression()
            self._expect(TokenType.RPAREN)
            return expr

        if tok.type == TokenType.OPERATOR and tok.value == "-":
            self._advance()
            operand = self.parse_factor()
            return UnaryExpression("-", operand)

        raise ParseError(
            f"expected an expression but found {tok.type.name} {tok.value!r}", tok
        )

    # ------------------------------------------------------------------
    # Token-stream helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        """Return the token at ``offset`` positions ahead (clamped to EOF)."""
        idx = min(self._pos + offset, len(self._tokens) - 1)
        return self._tokens[idx]

    def _advance(self) -> Token:
        """Consume and return the current token, advancing the position."""
        tok = self._tokens[self._pos]
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return tok

    def _expect(self, type_: TokenType) -> Token:
        """Consume the current token if it matches *type_*; otherwise raise."""
        tok = self._peek()
        if tok.type != type_:
            raise ParseError(
                f"expected {type_.name} but found {tok.type.name} {tok.value!r}", tok
            )
        return self._advance()

    def _expect_keyword(self, keyword: str) -> Token:
        """Consume the current token if it is the keyword *keyword*; otherwise raise."""
        tok = self._peek()
        if tok.type != TokenType.KEYWORD or tok.value != keyword:
            raise ParseError(
                f"expected keyword '{keyword}' but found {tok.type.name} {tok.value!r}",
                tok,
            )
        return self._advance()
