"""Lexical analyzer.

Tokenizes source code into a stream of tokens consumed by the parser.
The lexer is hand-written and processes the source character by character,
tracking line and column information for error reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Categories of tokens produced by the lexer."""

    KEYWORD = auto()
    IDENTIFIER = auto()
    INTEGER = auto()
    OPERATOR = auto()
    RELATIONAL_OPERATOR = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    EOF = auto()


@dataclass
class Token:
    """A single token produced by the lexer.

    Attributes:
        type: The token category.
        value: The literal text of the token.
        line: 1-based line number where the token starts.
        column: 1-based column number where the token starts.
    """

    type: TokenType
    value: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"


class LexError(Exception):
    """Raised when the lexer encounters an invalid character."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"LexError at line {line}, column {column}: {message}")
        self.line = line
        self.column = column


# Keywords recognised by the lexer.
KEYWORDS: frozenset[str] = frozenset({"if", "else", "while"})

# Multi-character relational operators (checked before single-char forms).
_TWO_CHAR_REL_OPS: frozenset[str] = frozenset({"<=", ">=", "==", "!="})
_ONE_CHAR_REL_OPS: frozenset[str] = frozenset({"<", ">"})


class Lexer:
    """Character-by-character lexical analyzer.

    Usage::

        tokens = Lexer(source_text).tokenize()

    The returned list always ends with an ``EOF`` token.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._line = 1
        self._col = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source text and return the token list (including EOF)."""
        tokens: list[Token] = []
        while self._pos < len(self._source):
            ch = self._source[self._pos]

            # Skip whitespace (spaces, tabs, newlines, carriage returns).
            if ch in (" ", "\t", "\r", "\n"):
                self._advance()
                continue

            start_line = self._line
            start_col = self._col

            if ch.isdigit():
                tokens.append(self._read_number(start_line, start_col))
            elif ch.isalpha() or ch == "_":
                tokens.append(self._read_word(start_line, start_col))
            elif ch in ("<", ">", "=", "!"):
                tokens.append(self._read_operator(start_line, start_col))
            elif ch in ("+", "-", "*"):
                self._advance()
                tokens.append(Token(TokenType.OPERATOR, ch, start_line, start_col))
            elif ch == "(":
                self._advance()
                tokens.append(Token(TokenType.LPAREN, ch, start_line, start_col))
            elif ch == ")":
                self._advance()
                tokens.append(Token(TokenType.RPAREN, ch, start_line, start_col))
            elif ch == "{":
                self._advance()
                tokens.append(Token(TokenType.LBRACE, ch, start_line, start_col))
            elif ch == "}":
                self._advance()
                tokens.append(Token(TokenType.RBRACE, ch, start_line, start_col))
            elif ch == ";":
                self._advance()
                tokens.append(Token(TokenType.SEMICOLON, ch, start_line, start_col))
            else:
                raise LexError(f"invalid character {ch!r}", start_line, start_col)

        tokens.append(Token(TokenType.EOF, "", self._line, self._col))
        return tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> str | None:
        """Return the character at ``offset`` from the current position, or None."""
        idx = self._pos + offset
        if idx < len(self._source):
            return self._source[idx]
        return None

    def _advance(self) -> str:
        """Consume and return the current character, updating line/column."""
        ch = self._source[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    def _read_number(self, start_line: int, start_col: int) -> Token:
        """Read a (possibly multi-digit) integer literal."""
        chars: list[str] = []
        while self._pos < len(self._source) and self._source[self._pos].isdigit():
            chars.append(self._advance())
        return Token(TokenType.INTEGER, "".join(chars), start_line, start_col)

    def _read_word(self, start_line: int, start_col: int) -> Token:
        """Read an identifier or keyword."""
        chars: list[str] = []
        while self._pos < len(self._source) and (
            self._source[self._pos].isalnum() or self._source[self._pos] == "_"
        ):
            chars.append(self._advance())
        word = "".join(chars)
        if word in KEYWORDS:
            return Token(TokenType.KEYWORD, word, start_line, start_col)
        return Token(TokenType.IDENTIFIER, word, start_line, start_col)

    def _read_operator(self, start_line: int, start_col: int) -> Token:
        """Read a relational or assignment operator.

        Handles two-character operators (``<=``, ``>=``, ``==``, ``!=``)
        and single-character forms (``<``, ``>``, ``=``).
        """
        ch = self._advance()
        next_ch = self._peek()
        two_char = ch + (next_ch or "")
        if two_char in _TWO_CHAR_REL_OPS:
            self._advance()
            return Token(TokenType.RELATIONAL_OPERATOR, two_char, start_line, start_col)
        if ch in _ONE_CHAR_REL_OPS:
            return Token(TokenType.RELATIONAL_OPERATOR, ch, start_line, start_col)
        if ch == "=":
            return Token(TokenType.OPERATOR, ch, start_line, start_col)
        if ch == "!":
            raise LexError("expected '!=' for inequality, got '!'", start_line, start_col)
        raise LexError(f"invalid operator {ch!r}", start_line, start_col)
