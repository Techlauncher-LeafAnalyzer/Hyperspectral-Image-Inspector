"""User-defined spectral index formulas, parsed safely without ``eval``.

A formula refers to reflectance at a wavelength as ``R<nm>`` (``R800``,
``r531.5``) and combines those with numbers, ``+ - * / ^``, parentheses and a
small set of functions. Requested wavelengths need not exist exactly in the
cube: each one is mapped to the nearest available band, and only wavelengths
outside the camera's range are rejected.

Supported syntax, in increasing precedence::

    a + b   a - b
    a * b   a / b   2R800 (implicit multiplication)
    -a      +a
    a ^ b   a ** b  (right-associative, so 2^3^2 = 2^9)
    sqrt(a) sq(a) abs(a) ln(a) log(a) log10(a) exp(a) min(a, b) max(a, b)
    pi  e

Typographic forms pasted from a document are accepted too: ``×``, ``·``,
``÷``, ``−``, ``√a`` and the ``²``/``³`` superscripts.

This module imports no Qt classes. Evaluation follows NumPy semantics; the
caller decides how non-finite results (division by zero, sqrt of negatives)
are displayed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .errors import VisualizationError

# Display text for the editor's function palette, in palette order.
CUSTOM_INDEX_FUNCTIONS: Mapping[str, str] = {
    "sqrt": "Square root",
    "sq": "Square (x²)",
    "abs": "Absolute value",
    "ln": "Natural logarithm",
    "log10": "Base-10 logarithm",
    "exp": "Exponential (eˣ)",
    "min": "Smaller of two values",
    "max": "Larger of two values",
}

_UNARY_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sqrt": np.sqrt,
    "sq": np.square,
    "abs": np.abs,
    "ln": np.log,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
}
_BINARY_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "min": np.minimum,
    "max": np.maximum,
    "pow": np.power,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}

_TYPOGRAPHIC = str.maketrans(
    {"×": "*", "·": "*", "∙": "*", "÷": "/", "−": "-", "–": "-"}
)
_TOKEN_PATTERN = re.compile(
    r"""
    (?P<space>\s+)
    | (?P<band>[Rr](?P<nm>\d+(?:\.\d+)?)(?![A-Za-z0-9_.]))
    | (?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
    | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<op>\*\*|[-+*/^(),√²³])
    """,
    re.VERBOSE,
)


class CustomIndexError(VisualizationError):
    """A custom index formula is malformed or cannot use this image's bands.

    ``position`` is the zero-based character offset of the problem in the
    formula text when known, so an editor can point at it.
    """

    def __init__(self, message: str, position: int | None = None) -> None:
        super().__init__(message)
        self.position = position


@dataclass(frozen=True, slots=True)
class ResolvedBand:
    """A requested ``R<nm>`` term and the actual cube band it reads."""

    token: str
    requested_nm: float
    band_index: int
    wavelength_nm: float

    @property
    def offset_nm(self) -> float:
        return self.wavelength_nm - self.requested_nm


# --------------------------------------------------------------------- #
# Syntax tree                                                           #
# --------------------------------------------------------------------- #

Environment = Mapping[float, np.ndarray]


class _Node:
    def evaluate(self, bands: Environment) -> np.ndarray | float:
        raise NotImplementedError

    def latex(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class _Number(_Node):
    value: float
    text: str

    def evaluate(self, bands: Environment) -> float:
        return self.value

    def latex(self) -> str:
        return self.text


@dataclass(frozen=True)
class _Constant(_Node):
    name: str

    def evaluate(self, bands: Environment) -> float:
        return _CONSTANTS[self.name]

    def latex(self) -> str:
        return r"\pi" if self.name == "pi" else "e"


@dataclass(frozen=True)
class _Band(_Node):
    wavelength_nm: float
    text: str

    def evaluate(self, bands: Environment) -> np.ndarray:
        return bands[self.wavelength_nm]

    def latex(self) -> str:
        return f"R_{{{self.text}}}"


@dataclass(frozen=True)
class _Group(_Node):
    """Parentheses the user typed, kept so the preview mirrors the input."""

    inner: _Node

    def evaluate(self, bands: Environment) -> np.ndarray | float:
        return self.inner.evaluate(bands)

    def latex(self) -> str:
        return rf"\left({self.inner.latex()}\right)"


@dataclass(frozen=True)
class _Negate(_Node):
    operand: _Node
    sign: str

    def evaluate(self, bands: Environment) -> np.ndarray | float:
        value = self.operand.evaluate(bands)
        return -value if self.sign == "-" else value

    def latex(self) -> str:
        return f"{self.sign}{self.operand.latex()}"


@dataclass(frozen=True)
class _Binary(_Node):
    operator: str
    left: _Node
    right: _Node

    def evaluate(self, bands: Environment) -> np.ndarray | float:
        left = self.left.evaluate(bands)
        right = self.right.evaluate(bands)
        if self.operator == "+":
            return np.add(left, right)
        if self.operator == "-":
            return np.subtract(left, right)
        if self.operator == "*":
            return np.multiply(left, right)
        if self.operator == "/":
            return np.divide(left, right)
        return np.power(left, right)

    def latex(self) -> str:
        if self.operator == "/":
            return rf"\frac{{{_unwrap(self.left).latex()}}}{{{_unwrap(self.right).latex()}}}"
        if self.operator == "^":
            base = self.left.latex()
            if not isinstance(self.left, (_Number, _Constant, _Band, _Group, _Call)):
                base = rf"\left({base}\right)"
            return f"{{{base}}}^{{{_unwrap(self.right).latex()}}}"
        symbol = r"\cdot" if self.operator == "*" else self.operator
        return f"{self.left.latex()} {symbol} {self.right.latex()}"


@dataclass(frozen=True)
class _Call(_Node):
    name: str
    arguments: tuple[_Node, ...]

    def evaluate(self, bands: Environment) -> np.ndarray | float:
        values = [argument.evaluate(bands) for argument in self.arguments]
        if self.name in _UNARY_FUNCTIONS:
            return _UNARY_FUNCTIONS[self.name](*values)
        return _BINARY_FUNCTIONS[self.name](*values)

    def latex(self) -> str:
        inner = [_unwrap(argument).latex() for argument in self.arguments]
        if self.name == "sqrt":
            return rf"\sqrt{{{inner[0]}}}"
        if self.name == "sq":
            return rf"{{\left({inner[0]}\right)}}^{{2}}"
        if self.name == "abs":
            return rf"\left|{inner[0]}\right|"
        if self.name == "log10":
            return rf"\log_{{10}}\left({inner[0]}\right)"
        command = {"log": r"\ln", "ln": r"\ln", "exp": r"\exp", "min": r"\min",
                   "max": r"\max", "pow": r"\mathrm{pow}"}[self.name]
        return rf"{command}\left({', '.join(inner)}\right)"


def _unwrap(node: _Node) -> _Node:
    """Drop typed parentheses where layout already groups (fractions, powers)."""

    return node.inner if isinstance(node, _Group) else node


# --------------------------------------------------------------------- #
# Parser                                                                #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str
    position: int


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN_PATTERN.match(text, position)
        if match is None:
            raise CustomIndexError(
                f"Unexpected character {text[position]!r} at position {position + 1}.",
                position,
            )
        kind = match.lastgroup
        if kind == "nm":  # Inner group of `band`; report the outer token.
            kind = "band"
        if kind != "space":
            tokens.append(_Token(kind, match.group(), position))
        position = match.end()
    tokens.append(_Token("end", "", len(text)))
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self._tokens = _tokenize(text)
        self._index = 0

    def parse(self) -> _Node:
        if self._peek().kind == "end":
            raise CustomIndexError("Enter a formula, for example (R800-R670)/(R800+R670).", 0)
        node = self._expression()
        token = self._peek()
        if token.kind != "end":
            raise self._unexpected(token)
        return node

    # expression := term (('+' | '-') term)*
    def _expression(self) -> _Node:
        node = self._term()
        while self._peek().text in ("+", "-"):
            operator = self._advance().text
            node = _Binary(operator, node, self._term())
        return node

    # term := unary (('*' | '/' | <implicit>) unary)*
    def _term(self) -> _Node:
        node = self._unary()
        while True:
            token = self._peek()
            if token.text in ("*", "/"):
                self._advance()
                node = _Binary(token.text, node, self._unary())
            elif self._starts_operand(token):
                node = _Binary("*", node, self._power())
            else:
                return node

    # unary := ('+' | '-') unary | power
    def _unary(self) -> _Node:
        if self._peek().text in ("+", "-"):
            sign = self._advance().text
            return _Negate(self._unary(), sign)
        return self._power()

    # power := primary ('²' | '³' | ('^' | '**') unary)?
    def _power(self) -> _Node:
        base = self._primary()
        if self._peek().text in ("²", "³"):
            exponent = "2" if self._advance().text == "²" else "3"
            return _Binary("^", base, _Number(float(exponent), exponent))
        if self._peek().text in ("^", "**"):
            self._advance()
            return _Binary("^", base, self._unary())
        return base

    def _primary(self) -> _Node:
        token = self._advance()
        if token.kind == "number":
            return _Number(float(token.text), token.text)
        if token.kind == "band":
            text = token.text[1:]
            return _Band(float(text), text)
        if token.text == "(":
            inner = self._expression()
            self._expect(")", token)
            return _Group(inner)
        if token.text == "√":
            return _Call("sqrt", (self._power(),))
        if token.kind == "name":
            name = token.text.lower()
            if name in _CONSTANTS and self._peek().text != "(":
                return _Constant(name)
            if name in _UNARY_FUNCTIONS or name in _BINARY_FUNCTIONS:
                return self._call(name, token)
            if name[0] in "rR" and name[1:2].isdigit():
                raise CustomIndexError(
                    f"{token.text!r} is not a band. Write a wavelength like R800 "
                    "and separate terms with an operator.",
                    token.position,
                )
            known = ", ".join(CUSTOM_INDEX_FUNCTIONS)
            raise CustomIndexError(
                f"Unknown name {token.text!r} at position {token.position + 1}. "
                f"Bands are written R<nm> (e.g. R800); functions are {known}.",
                token.position,
            )
        raise self._unexpected(token)

    def _call(self, name: str, token: _Token) -> _Node:
        arity = 1 if name in _UNARY_FUNCTIONS else 2
        if self._peek().text != "(":
            raise CustomIndexError(
                f"{token.text} needs parentheses, e.g. {token.text}("
                f"{'R800' if arity == 1 else 'R800, R670'}).",
                token.position,
            )
        opening = self._advance()
        arguments = [self._expression()]
        while self._peek().text == ",":
            self._advance()
            arguments.append(self._expression())
        self._expect(")", opening)
        if len(arguments) != arity:
            raise CustomIndexError(
                f"{token.text} takes {arity} argument{'s' if arity > 1 else ''}, "
                f"got {len(arguments)}.",
                token.position,
            )
        return _Call(name, tuple(arguments))

    @staticmethod
    def _starts_operand(token: _Token) -> bool:
        return token.kind in ("number", "band", "name") or token.text in ("(", "√")

    def _expect(self, text: str, opening: _Token) -> None:
        token = self._peek()
        if token.text != text:
            if token.kind == "end":
                raise CustomIndexError(
                    f"The '(' at position {opening.position + 1} is never closed.",
                    opening.position,
                )
            raise self._unexpected(token)
        self._advance()

    def _peek(self) -> _Token:
        return self._tokens[self._index]

    def _advance(self) -> _Token:
        token = self._tokens[self._index]
        if token.kind != "end":
            self._index += 1
        return token

    @staticmethod
    def _unexpected(token: _Token) -> CustomIndexError:
        if token.kind == "end":
            return CustomIndexError("The formula ends unexpectedly.", token.position)
        return CustomIndexError(
            f"Unexpected {token.text!r} at position {token.position + 1}.",
            token.position,
        )


def _collect_bands(node: _Node, found: dict[float, str]) -> None:
    if isinstance(node, _Band):
        found.setdefault(node.wavelength_nm, f"R{node.text}")
    for child in vars(node).values():
        if isinstance(child, _Node):
            _collect_bands(child, found)
        elif isinstance(child, tuple):
            for item in child:
                _collect_bands(item, found)


# --------------------------------------------------------------------- #
# Public API                                                            #
# --------------------------------------------------------------------- #


class CustomIndexFormula:
    """A parsed, validated formula ready to resolve against a cube.

    Construct with :func:`parse_custom_index`. Instances are immutable.
    """

    def __init__(self, text: str, root: _Node) -> None:
        self._text = text.strip()
        self._root = root
        found: dict[float, str] = {}
        _collect_bands(root, found)
        if not found:
            raise CustomIndexError(
                "Use at least one band, written R<nm> (e.g. R800).", None
            )
        self._bands = tuple(found.items())

    @property
    def text(self) -> str:
        return self._text

    @property
    def band_tokens(self) -> tuple[str, ...]:
        """Each distinct ``R<nm>`` term, in first-use order."""
        return tuple(token for _, token in self._bands)

    @property
    def requested_wavelengths_nm(self) -> tuple[float, ...]:
        return tuple(wavelength for wavelength, _ in self._bands)

    def latex(self) -> str:
        """Matplotlib-mathtext source for a typeset preview of the formula."""
        return self._root.latex()

    def resolve(self, wavelengths_nm: Sequence[float] | np.ndarray) -> tuple[ResolvedBand, ...]:
        """Map each requested wavelength to the nearest available band.

        Raises:
            CustomIndexError: A requested wavelength lies outside the cube's
                range by more than one typical band spacing.
        """

        wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
        if wavelengths.ndim != 1 or wavelengths.size == 0 or not np.isfinite(wavelengths).all():
            raise CustomIndexError("This image has no usable wavelength metadata.")
        low, high = float(wavelengths.min()), float(wavelengths.max())
        slack = float(np.median(np.abs(np.diff(np.sort(wavelengths))))) if wavelengths.size > 1 else 0.0
        resolved = []
        for requested, token in self._bands:
            if not low - slack <= requested <= high + slack:
                raise CustomIndexError(
                    f"{token} is outside this image's {low:g}–{high:g} nm range."
                )
            index = int(np.abs(wavelengths - requested).argmin())
            resolved.append(ResolvedBand(token, requested, index, float(wavelengths[index])))
        return tuple(resolved)

    def evaluate(self, bands: Mapping[float, np.ndarray]) -> np.ndarray:
        """Evaluate with ``bands`` keyed by requested wavelength in nm.

        Returns a ``float32`` array broadcast to the band arrays' shape.
        Invalid arithmetic produces NaN/inf rather than raising.
        """

        missing = [token for wavelength, token in self._bands if wavelength not in bands]
        if missing:
            raise CustomIndexError(f"No values supplied for {', '.join(missing)}.")
        shape = np.shape(next(iter(bands.values())))
        with np.errstate(all="ignore"):
            values = self._root.evaluate(
                {wavelength: np.asarray(array, dtype=np.float64) for wavelength, array in bands.items()}
            )
            return np.broadcast_to(np.asarray(values, dtype=np.float32), shape).copy()


def parse_custom_index(text: str) -> CustomIndexFormula:
    """Parse ``text`` into a formula, raising :class:`CustomIndexError` on failure."""

    # One-for-one replacements, so reported positions still match `text`.
    return CustomIndexFormula(text, _Parser(text.translate(_TYPOGRAPHIC)).parse())
