# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Typed filter expressions as an alternative to hand-written SQL strings.

Raw SQL filters are parsed without any knowledge of the caller's intent:
an integer column compared with a fractional literal fails to plan
(see `issue 9317 <https://github.com/lance-format/lance/issues/9317>`_),
a double-typed literal against a ``Float32`` column silently casts the
column and skips its scalar index
(see `issue 9318 <https://github.com/lance-format/lance/issues/9318>`_),
and a boolean column compared with a boolean expression fails in one
operand order but not the other
(see `issue 9319 <https://github.com/lance-format/lance/issues/9319>`_).
The ``pyarrow.compute.Expression`` path (``scanner(filter=...)`` via
Substrait) avoids some of these traps but cannot express nested field
access, string functions, or mixed integer/float comparisons at all:
pyarrow refuses to serialize the implicit safe cast, and the planner
rejects nested references in Substrait filters.

This module offers a small typed builder instead. Expressions carry
Python values with their intended types, and rendering them against the
dataset schema applies three index-preserving rules:

* a fractional literal against an integer column is rewritten to the
  equivalent integer bound (``i > 1.5`` becomes ``i > 1``), which the
  SQL planner rejects outright;
* a literal against a ``Float32`` column is spelled
  ``CAST(<literal> AS float)`` so the column keeps its type and its
  index, instead of being cast to ``double``;
* a comparison between two boolean expressions is spelled out with
  ``AND``/``OR``/``NOT`` so it plans regardless of operand order.

Example:

```python
import lance
from lance.filter import col

ds = lance.dataset("data.lance")
# No `schema=` needed here: `scanner` supplies the dataset schema
# itself when it is given an expression.
ds.scanner(filter=(col("i") > 1.5) & col("s").starts_with("a"))
```

Rendering without a schema (``to_sql(expr)``) is supported but
best-effort: literals are spelled generically and no rewrite fires, so
a ``Float32`` column pays the same column cast a hand-written double
literal would pay. Prefer passing expressions to ``scanner`` (or to
``to_sql`` with the dataset schema).

Only constructs with exact filter semantics are offered. Division,
modulo, and power operators raise instead of guessing between
truncated and floor semantics, and ``NaN`` literals raise instead of
picking an ordering. Anything this module cannot spell raises
``FilterError`` with the reason. Timezone-aware timestamps, decimals,
and binary data are not representable yet.

Float filters need at least ``pylance 12.0.0b7`` (the release that
fixed signed-zero literal handling in `PR 6236
<https://github.com/lance-format/lance/pull/6236>`_); every other
spelling this module emits already plans on ``pylance 9.0.0``.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["Expr", "FilterError", "col", "lit", "to_sql"]

# A filter that keeps no row, null included: the exact positive rendering
# of an unsatisfiable predicate such as `i = 1.5` on an integer column.
_NULL_BOOLEAN = "CAST(NULL AS boolean)"

_TIMESTAMP_PRECISION = {"s": 0, "ms": 3, "us": 6, "ns": 9}

_MIRRORED = {
    "=": "=",
    "!=": "!=",
    "<": ">",
    "<=": ">=",
    ">": "<",
    ">=": "<=",
}

_NEGATED = {
    "=": "!=",
    "!=": "=",
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
}


class FilterError(ValueError):
    """An expression has no exact Lance filter spelling."""


def _quote(segment: str) -> str:
    if not segment:
        raise FilterError("empty field name in column path")
    return "`" + segment.replace("`", "``") + "`"


def _quote_path(path: tuple[str, ...]) -> str:
    return ".".join(_quote(segment) for segment in path)


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _mirror(op: str, mirror: bool) -> str:
    return _MIRRORED[op] if mirror else op


class Expr:
    """A typed filter expression; build with :func:`col` and :func:`lit`."""

    def __and__(self, other: object) -> Expr:
        return _And((self, _as_boolean(other)))

    def __rand__(self, other: object) -> Expr:
        return _And((_as_boolean(other), self))

    def __or__(self, other: object) -> Expr:
        return _Or((self, _as_boolean(other)))

    def __ror__(self, other: object) -> Expr:
        return _Or((_as_boolean(other), self))

    def __invert__(self) -> Expr:
        return _Not(self)

    def _boolean(self) -> bool:
        """Whether this node is known to evaluate to a boolean."""
        return False

    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return _Compare("=", self, _as_value(other))

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        return _Compare("!=", self, _as_value(other))

    def __lt__(self, other: object) -> Expr:
        return _Compare("<", self, _as_value(other))

    def __le__(self, other: object) -> Expr:
        return _Compare("<=", self, _as_value(other))

    def __gt__(self, other: object) -> Expr:
        return _Compare(">", self, _as_value(other))

    def __ge__(self, other: object) -> Expr:
        return _Compare(">=", self, _as_value(other))

    def __add__(self, other: object) -> Expr:
        return _Arith("+", self, _as_value(other))

    def __radd__(self, other: object) -> Expr:
        return _Arith("+", _as_value(other), self)

    def __sub__(self, other: object) -> Expr:
        return _Arith("-", self, _as_value(other))

    def __rsub__(self, other: object) -> Expr:
        return _Arith("-", _as_value(other), self)

    def __mul__(self, other: object) -> Expr:
        return _Arith("*", self, _as_value(other))

    def __rmul__(self, other: object) -> Expr:
        return _Arith("*", _as_value(other), self)

    def __truediv__(self, other: object) -> Expr:
        raise FilterError(
            "division has no single exact filter spelling: Lance truncates"
            " while other engines divide exactly; spell the comparison out"
            " in SQL if that is what was meant"
        )

    def __mod__(self, other: object) -> Expr:
        raise FilterError(
            "modulo is truncated in Lance but floored in engines such as"
            " Polars; spell the corrected form out in SQL if that is what"
            " was meant"
        )

    def __pow__(self, other: object) -> Expr:
        raise FilterError("power expressions are not supported in filters")

    def is_null(self) -> Expr:
        """Rows where this value is null."""
        return _IsNull(self, negated=False)

    def is_not_null(self) -> Expr:
        """Rows where this value is not null."""
        return _IsNull(self, negated=True)

    def is_nan(self) -> Expr:
        """Rows where this float value is NaN."""
        return _Func("isnan", (self,))

    def isin(self, values: list[object]) -> Expr:
        """Rows where this value matches one of `values`.

        ``None`` entries never match, mirroring ``IN`` filter semantics,
        so they are dropped. An empty list matches no row.
        """
        if not isinstance(values, list):
            raise TypeError(
                f"isin() takes a list of literals, got {type(values).__name__}"
            )
        kept = [_as_literal(v) for v in values if v is not None]
        for item in kept:
            if isinstance(item.value, float) and math.isnan(item.value):
                raise FilterError(
                    "NaN never matches in filters; test it with is_nan() instead"
                )
        return _InList(self, tuple(kept))

    def between(self, low: object, high: object) -> Expr:
        """Rows where this value lies in [`low`, `high`]."""
        return _Between(self, _as_literal(low), _as_literal(high))

    def starts_with(self, prefix: str) -> Expr:
        """Rows where this string starts with `prefix`."""
        return _string_func("starts_with", self, prefix)

    def ends_with(self, suffix: str) -> Expr:
        """Rows where this string ends with `suffix`."""
        return _string_func("ends_with", self, suffix)

    def contains(self, needle: str) -> Expr:
        """Rows where this string contains `needle`."""
        return _string_func("contains", self, needle)

    def field(self, name: str) -> Expr:
        """The struct field `name` of this column."""
        if not isinstance(self, _Col):
            raise FilterError("field() is only supported on a column")
        if not isinstance(name, str) or not name:
            raise FilterError("struct field name must be a non-empty string")
        return _Col(self.path + (name,))


def _string_func(name: str, value: Expr, affix: object) -> Expr:
    if not isinstance(affix, str):
        raise FilterError(f"{name}() takes a string literal")
    return _Func(name, (value, _Lit(affix)))


def _as_value(other: object) -> Expr:
    if isinstance(other, Expr):
        return other
    return _as_literal(other)


def _as_boolean(other: object) -> Expr:
    value = _as_value(other)
    if isinstance(value, _Lit) and not isinstance(value.value, bool):
        raise FilterError(
            f"boolean logic takes boolean expressions, got {value.value!r}"
        )
    return value


def _as_literal(other: object) -> _Lit:
    if isinstance(other, _Lit):
        return other
    return lit(other)


class _Col(Expr):
    __slots__ = ("path",)

    def __init__(self, path: tuple[str, ...]):
        self.path = path

    def __repr__(self) -> str:
        return f"col({'.'.join(self.path)!r})"


class _Lit(Expr):
    __slots__ = ("value",)

    def __init__(self, value: bool | int | float | str | dt.date | dt.datetime):
        self.value = value

    def __repr__(self) -> str:
        return f"lit({self.value!r})"


class _Compare(Expr):
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def _boolean(self) -> bool:
        return True


class _And(Expr):
    __slots__ = ("terms",)

    def __init__(self, terms: tuple[Expr, ...]):
        self.terms = terms

    def _boolean(self) -> bool:
        return True


class _Or(Expr):
    __slots__ = ("terms",)

    def __init__(self, terms: tuple[Expr, ...]):
        self.terms = terms

    def _boolean(self) -> bool:
        return True


class _Not(Expr):
    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def _boolean(self) -> bool:
        return True


class _IsNull(Expr):
    __slots__ = ("value", "negated")

    def __init__(self, value: Expr, negated: bool):
        self.value = value
        self.negated = negated

    def _boolean(self) -> bool:
        return True


class _InList(Expr):
    __slots__ = ("value", "items")

    def __init__(self, value: Expr, items: tuple[_Lit, ...]):
        self.value = value
        self.items = items

    def _boolean(self) -> bool:
        return True


class _Between(Expr):
    __slots__ = ("value", "low", "high")

    def __init__(self, value: Expr, low: _Lit, high: _Lit):
        self.value = value
        self.low = low
        self.high = high

    def _boolean(self) -> bool:
        return True


class _Arith(Expr):
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right


class _Func(Expr):
    __slots__ = ("name", "args")

    def __init__(self, name: str, args: tuple[Expr, ...]):
        self.name = name
        self.args = args

    def _boolean(self) -> bool:
        return True


def col(name: str) -> Expr:
    """Reference the column `name`; ``col("a.b")`` reads struct field ``b``."""
    if not isinstance(name, str) or not name:
        raise FilterError("column name must be a non-empty string")
    segments = tuple(name.split("."))
    if any(not segment for segment in segments):
        raise FilterError(f"invalid column path {name!r}")
    return _Col(segments)


def lit(value: object) -> Expr:
    """A literal filter value.

    Supported types are ``bool``, ``int``, ``float`` (except NaN),
    ``str``, ``datetime.date`` and naive ``datetime.datetime``.
    Use ``col(...).is_null()`` instead of a ``None`` literal.
    """
    if isinstance(value, bool):
        return _Lit(value)
    if isinstance(value, int):
        return _Lit(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise FilterError(
                "NaN literals have no exact filter spelling;"
                " test them with is_nan() instead"
            )
        return _Lit(value)
    if isinstance(value, str):
        return _Lit(value)
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            raise FilterError("timezone-aware datetimes are not supported in filters")
        return _Lit(value)
    if isinstance(value, dt.date):
        return _Lit(value)
    raise TypeError(
        "filter literals support bool, int, float, str, date and naive"
        f" datetime, got {type(value).__name__}"
    )


def to_sql(expr: Expr, schema: pa.Schema | None = None) -> str:
    """Render `expr` as a Lance SQL filter.

    Args:
        expr: A boolean expression built with :func:`col` and :func:`lit`.
        schema: The dataset schema. With one, literals are typed for
            their column (``Float32`` literals, integer bounds for
            fractional comparisons) and unknown columns are rejected.
            Without one, literals are spelled generically and no rewrite
            fires; prefer passing expressions to ``scanner`` instead,
            which supplies the schema itself.
    """
    if not isinstance(expr, Expr):
        raise TypeError(f"to_sql() takes an Expr, got {type(expr).__name__}")
    return _Renderer(schema).boolean(expr)


class _Renderer:
    def __init__(self, schema: pa.Schema | None):
        self._schema = schema

    def boolean(self, expr: Expr) -> str:
        if isinstance(expr, _Col):
            target = self.column_type(expr)
            if target is not None and not pa.types.is_boolean(target):
                raise FilterError(
                    f"column {'.'.join(expr.path)!r} is {target}, not a boolean filter"
                )
            return _quote_path(expr.path)
        if isinstance(expr, _Lit):
            if not isinstance(expr.value, bool):
                raise FilterError(f"{expr.value!r} is not a boolean filter")
            return "TRUE" if expr.value else _NULL_BOOLEAN
        if isinstance(expr, _Not):
            return self.negation(expr.inner)
        if isinstance(expr, _And):
            return "(" + " AND ".join(self.boolean(t) for t in expr.terms) + ")"
        if isinstance(expr, _Or):
            return "(" + " OR ".join(self.boolean(t) for t in expr.terms) + ")"
        if isinstance(expr, _Compare):
            return self.comparison(expr.op, expr.left, expr.right)
        if isinstance(expr, _IsNull):
            keyword = "IS NOT NULL" if expr.negated else "IS NULL"
            return f"({self.value(expr.value)} {keyword})"
        if isinstance(expr, _InList):
            return self.membership(expr.value, expr.items, negated=False)
        if isinstance(expr, _Between):
            return self.between(expr.value, expr.low, expr.high, False)
        if isinstance(expr, _Func):
            return self.call(expr.name, expr.args)
        raise FilterError(f"{expr!r} is not a boolean filter")

    def negation(self, inner: Expr) -> str:
        if isinstance(inner, _Not):
            return self.boolean(inner.inner)
        if isinstance(inner, _Lit):
            if not isinstance(inner.value, bool):
                raise FilterError(f"{inner.value!r} is not a boolean filter")
            return f"{_NULL_BOOLEAN}" if inner.value else "TRUE"
        if isinstance(inner, _Compare):
            # Push the negation through the comparison: the rewrite rules
            # in `comparison` are filter-equivalent but not value-equivalent
            # (`i != 1.5` renders as `i IS NOT NULL`, whose SQL negation
            # would keep null rows), while the mirrored comparison is exact.
            return self.comparison(_NEGATED[inner.op], inner.left, inner.right)
        if isinstance(inner, _InList):
            if not inner.items:
                # An empty membership keeps no row, null included, so its
                # negation keeps the non-null rows to match `~is_in([])`.
                return f"({self.value(inner.value)} IS NOT NULL)"
            return self.membership(inner.value, inner.items, negated=True)
        if isinstance(inner, _Between):
            return self.between(inner.value, inner.low, inner.high, True)
        if isinstance(inner, _IsNull):
            keyword = "IS NULL" if inner.negated else "IS NOT NULL"
            return f"({self.value(inner.value)} {keyword})"
        return f"(NOT {self.boolean(inner)})"

    def comparison(self, op: str, left: Expr, right: Expr) -> str:
        if self._is_boolean_side(left) and self._is_boolean_side(right):
            if isinstance(left, _Lit) or isinstance(right, _Lit):
                # `flag = TRUE` plans as written; only two computed sides
                # need the expanded spelling below.
                return f"({self.boolean(left)} {op} {self.boolean(right)})"
            return self.boolean_logic(op, left, right)
        rewritten = self.numeric_comparison(op, left, right)
        if rewritten is not None:
            return rewritten
        return self._plain_comparison(op, left, right)

    def _plain_comparison(self, op: str, left: Expr, right: Expr) -> str:
        # Literals take the other side's column type so a `Float32`
        # column keeps its type (and its index) instead of being cast.
        left_sql = self._operand(left, self.value_type(right))
        right_sql = self._operand(right, self.value_type(left))
        return f"({left_sql} {op} {right_sql})"

    def _operand(self, side: Expr, target: pa.DataType | None) -> str:
        if isinstance(side, _Lit):
            return self.typed_literal(side.value, target)
        return self.value(side)

    def _is_boolean_side(self, side: Expr) -> bool:
        if isinstance(side, _Lit):
            return isinstance(side.value, bool)
        if side._boolean():
            return True
        target = self.value_type(side)
        return target is not None and pa.types.is_boolean(target)

    def boolean_logic(self, op: str, left: Expr, right: Expr) -> str:
        """``=``/``!=`` between two computed boolean expressions.

        The planner only converts a bare literal operand to the other
        side's type, so ``flag != (id > 0)`` fails while the swapped form
        plans. Spelling the logic out plans in either order.
        """
        if op not in ("=", "!="):
            raise FilterError(f"boolean expressions support = and !=, got {op}")
        a, b = self.boolean(left), self.boolean(right)
        if op == "!=":
            return f"(({a} AND NOT ({b})) OR ((NOT ({a})) AND ({b})))"
        return f"(({a} AND ({b})) OR ((NOT ({a})) AND (NOT ({b}))))"

    def numeric_comparison(self, op: str, left: Expr, right: Expr) -> str | None:
        for column, literal, mirror in ((left, right, False), (right, left, True)):
            if isinstance(column, _Col) and isinstance(literal, _Lit):
                target = self.column_type(column)
                if target is None:
                    continue
                return self.typed_literal_comparison(
                    op, column, literal, target, mirror
                )
        return None

    def typed_literal_comparison(
        self, op: str, column: _Col, literal: _Lit, target: pa.DataType, mirror: bool
    ) -> str | None:
        value = literal.value
        name = ".".join(column.path)
        if pa.types.is_boolean(target):
            if not isinstance(value, bool):
                raise FilterError(f"column {name!r} is boolean, got {value!r}")
            return None
        if _is_integer_type(target):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FilterError(f"column {name!r} is {target}, got {value!r}")
            if isinstance(value, float):
                return self.float_against_int(op, column, value, target, mirror)
            _check_range(value, target)
            return None
        if pa.types.is_floating(target):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FilterError(f"column {name!r} is {target}, got {value!r}")
            return None
        if pa.types.is_string(target) or pa.types.is_large_string(target):
            if not isinstance(value, str):
                raise FilterError(f"column {name!r} is {target}, got {value!r}")
            return None
        if pa.types.is_date32(target):
            if not isinstance(value, dt.date) or isinstance(value, dt.datetime):
                raise FilterError(f"column {name!r} is {target}, takes a date")
            return None
        if pa.types.is_timestamp(target):
            if not isinstance(value, dt.datetime):
                raise FilterError(f"column {name!r} is {target}, takes a datetime")
            if target.tz is not None:
                raise FilterError(
                    "timezone-aware timestamp columns are not yet supported"
                )
            return None
        raise FilterError(
            f"column {name!r} is {target}, which has no filter spelling yet"
        )

    def float_against_int(
        self, op: str, column: _Col, value: float, target: pa.DataType, mirror: bool
    ) -> str:
        """An integer column against a float literal.

        The planner rejects the fractional spelling instead of coercing
        it, so values become the equivalent integer bound and stay on the
        column, keeping any scalar index.
        """
        name = _quote_path(column.path)
        cop = _mirror(op, mirror)
        if math.isinf(value):
            return self._infinite_bound(name, cop, value > 0)
        if value.is_integer():
            clamped = _clamp_bound(int(value), cop, target)
            if clamped is None:
                return f"{_NULL_BOOLEAN}"
            if clamped == "all":
                return f"({name} IS NOT NULL)"
            return f"({name} {cop} {clamped})"
        if cop in ("=", "!="):
            # No integer equals a fractional value; `!=` still keeps every
            # non-null row, mirroring three-valued filter logic.
            if cop == "=":
                return f"{_NULL_BOOLEAN}"
            return f"({name} IS NOT NULL)"
        bound = math.floor(value) if cop in (">", "<=") else math.ceil(value)
        clamped = _clamp_bound(bound, cop, target)
        if clamped is None:
            return f"{_NULL_BOOLEAN}"
        if clamped == "all":
            return f"({name} IS NOT NULL)"
        return f"({name} {cop} {clamped})"

    def _infinite_bound(self, name: str, op: str, positive: bool) -> str:
        if op == "=":
            return f"{_NULL_BOOLEAN}"
        if op == "!=":
            return f"({name} IS NOT NULL)"
        if op in (">", ">="):
            return f"({name} IS NOT NULL)" if not positive else f"{_NULL_BOOLEAN}"
        return f"({name} IS NOT NULL)" if positive else f"{_NULL_BOOLEAN}"

    def membership(self, value: Expr, items: tuple[_Lit, ...], negated: bool) -> str:
        target = self.value_type(value)
        if isinstance(value, _Col) and target is not None and _is_integer_type(target):
            kept: list[str] = []
            for item in items:
                needle = item.value
                if isinstance(needle, bool) or not isinstance(needle, (int, float)):
                    raise FilterError(
                        f"column {'.'.join(value.path)!r} is {target}, got {needle!r}"
                    )
                bound = int(needle) if isinstance(needle, float) else needle
                if isinstance(needle, float) and not needle.is_integer():
                    continue  # fractional needles match no integer row
                if not _in_range(bound, target):
                    continue  # out-of-range needles match no integer row
                kept.append(str(bound))
            return self._in_list(value, kept, negated)
        rendered = [self.typed_literal(item.value, target) for item in items]
        return self._in_list(value, rendered, negated)

    def _in_list(self, value: Expr, rendered: list[str], negated: bool) -> str:
        name = self.value(value)
        if not rendered:
            if negated:
                return f"({name} IS NOT NULL)"
            return f"{_NULL_BOOLEAN}"
        body = f"{name} IN ({', '.join(rendered)})"
        if negated:
            return f"(NOT ({body}))"
        return f"({body})"

    def between(self, value: Expr, low: _Lit, high: _Lit, negated: bool) -> str:
        target = self.value_type(value)
        if isinstance(value, _Col) and target is not None and _is_integer_type(target):
            return self._int_between(value, low.value, high.value, target, negated)
        body = (
            f"{self.value(value)} BETWEEN"
            f" {self.typed_literal(low.value, target)}"
            f" AND {self.typed_literal(high.value, target)}"
        )
        if negated:
            return f"(NOT ({body}))"
        return f"({body})"

    def _int_between(
        self,
        value: _Col,
        low: int | float,
        high: int | float,
        target: pa.DataType,
        negated: bool,
    ) -> str:
        name = _quote_path(value.path)
        bounds = _integer_bounds(target)
        floored = _int_bound(low, math.ceil, bounds, low_side=True)
        ceiling = _int_bound(high, math.floor, bounds, low_side=False)
        if floored is None or ceiling is None:
            # An unsatisfiable range keeps no row; its negation keeps the
            # non-null rows, mirroring `~is_between`.
            if negated:
                return f"({name} IS NOT NULL)"
            return f"{_NULL_BOOLEAN}"
        if floored == "all" and ceiling == "all":
            if negated:
                return f"{_NULL_BOOLEAN}"
            return f"({name} IS NOT NULL)"
        if floored == "all":
            return self._half_bound(name, "<=", ceiling, negated)
        if ceiling == "all":
            return self._half_bound(name, ">=", floored, negated)
        if floored > ceiling:
            if negated:
                return f"({name} IS NOT NULL)"
            return f"{_NULL_BOOLEAN}"
        body = f"{name} BETWEEN {floored} AND {ceiling}"
        if negated:
            return f"(NOT ({body}))"
        return f"({body})"

    def _half_bound(self, name: str, op: str, bound: int | str, negated: bool) -> str:
        body = f"{name} {op} {bound}"
        if negated:
            return f"(NOT ({body}))"
        return f"({body})"

    def call(self, name: str, args: tuple[Expr, ...]) -> str:
        if name == "isnan":
            (operand,) = args
            target = self.value_type(operand)
            if target is not None and not pa.types.is_floating(target):
                raise FilterError(f"is_nan() takes a float column, got {target}")
            return f"(isnan({self.value(operand)}))"
        if name in ("starts_with", "ends_with", "contains"):
            operand, needle = args
            if not isinstance(needle, _Lit) or not isinstance(needle.value, str):
                raise FilterError(f"{name}() takes a string literal")
            target = self.value_type(operand)
            if target is not None and not (
                pa.types.is_string(target) or pa.types.is_large_string(target)
            ):
                raise FilterError(f"{name}() takes a string column, got {target}")
            return f"({name}({self.value(operand)}, {_quote_literal(needle.value)}))"
        raise AssertionError(f"unknown filter function {name}")  # noqa: S101

    def value(self, expr: Expr) -> str:
        if isinstance(expr, _Col):
            self.column_type(expr)
            return _quote_path(expr.path)
        if isinstance(expr, _Lit):
            return self.typed_literal(expr.value, None)
        if isinstance(expr, _Arith):
            return f"({self.value(expr.left)} {expr.op} {self.value(expr.right)})"
        raise FilterError(f"{expr!r} is not a filter value")

    def value_type(self, expr: Expr) -> pa.DataType | None:
        if isinstance(expr, _Col):
            return self.column_type(expr)
        if isinstance(expr, _Lit):
            return _literal_type(expr.value)
        return None

    def column_type(self, column: _Col) -> pa.DataType | None:
        if self._schema is None:
            return None
        try:
            field = self._schema.field(column.path[0])
        except KeyError:
            raise FilterError(f"unknown column {column.path[0]!r}") from None
        for segment in column.path[1:]:
            if not pa.types.is_struct(field.type):
                raise FilterError(
                    f"column {'.'.join(column.path)!r}:"
                    f" {field.name!r} is {field.type}, not a struct"
                )
            try:
                field = field.type.field(segment)
            except KeyError:
                raise FilterError(
                    f"unknown field {segment!r} in {field.name!r}"
                ) from None
        return field.type

    def typed_literal(self, value: object, target: pa.DataType | None) -> str:
        if target is None:
            return _generic_literal(value)
        if pa.types.is_boolean(target):
            if not isinstance(value, bool):
                raise FilterError(f"got literal {value!r} for {target}")
            return "TRUE" if value else "FALSE"
        if _is_integer_type(target):
            if isinstance(value, bool) or not isinstance(value, int):
                raise FilterError(f"got literal {value!r} for {target}")
            _check_range(value, target)
            return str(value)
        if pa.types.is_float32(target):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FilterError(f"got literal {value!r} for {target}")
            return _float_literal(value, "float")
        if pa.types.is_float64(target):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FilterError(f"got literal {value!r} for {target}")
            return _float_literal(value, "double")
        if pa.types.is_string(target) or pa.types.is_large_string(target):
            if not isinstance(value, str):
                raise FilterError(f"got literal {value!r} for {target}")
            return _quote_literal(value)
        if pa.types.is_date32(target):
            if not isinstance(value, dt.date) or isinstance(value, dt.datetime):
                raise FilterError(f"got literal {value!r} for {target}")
            return f"date '{value.isoformat()}'"
        if pa.types.is_timestamp(target):
            if not isinstance(value, dt.datetime):
                raise FilterError(f"got literal {value!r} for {target}")
            if target.tz is not None:
                raise FilterError(
                    "timezone-aware timestamp columns are not yet supported"
                )
            precision = _TIMESTAMP_PRECISION[target.unit]
            return f"timestamp({precision}) '{_format_datetime(value, target.unit)}'"
        raise FilterError(f"literals against {target} are not supported yet")


def _format_datetime(value: dt.datetime, unit: str) -> str:
    if unit == "s":
        value = value.replace(microsecond=0)
    elif unit == "ms":
        value = value.replace(microsecond=(value.microsecond // 1000) * 1000)
    text = value.isoformat(sep=" ")
    if unit == "ns":
        base, _, _ = text.partition(".")
        return f"{base}.{value.microsecond:06d}000"
    return text


def _generic_literal(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float_literal(value, "double")
    if isinstance(value, str):
        return _quote_literal(value)
    if isinstance(value, dt.datetime):
        return f"timestamp '{value.isoformat(sep=' ')}'"
    if isinstance(value, dt.date):
        return f"date '{value.isoformat()}'"
    raise FilterError(f"got literal {value!r}")


def _float_literal(value: int | float, width: str) -> str:
    if isinstance(value, float) and math.isinf(value):
        sign = "-" if value < 0 else ""
        return f"CAST('{sign}inf' AS {width})"
    if width == "float":
        return f"CAST({value!r} AS float)"
    return repr(float(value))


def _literal_type(value: object) -> pa.DataType | None:
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, str):
        return pa.string()
    return None


def _is_integer_type(target: pa.DataType) -> bool:
    return (
        pa.types.is_int8(target)
        or pa.types.is_int16(target)
        or pa.types.is_int32(target)
        or pa.types.is_int64(target)
        or pa.types.is_uint8(target)
        or pa.types.is_uint16(target)
        or pa.types.is_uint32(target)
        or pa.types.is_uint64(target)
    )


def _integer_bounds(target: pa.DataType) -> tuple[int, int]:
    bits = target.bit_width
    if (
        pa.types.is_uint8(target)
        or pa.types.is_uint16(target)
        or pa.types.is_uint32(target)
        or pa.types.is_uint64(target)
    ):
        return 0, 2**bits - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _in_range(value: int, target: pa.DataType) -> bool:
    low, high = _integer_bounds(target)
    return low <= value <= high


def _check_range(value: int, target: pa.DataType) -> None:
    if not _in_range(value, target):
        raise FilterError(f"literal {value} is out of range for {target}")


def _clamp_bound(bound: int, op: str, target: pa.DataType) -> int | str | None:
    """Fit an integer bound to `target`, or map the predicate to a tautology.

    Returns the bound, ``"all"`` when every non-null row matches, or
    ``None`` when no row matches (null included, mirroring three-valued
    filter logic).
    """
    low, high = _integer_bounds(target)
    if low <= bound <= high:
        return bound
    if op in (">", ">="):
        return None if bound >= high else "all"
    if op in ("<", "<="):
        return "all" if bound >= high else None
    if op == "=":
        return None
    return "all"


def _int_bound(
    value: int | float,
    rounding: Callable[[float], int],
    bounds: tuple[int, int],
    low_side: bool,
) -> int | str | None:
    """An integer `between` bound from an int or float literal.

    Out-of-range bounds clamp: a bound the whole column already satisfies
    is ``"all"`` (vacuous), while one no row satisfies is ``None``
    (unsatisfiable).
    """
    side = "low" if low_side else "high"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FilterError(f"got literal {value!r} for the {side} bound")
    if isinstance(value, float):
        if math.isinf(value):
            return "all" if (value < 0) == low_side else None
        value = rounding(value)
    low, high = bounds
    if low_side:
        if value <= low:
            return "all"
        if value > high:
            return None
    else:
        if value >= high:
            return "all"
        if value < low:
            return None
    return value
