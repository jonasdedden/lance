// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact rewrites for numeric comparisons that keep the filter on the column.
//!
//! Two cases lose the scalar index today:
//!
//! * An integer column compared with a fractional float literal fails to plan,
//!   because the literal cannot be coerced to the column's type without losing
//!   precision. `CAST(i AS double) > 1.5` plans but casts the column and skips
//!   the index.
//! * A `Float32` column compared with a double-typed literal (for example
//!   `CAST(0.5 AS double)` or `CAST('-inf' AS double)`) casts the column to
//!   `Float64` and skips the index, while the same value written as an untyped
//!   literal is coerced to `Float32` and uses the index.
//!
//! This module rewrites both cases exactly, without lossy conversion:
//!
//! * Integer versus float: `i > 1.5` becomes `i > 1`, `i >= 1.5` becomes
//!   `i > 1`, `i < 1.5` and `i <= 1.5` become `i <= 1`, `i = 1.5` becomes
//!   false (null-preserving) and `i != 1.5` becomes true (null-preserving),
//!   with out-of-range literals clamped to constants. Integral floats in range
//!   become integer literals with the same operator. `CAST(i AS double)`
//!   unwraps to `i` first, so the same rewrite applies and the index stays.
//!
//!   The rewrite is mathematical exactness, not DataFusion's lossy
//!   int-to-double coercion. For small literals the two agree; for large
//!   integers with a close literal they can differ, and the exact form is what
//!   keeps the comparison on the column. This mirrors DataFusion's
//!   unwrap-cast-in-comparison rule for integer casts.
//!
//! * `Float64` versus `Float32`: a double-typed literal that converts to
//!   `Float32` exactly (round-trips, including infinities and NaN) becomes a
//!   `Float32` literal, so the comparison stays on the column. Anything else
//!   is left alone so the planner still casts the column and stays correct,
//!   just without the index. Bare untyped literals keep their existing
//!   always-downcast behaviour; only `CAST`-wrapped (typed) literals take the
//!   exactness check, which is what distinguishes `x > 0.5` (already indexed)
//!   from `x > CAST(0.5 AS double)` (currently not).

use arrow_schema::DataType;
use datafusion::logical_expr::{BinaryExpr, Operator};
use datafusion::prelude::Expr;
use datafusion::scalar::ScalarValue;

use crate::logical_expr::resolve_column_type;
use lance_core::datatypes::Schema;

/// True for the eight integer types.
fn is_integer_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

/// Inclusive minimum and exclusive maximum of an integer type, as `f64`.
///
/// Every bound is a power of two (or zero), so it is exactly representable.
/// An integral `f64` is in range exactly when `min <= value < max_exclusive`.
fn int_bounds(data_type: &DataType) -> Option<(f64, f64)> {
    match data_type {
        DataType::Int8 => Some((-128.0, 128.0)),
        DataType::Int16 => Some((-32768.0, 32768.0)),
        DataType::Int32 => Some((-2147483648.0, 2147483648.0)),
        DataType::Int64 => Some((-9223372036854775808.0, 9223372036854775808.0)),
        DataType::UInt8 => Some((0.0, 256.0)),
        DataType::UInt16 => Some((0.0, 65536.0)),
        DataType::UInt32 => Some((0.0, 4294967296.0)),
        DataType::UInt64 => Some((0.0, 18446744073709551616.0)),
        _ => None,
    }
}

/// A float scalar as `f64`, or `None` when it is not a float scalar.
///
/// Returns `Some(None)` for a null float literal so callers can lower it to a
/// typed null instead of erroring.
fn float_scalar_to_f64(value: &ScalarValue) -> Option<Option<f64>> {
    match value {
        ScalarValue::Float16(v) => Some(v.map(|v| v.to_f64())),
        ScalarValue::Float32(v) => Some(v.map(f64::from)),
        ScalarValue::Float64(v) => Some(*v),
        _ => None,
    }
}

/// Build an integer literal of `data_type` from an in-range integral `f64`.
///
/// The caller must have checked `min <= value < max_exclusive` with
/// [`int_bounds`]; the conversion below is then exact.
fn make_int_literal(data_type: &DataType, value: f64) -> Option<ScalarValue> {
    match data_type {
        DataType::Int8 => Some(ScalarValue::Int8(Some(value as i8))),
        DataType::Int16 => Some(ScalarValue::Int16(Some(value as i16))),
        DataType::Int32 => Some(ScalarValue::Int32(Some(value as i32))),
        DataType::Int64 => Some(ScalarValue::Int64(Some(value as i64))),
        DataType::UInt8 => Some(ScalarValue::UInt8(Some(value as u8))),
        DataType::UInt16 => Some(ScalarValue::UInt16(Some(value as u16))),
        DataType::UInt32 => Some(ScalarValue::UInt32(Some(value as u32))),
        DataType::UInt64 => Some(ScalarValue::UInt64(Some(value as u64))),
        _ => None,
    }
}

/// Whether a `f64` round-trips through `f32` exactly.
///
/// NaN converts to NaN (checked separately, since `NaN != NaN`); infinities
/// and finite values must survive the round trip bit-for-bit as values.
fn is_exact_f64_to_f32(value: f64) -> bool {
    if value.is_nan() {
        // `as f32` preserves NaN-ness; keep the sign so negative NaN stays
        // negative through the rewrite.
        let converted = value as f32;
        return converted.is_nan() && converted.is_sign_negative() == value.is_sign_negative();
    }
    (value as f32) as f64 == value
}

/// False for non-null rows, null for null rows: `col IS NULL AND NULL`.
fn false_preserving_null(col: &Expr) -> Expr {
    col.clone()
        .is_null()
        .and(Expr::Literal(ScalarValue::Boolean(None), None))
}

/// True for non-null rows, null for null rows: `col IS NOT NULL OR NULL`.
fn true_preserving_null(col: &Expr) -> Expr {
    col.clone()
        .is_not_null()
        .or(Expr::Literal(ScalarValue::Boolean(None), None))
}

/// Fold `CAST`/`TRY_CAST` around literals to a literal.
///
/// Returns the folded scalar and whether a cast was unwrapped. A bare literal
/// reports `false`; anything through a cast reports `true`, which is what lets
/// the `Float32` rewrite tell an untyped `0.5` (existing always-downcast path)
/// from a typed `CAST(0.5 AS double)` (exactness-checked path).
///
/// String literals such as `'-inf'` fold through a float cast, including the
/// spellings Arrow's own cast rejects, so filter generators can spell special
/// values as `CAST('-inf' AS double)` and still use the index.
fn extract_literal(expr: &Expr) -> Option<(ScalarValue, bool)> {
    match expr {
        Expr::Literal(value, _) => Some((value.clone(), false)),
        Expr::Cast(cast) => {
            let target = cast.field.data_type().clone();
            let (inner, _) = extract_literal(&cast.expr)?;
            fold_cast(&inner, &target, false).map(|v| (v, true))
        }
        Expr::TryCast(try_cast) => {
            let target = try_cast.field.data_type().clone();
            let (inner, _) = extract_literal(&try_cast.expr)?;
            fold_cast(&inner, &target, true).map(|v| (v, true))
        }
        Expr::Negative(inner) => {
            let (inner, was_cast) = extract_literal(inner)?;
            Some((negate_scalar(&inner)?, was_cast))
        }
        _ => None,
    }
}

fn fold_cast(inner: &ScalarValue, target: &DataType, is_try: bool) -> Option<ScalarValue> {
    // Fast path for the common numeric and string spellings.
    if let Ok(converted) = inner.cast_to(target) {
        return Some(converted);
    }
    if is_try {
        // `TRY_CAST('abc' AS double)` is null, not an error.
        if let Ok(null) = ScalarValue::try_new_null(target) {
            return Some(null);
        }
    }
    // Arrow's Utf8-to-float cast rejects some spellings filter generators
    // emit (`'-inf'`, `'Infinity'`, `'nan'` with signs and cases). Parse them
    // explicitly so those literals still fold and use the index.
    let text = match inner {
        ScalarValue::Utf8(v) | ScalarValue::LargeUtf8(v) => v.clone(),
        ScalarValue::Utf8View(v) => v.clone(),
        _ => None,
    }?;
    let parsed = parse_special_float(&text)?;
    match target {
        DataType::Float16 => {
            use half::f16;
            // Only infinities and NaN arrive here; finite magnitudes need a
            // correctly rounded conversion, which the `cast_to` above already
            // attempted.
            if parsed.is_nan() {
                Some(ScalarValue::Float16(Some(f16::NAN)))
            } else if parsed.is_infinite() {
                Some(ScalarValue::Float16(Some(if parsed.is_sign_positive() {
                    f16::INFINITY
                } else {
                    f16::NEG_INFINITY
                })))
            } else {
                None
            }
        }
        DataType::Float32 => Some(ScalarValue::Float32(Some(parsed as f32))),
        DataType::Float64 => Some(ScalarValue::Float64(Some(parsed))),
        _ => None,
    }
}

fn parse_special_float(text: &str) -> Option<f64> {
    match text.trim().to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => Some(f64::NAN),
        _ => None,
    }
}

fn negate_scalar(value: &ScalarValue) -> Option<ScalarValue> {
    match value {
        ScalarValue::Float16(v) => Some(ScalarValue::Float16(v.map(|v| -v))),
        ScalarValue::Float32(v) => Some(ScalarValue::Float32(v.map(|v| -v))),
        ScalarValue::Float64(v) => Some(ScalarValue::Float64(v.map(|v| -v))),
        ScalarValue::Int8(v) => match v {
            None => Some(ScalarValue::Int8(None)),
            Some(x) => Some(ScalarValue::Int8(Some(x.checked_neg()?))),
        },
        ScalarValue::Int16(v) => match v {
            None => Some(ScalarValue::Int16(None)),
            Some(x) => Some(ScalarValue::Int16(Some(x.checked_neg()?))),
        },
        ScalarValue::Int32(v) => match v {
            None => Some(ScalarValue::Int32(None)),
            Some(x) => Some(ScalarValue::Int32(Some(x.checked_neg()?))),
        },
        ScalarValue::Int64(v) => match v {
            None => Some(ScalarValue::Int64(None)),
            Some(x) => Some(ScalarValue::Int64(Some(x.checked_neg()?))),
        },
        _ => None,
    }
}

/// The underlying column behind `CAST(col AS float)` chains, if any.
///
/// Returns the column expression to keep in the rewritten filter (so the
/// scalar index still applies) and its type. Only float-targeted casts over
/// integer or float columns unwrap; anything else stays as written.
fn extract_column(expr: &Expr, schema: &Schema) -> Option<(Expr, DataType)> {
    match expr {
        Expr::Cast(_) | Expr::TryCast(_) => extract_column_cast(expr, schema),
        _ => {
            let data_type = resolve_column_type(expr, schema)?;
            Some((expr.clone(), data_type))
        }
    }
}

fn extract_column_cast(expr: &Expr, schema: &Schema) -> Option<(Expr, DataType)> {
    let (inner, target) = match expr {
        Expr::Cast(cast) => (cast.expr.as_ref(), cast.field.data_type().clone()),
        Expr::TryCast(try_cast) => (try_cast.expr.as_ref(), try_cast.field.data_type().clone()),
        _ => return None,
    };
    if !matches!(
        target,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    ) {
        return None;
    }
    let (col_expr, col_type) = extract_column(inner, schema)?;
    if !(is_integer_type(&col_type)
        || matches!(
            col_type,
            DataType::Float16 | DataType::Float32 | DataType::Float64
        ))
    {
        return None;
    }
    Some((col_expr, col_type))
}

/// Rewrite `col op float_literal` where `col` is an integer column.
///
/// `op` must already be oriented with the column on the left (callers swap a
/// literal-on-the-left comparison first). Handles `Eq`, `NotEq`, `Lt`,
/// `LtEq`, `Gt` and `GtEq`; anything else returns `None`.
fn rewrite_int_float(
    col: &Expr,
    col_type: &DataType,
    op: Operator,
    float: Option<f64>,
) -> Option<Expr> {
    let (min, max_exclusive) = int_bounds(col_type)?;
    let binary = |op: Operator, scalar: ScalarValue| {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(col.clone()),
            op,
            right: Box::new(Expr::Literal(scalar, None)),
        })
    };

    let Some(value) = float else {
        // `i op NULL` is null for every row. Lower to a typed null so the
        // filter plans instead of erroring on the coercion.
        let null = ScalarValue::try_new_null(col_type).ok()?;
        return Some(binary(op, null));
    };

    if value.is_nan() {
        return Some(match op {
            Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                false_preserving_null(col)
            }
            Operator::NotEq => true_preserving_null(col),
            _ => return None,
        });
    }

    if value.is_infinite() {
        let positive = value.is_sign_positive();
        return Some(match op {
            Operator::Eq => false_preserving_null(col),
            Operator::NotEq => true_preserving_null(col),
            Operator::Lt => {
                if positive {
                    true_preserving_null(col)
                } else {
                    false_preserving_null(col)
                }
            }
            Operator::LtEq => {
                if positive {
                    true_preserving_null(col)
                } else {
                    false_preserving_null(col)
                }
            }
            Operator::Gt => {
                if positive {
                    false_preserving_null(col)
                } else {
                    true_preserving_null(col)
                }
            }
            Operator::GtEq => {
                if positive {
                    false_preserving_null(col)
                } else {
                    true_preserving_null(col)
                }
            }
            _ => return None,
        });
    }

    if value.fract() == 0.0 {
        if min <= value && value < max_exclusive {
            let literal = make_int_literal(col_type, value)?;
            return Some(binary(op, literal));
        }
        // Out of range: the column can never equal the literal, and ordering
        // is decided by which side of the range the literal sits on.
        let above = value >= max_exclusive;
        return Some(match op {
            Operator::Eq => false_preserving_null(col),
            Operator::NotEq => true_preserving_null(col),
            Operator::Lt | Operator::LtEq => {
                if above {
                    true_preserving_null(col)
                } else {
                    false_preserving_null(col)
                }
            }
            Operator::Gt | Operator::GtEq => {
                if above {
                    false_preserving_null(col)
                } else {
                    true_preserving_null(col)
                }
            }
            _ => return None,
        });
    }

    // Fractional value: equality never holds; ordering goes through `floor`.
    match op {
        Operator::Eq => Some(false_preserving_null(col)),
        Operator::NotEq => Some(true_preserving_null(col)),
        Operator::Gt | Operator::GtEq | Operator::Lt | Operator::LtEq => {
            let floor = value.floor();
            if floor < min {
                // Every column value exceeds the bound.
                return Some(match op {
                    Operator::Gt | Operator::GtEq => true_preserving_null(col),
                    _ => false_preserving_null(col),
                });
            }
            if floor >= max_exclusive {
                return Some(match op {
                    Operator::Gt | Operator::GtEq => false_preserving_null(col),
                    _ => true_preserving_null(col),
                });
            }
            let literal = make_int_literal(col_type, floor)?;
            match op {
                Operator::Gt | Operator::GtEq => Some(binary(Operator::Gt, literal)),
                _ => Some(binary(Operator::LtEq, literal)),
            }
        }
        _ => None,
    }
}

/// Rewrite `col op double_literal` where `col` is `Float32`.
///
/// Only `CAST`-wrapped literals are considered here; bare literals keep the
/// existing always-downcast path. Returns `Some` when the double converts
/// exactly (or is null, lowered to a typed null), and `None` otherwise so the
/// planner still casts the column and stays correct without the index.
fn rewrite_float32_double(
    col: &Expr,
    op: Operator,
    literal: &ScalarValue,
    was_cast: bool,
) -> Option<Expr> {
    if !was_cast {
        return None;
    }
    let ScalarValue::Float64(value) = literal else {
        return None;
    };
    let Some(value) = value else {
        return Some(Expr::BinaryExpr(BinaryExpr {
            left: Box::new(col.clone()),
            op,
            right: Box::new(Expr::Literal(ScalarValue::Float32(None), None)),
        }));
    };
    if !is_exact_f64_to_f32(*value) {
        return None;
    }
    Some(Expr::BinaryExpr(BinaryExpr {
        left: Box::new(col.clone()),
        op,
        right: Box::new(Expr::Literal(
            ScalarValue::Float32(Some(*value as f32)),
            None,
        )),
    }))
}

/// Try to rewrite one comparison with the column on the left.
///
/// Returns `Some` when this module owns the shape (integer versus float, or an
/// exactly convertible typed double against `Float32`), and `None` when the
/// existing coercion path should run.
fn rewrite_comparison(
    col_expr: &Expr,
    col_type: &DataType,
    op: Operator,
    lit_scalar: &ScalarValue,
    was_cast: bool,
) -> Option<Expr> {
    if is_integer_type(col_type) {
        let float = float_scalar_to_f64(lit_scalar)?;
        return rewrite_int_float(col_expr, col_type, op, float);
    }
    if matches!(col_type, DataType::Float32)
        && let Some(rewritten) = rewrite_float32_double(col_expr, op, lit_scalar, was_cast)
    {
        return Some(rewritten);
    }
    None
}

/// Whether `op` is one of the six comparisons this module rewrites.
fn is_rewritable(op: Operator) -> bool {
    matches!(
        op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    )
}

/// Rewrite a `BinaryExpr` comparison, handling a column on either side.
///
/// Unwraps `CAST(col AS float)` on the column side and `CAST(literal AS ...)`
/// on the literal side before dispatching. `AND`/`OR` are left for the caller.
pub fn rewrite_binary(left: &Expr, op: Operator, right: &Expr, schema: &Schema) -> Option<Expr> {
    if !is_rewritable(op) {
        return None;
    }
    if let Some((col_expr, col_type)) = extract_column(left, schema)
        && let Some((lit_scalar, was_cast)) = extract_literal(right)
        && let Some(rewritten) = rewrite_comparison(&col_expr, &col_type, op, &lit_scalar, was_cast)
    {
        return Some(rewritten);
    }
    if let Some((col_expr, col_type)) = extract_column(right, schema)
        && let Some((lit_scalar, was_cast)) = extract_literal(left)
    {
        let swapped = op.swap()?;
        if let Some(rewritten) =
            rewrite_comparison(&col_expr, &col_type, swapped, &lit_scalar, was_cast)
        {
            return Some(rewritten);
        }
    }
    None
}

/// Ceiling of `value` as an in-range integer literal, or the out-of-range side.
///
/// Returns `Ok` with the literal for the `>=` bound, and `Err(true)` when the
/// bound is always satisfied (`x >= very_negative`) or `Err(false)` when it
/// never is (`x >= very_positive`).
fn ceil_for_lower(col_type: &DataType, value: f64) -> Result<ScalarValue, bool> {
    let (min, max_exclusive) = int_bounds(col_type).ok_or(false)?;
    if value.is_nan() {
        return Err(false);
    }
    if value.is_infinite() {
        return Err(!value.is_sign_positive());
    }
    let ceil = if value.fract() == 0.0 {
        value
    } else {
        value.ceil()
    };
    if ceil < min {
        return Err(true);
    }
    if ceil >= max_exclusive {
        return Err(false);
    }
    make_int_literal(col_type, ceil).ok_or(false)
}

/// Floor of `value` as an in-range integer literal, or the out-of-range side.
///
/// Returns `Ok` with the literal for the `<=` bound, and `Err(true)` when the
/// bound is always satisfied (`x <= very_positive`) or `Err(false)` when it
/// never is (`x <= very_negative`).
fn floor_for_upper(col_type: &DataType, value: f64) -> Result<ScalarValue, bool> {
    let (min, max_exclusive) = int_bounds(col_type).ok_or(false)?;
    if value.is_nan() {
        return Err(false);
    }
    if value.is_infinite() {
        return Err(value.is_sign_positive());
    }
    let floor = value.floor();
    if floor < min {
        return Err(false);
    }
    if floor >= max_exclusive {
        return Err(true);
    }
    make_int_literal(col_type, floor).ok_or(false)
}

/// Rewrite `col BETWEEN low AND high` (or `NOT BETWEEN`) over an integer column
/// with float bounds.
///
/// Fractional bounds go through `ceil` (low) and `floor` (high); out-of-range
/// bounds clamp to constants or to a single-sided comparison. Returns `None`
/// when the bounds are not float literals this module owns.
pub fn rewrite_between(
    col_expr: &Expr,
    col_type: &DataType,
    low: &Expr,
    high: &Expr,
    negated: bool,
) -> Option<Expr> {
    if !is_integer_type(col_type) {
        // Float32 with exactly convertible typed double bounds: convert both
        // or leave the whole predicate alone so it stays correct.
        if matches!(col_type, DataType::Float32) {
            let (low_scalar, low_cast) = extract_literal(low)?;
            let (high_scalar, high_cast) = extract_literal(high)?;
            let low_ok = match &low_scalar {
                ScalarValue::Float64(v) => match v {
                    None => true,
                    Some(v) => !low_cast || is_exact_f64_to_f32(*v),
                },
                _ => return None,
            };
            let high_ok = match &high_scalar {
                ScalarValue::Float64(v) => match v {
                    None => true,
                    Some(v) => !high_cast || is_exact_f64_to_f32(*v),
                },
                _ => return None,
            };
            if !low_ok || !high_ok {
                return None;
            }
            // Bare literals keep the existing path; only typed literals need a
            // rewrite here, and only when exact.
            if !low_cast && !high_cast {
                return None;
            }
            let convert = |s: &ScalarValue| match s {
                ScalarValue::Float64(None) => ScalarValue::Float32(None),
                ScalarValue::Float64(Some(v)) => ScalarValue::Float32(Some(*v as f32)),
                _ => s.clone(),
            };
            return Some(Expr::Between(datafusion::logical_expr::Between {
                expr: Box::new(col_expr.clone()),
                negated,
                low: Box::new(Expr::Literal(convert(&low_scalar), None)),
                high: Box::new(Expr::Literal(convert(&high_scalar), None)),
            }));
        }
        return None;
    }
    let (low_scalar, _) = extract_literal(low)?;
    let (high_scalar, _) = extract_literal(high)?;
    let low_float = float_scalar_to_f64(&low_scalar)?;
    let high_float = float_scalar_to_f64(&high_scalar)?;

    // A null bound stays a typed null so the predicate still plans (it
    // evaluates to null/false, never true). Convert the non-null bound
    // normally; if it is always-false the whole predicate is false/null, and
    // if it is always-true the predicate is driven by the null bound alone
    // (still null/false, so keep the shape with typed nulls).
    if low_float.is_none() || high_float.is_none() {
        let null = ScalarValue::try_new_null(col_type).ok()?;
        let low_expr = match low_float {
            None => Expr::Literal(null.clone(), None),
            Some(v) => match ceil_for_lower(col_type, v) {
                Ok(lit) => Expr::Literal(lit, None),
                Err(_) => {
                    // An always-false low makes `BETWEEN` false/null; an
                    // always-true low leaves `null AND ...`, which is still
                    // null/false. Either way the typed-null shape below plans
                    // correctly, so fall through.
                    Expr::Literal(null.clone(), None)
                }
            },
        };
        let high_expr = match high_float {
            None => Expr::Literal(null, None),
            Some(v) => match floor_for_upper(col_type, v) {
                Ok(lit) => Expr::Literal(lit, None),
                Err(_) => Expr::Literal(null, None),
            },
        };
        return Some(Expr::Between(datafusion::logical_expr::Between {
            expr: Box::new(col_expr.clone()),
            negated,
            low: Box::new(low_expr),
            high: Box::new(high_expr),
        }));
    }

    let low_value = low_float.unwrap();
    let high_value = high_float.unwrap();
    let low = ceil_for_lower(col_type, low_value);
    let high = floor_for_upper(col_type, high_value);

    let binary = |op: Operator, scalar: ScalarValue| {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(col_expr.clone()),
            op,
            right: Box::new(Expr::Literal(scalar, None)),
        })
    };

    match (low, high) {
        (Err(false), _) | (_, Err(false)) => {
            // The non-negated predicate never holds.
            Some(if negated {
                true_preserving_null(col_expr)
            } else {
                false_preserving_null(col_expr)
            })
        }
        (Err(true), Err(true)) => Some(if negated {
            false_preserving_null(col_expr)
        } else {
            true_preserving_null(col_expr)
        }),
        (Err(true), Ok(high_lit)) => {
            // `low` always holds: `BETWEEN` is `x <= high`, `NOT BETWEEN` is
            // `x > high`.
            Some(if negated {
                binary(Operator::Gt, high_lit)
            } else {
                binary(Operator::LtEq, high_lit)
            })
        }
        (Ok(low_lit), Err(true)) => Some(if negated {
            binary(Operator::Lt, low_lit)
        } else {
            binary(Operator::GtEq, low_lit)
        }),
        (Ok(low_lit), Ok(high_lit)) => {
            // Empty range when the integer low exceeds the integer high.
            let empty = {
                let low_f = match &low_lit {
                    ScalarValue::Int8(v) => v.map(|v| v as f64),
                    ScalarValue::Int16(v) => v.map(|v| v as f64),
                    ScalarValue::Int32(v) => v.map(|v| v as f64),
                    ScalarValue::Int64(v) => v.map(|v| v as f64),
                    ScalarValue::UInt8(v) => v.map(|v| v as f64),
                    ScalarValue::UInt16(v) => v.map(|v| v as f64),
                    ScalarValue::UInt32(v) => v.map(|v| v as f64),
                    ScalarValue::UInt64(v) => v.map(|v| v as f64),
                    _ => None,
                };
                let high_f = match &high_lit {
                    ScalarValue::Int8(v) => v.map(|v| v as f64),
                    ScalarValue::Int16(v) => v.map(|v| v as f64),
                    ScalarValue::Int32(v) => v.map(|v| v as f64),
                    ScalarValue::Int64(v) => v.map(|v| v as f64),
                    ScalarValue::UInt8(v) => v.map(|v| v as f64),
                    ScalarValue::UInt16(v) => v.map(|v| v as f64),
                    ScalarValue::UInt32(v) => v.map(|v| v as f64),
                    ScalarValue::UInt64(v) => v.map(|v| v as f64),
                    _ => None,
                };
                match (low_f, high_f) {
                    (Some(l), Some(h)) => l > h,
                    _ => false,
                }
            };
            if empty {
                return Some(if negated {
                    true_preserving_null(col_expr)
                } else {
                    false_preserving_null(col_expr)
                });
            }
            Some(Expr::Between(datafusion::logical_expr::Between {
                expr: Box::new(col_expr.clone()),
                negated,
                low: Box::new(Expr::Literal(low_lit, None)),
                high: Box::new(Expr::Literal(high_lit, None)),
            }))
        }
    }
}

/// Rewrite `col IN (floats...)` over an integer column.
///
/// Fractional and out-of-range elements never match and are dropped; integral
/// in-range elements become integer literals. Null elements are kept as typed
/// nulls. An empty survivors list becomes a null-preserving constant.
pub fn rewrite_in_list(
    col_expr: &Expr,
    col_type: &DataType,
    list: &[Expr],
    negated: bool,
) -> Option<Expr> {
    if !is_integer_type(col_type) {
        if matches!(col_type, DataType::Float32) {
            // All typed doubles must be exactly convertible, or the whole
            // predicate stays as written (correct, without the index).
            let mut converted = Vec::with_capacity(list.len());
            let mut needs_rewrite = false;
            for item in list {
                let (scalar, was_cast) = extract_literal(item)?;
                match &scalar {
                    ScalarValue::Float64(None) => {
                        converted.push(Expr::Literal(ScalarValue::Float32(None), None));
                        if was_cast {
                            needs_rewrite = true;
                        }
                    }
                    ScalarValue::Float64(Some(v)) if was_cast => {
                        if !is_exact_f64_to_f32(*v) {
                            return None;
                        }
                        needs_rewrite = true;
                        converted.push(Expr::Literal(ScalarValue::Float32(Some(*v as f32)), None));
                    }
                    _ => return None,
                }
            }
            if !needs_rewrite {
                return None;
            }
            return Some(Expr::in_list(col_expr.clone(), converted, negated));
        }
        return None;
    }
    let (min, max_exclusive) = int_bounds(col_type)?;
    let mut survivors: Vec<Expr> = Vec::with_capacity(list.len());
    for item in list {
        let (scalar, _) = extract_literal(item)?;
        {
            let float = float_scalar_to_f64(&scalar)?;
            match float {
                None => {
                    let null = ScalarValue::try_new_null(col_type).ok()?;
                    survivors.push(Expr::Literal(null, None));
                }
                Some(v) => {
                    if v.is_nan() || v.is_infinite() || v.fract() != 0.0 {
                        continue;
                    }
                    if min <= v && v < max_exclusive {
                        let lit = make_int_literal(col_type, v)?;
                        survivors.push(Expr::Literal(lit, None));
                    }
                }
            }
        }
    }
    if survivors.is_empty() {
        return Some(if negated {
            true_preserving_null(col_expr)
        } else {
            false_preserving_null(col_expr)
        });
    }
    Some(Expr::in_list(col_expr.clone(), survivors, negated))
}

/// Try to rewrite a `BETWEEN` predicate, unwrapping `CAST(col AS float)`.
///
/// Returns `Some` when the column is an integer with float bounds this module
/// owns, or a `Float32` column with exactly convertible typed double bounds.
pub fn try_rewrite_between_expr(
    inner: &Expr,
    low: &Expr,
    high: &Expr,
    negated: bool,
    schema: &Schema,
) -> Option<Expr> {
    let (col_expr, col_type) = extract_column(inner, schema)?;
    rewrite_between(&col_expr, &col_type, low, high, negated)
}

/// Try to rewrite an `IN` list, unwrapping `CAST(col AS float)`.
///
/// Returns `Some` when the column is an integer with float elements this
/// module owns, or a `Float32` column with exactly convertible typed doubles.
pub fn try_rewrite_in_list_expr(
    col: &Expr,
    list: &[Expr],
    negated: bool,
    schema: &Schema,
) -> Option<Expr> {
    let (col_expr, col_type) = extract_column(col, schema)?;
    rewrite_in_list(&col_expr, &col_type, list, negated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::{col, lit};

    fn int_col(op: Operator, value: f64) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(col("i")),
            op,
            right: Box::new(lit(value)),
        })
    }

    #[test]
    fn fractional_ordering_uses_floor() {
        let int32 = DataType::Int32;
        let col_expr = col("i");
        assert_eq!(
            rewrite_int_float(&col_expr, &int32, Operator::Gt, Some(1.5)),
            Some(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int32(Some(1)), None)),
            }))
        );
        assert_eq!(
            rewrite_int_float(&col_expr, &int32, Operator::GtEq, Some(1.5)),
            Some(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int32(Some(1)), None)),
            }))
        );
        assert_eq!(
            rewrite_int_float(&col_expr, &int32, Operator::Lt, Some(1.5)),
            Some(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::LtEq,
                right: Box::new(Expr::Literal(ScalarValue::Int32(Some(1)), None)),
            }))
        );
        // `int_col` helper pins the shape above; a negative fractional rounds
        // toward negative infinity, not toward zero.
        assert_eq!(
            rewrite_int_float(&col_expr, &int32, Operator::Gt, Some(-1.5)),
            Some(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int32(Some(-2)), None)),
            }))
        );
        let _ = int_col(Operator::Gt, 1.5);
    }

    #[test]
    fn integral_float_becomes_integer() {
        let int32 = DataType::Int32;
        let col_expr = col("i");
        assert_eq!(
            rewrite_int_float(&col_expr, &int32, Operator::Gt, Some(2.0)),
            Some(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int32(Some(2)), None)),
            }))
        );
    }

    #[test]
    fn out_of_range_clamps_to_constants() {
        let int8 = DataType::Int8;
        let col_expr = col("i");
        // `i > 1000.5` never holds; `i <= 1000.5` always holds (nulls aside).
        assert_eq!(
            rewrite_int_float(&col_expr, &int8, Operator::Gt, Some(1000.5)),
            Some(false_preserving_null(&col_expr))
        );
        assert_eq!(
            rewrite_int_float(&col_expr, &int8, Operator::LtEq, Some(1000.5)),
            Some(true_preserving_null(&col_expr))
        );
        assert_eq!(
            rewrite_int_float(&col_expr, &int8, Operator::Eq, Some(1.5)),
            Some(false_preserving_null(&col_expr))
        );
        assert_eq!(
            rewrite_int_float(&col_expr, &int8, Operator::NotEq, Some(1.5)),
            Some(true_preserving_null(&col_expr))
        );
    }

    #[test]
    fn exact_double_to_float32() {
        assert!(is_exact_f64_to_f32(0.5));
        assert!(is_exact_f64_to_f32(f64::INFINITY));
        assert!(is_exact_f64_to_f32(f64::NEG_INFINITY));
        assert!(!is_exact_f64_to_f32(0.1));
        assert!(!is_exact_f64_to_f32(1e40));
    }

    #[test]
    fn special_strings_fold() {
        assert_eq!(parse_special_float("-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(parse_special_float("Infinity"), Some(f64::INFINITY));
        assert!(parse_special_float("nan").unwrap().is_nan());
        assert_eq!(parse_special_float("0.5"), None);
    }

    use std::sync::Arc;

    use crate::planner::Planner;
    use arrow_array::{ArrayRef, BooleanArray, Float32Array, Int64Array, RecordBatch};
    use arrow_schema::{Field, Schema as ArrowSchema};

    fn test_planner() -> Planner {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("x", DataType::Float32, true),
        ]));
        Planner::new(schema)
    }

    fn eval_filter(planner: &Planner, filter: &str, batch: &RecordBatch) -> Vec<Option<bool>> {
        let expr = planner.parse_filter(filter).unwrap();
        let expr = planner.optimize_expr(expr).unwrap();
        let physical = planner.create_physical_expr(&expr).unwrap();
        let result = physical.evaluate(batch).unwrap().into_array(0).unwrap();
        result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn int_float_plans_to_int_and_evaluates() {
        let planner = test_planner();
        // `i > 1.5` must plan (previously errored) as `i > 1`.
        let expr = planner.parse_filter("i > 1.5").unwrap();
        assert_eq!(
            expr,
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int64(Some(1)), None)),
            })
        );
        let expr = planner.parse_filter("i >= 1.5").unwrap();
        assert_eq!(
            expr,
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int64(Some(1)), None)),
            })
        );
        let expr = planner.parse_filter("i < 1.5").unwrap();
        assert_eq!(
            expr,
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::LtEq,
                right: Box::new(Expr::Literal(ScalarValue::Int64(Some(1)), None)),
            })
        );
        // Mirrored literal on the left swaps the operator.
        let expr = planner.parse_filter("1.5 < i").unwrap();
        assert_eq!(
            expr,
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int64(Some(1)), None)),
            })
        );
        // `CAST(i AS double)` unwraps to the column.
        let expr = planner.parse_filter("CAST(i AS double) > 1.5").unwrap();
        assert_eq!(
            expr,
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("i")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int64(Some(1)), None)),
            })
        );

        let batch = RecordBatch::try_new(
            planner_schema(),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2), None])) as ArrayRef,
                Arc::new(Float32Array::from(vec![Some(0.5), Some(0.5), None])) as ArrayRef,
            ],
        )
        .unwrap();
        // `i > 1.5` keeps 2, drops 1, filters null (null, not false).
        assert_eq!(
            eval_filter(&planner, "i > 1.5", &batch),
            vec![Some(false), Some(true), None]
        );
        // `i = 1.5` is false for non-null, null for null (so NOT preserves null).
        assert_eq!(
            eval_filter(&planner, "i = 1.5", &batch),
            vec![Some(false), Some(false), None]
        );
        assert_eq!(
            eval_filter(&planner, "i != 1.5", &batch),
            vec![Some(true), Some(true), None]
        );
        assert_eq!(
            eval_filter(&planner, "NOT (i = 1.5)", &batch),
            vec![Some(true), Some(true), None]
        );
    }

    fn planner_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("x", DataType::Float32, true),
        ]))
    }

    #[test]
    fn float32_exact_double_uses_float32() {
        let planner = test_planner();
        let expr = planner.parse_filter("x > CAST(0.5 AS double)").unwrap();
        assert_eq!(
            expr,
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("x")),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Float32(Some(0.5)), None)),
            })
        );
        let expr = planner.parse_filter("x < CAST('-inf' AS double)").unwrap();
        assert_eq!(
            expr,
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col("x")),
                op: Operator::Lt,
                right: Box::new(Expr::Literal(
                    ScalarValue::Float32(Some(f32::NEG_INFINITY)),
                    None
                )),
            })
        );
        // Inexact doubles stay on the double path (correct, without the index).
        let expr = planner.parse_filter("x > CAST(0.1 AS double)").unwrap();
        let optimized = planner.optimize_expr(expr).unwrap();
        assert!(
            format!("{optimized:?}").contains("Float64"),
            "inexact double must not downcast, got: {optimized:?}"
        );
    }
}
