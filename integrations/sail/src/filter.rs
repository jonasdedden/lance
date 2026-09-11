// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Translation of DataFusion filter expressions into Lance scan filters.
//!
//! Lance takes its filter as a SQL string that it parses and binds against the
//! dataset schema itself, so a DataFusion [`Expr`] has to be rendered back into
//! SQL. Only the expressions below are rendered; anything else stays with
//! DataFusion.
//!
//! Rendering is done here rather than with DataFusion's `Unparser` because Sail
//! builds DataFusion without the `sql` feature, and because the rendering has
//! to stay inside the subset of SQL that Lance's own parser accepts. Getting
//! that wrong cannot produce a wrong answer: the caller reports pushed down
//! filters as [`Inexact`], so DataFusion re-evaluates every one of them, and
//! offers each rendered filter to Lance's parser before using it.
//!
//! [`Inexact`]: datafusion::logical_expr::TableProviderFilterPushDown::Inexact

use datafusion_common::ScalarValue;
use datafusion_expr::{Between, BinaryExpr, Expr, Like, Operator};

/// Renders `expr` as a Lance filter string, or returns `None` when Lance cannot
/// be asked to evaluate it.
pub fn to_lance_filter(expr: &Expr) -> Option<String> {
    render(expr)
}

/// Combines rendered filters into a single Lance filter string.
pub fn combine_lance_filters(filters: &[String]) -> Option<String> {
    match filters {
        [] => None,
        [filter] => Some(filter.clone()),
        filters => Some(filters.join(" AND ")),
    }
}

fn render(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => render_identifier(column.name()),
        Expr::Literal(value, _) => render_literal(value),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            let op = render_operator(op)?;
            Some(format!("({} {} {})", render(left)?, op, render(right)?))
        }
        Expr::Not(inner) => Some(format!("(NOT {})", render(inner)?)),
        Expr::Negative(inner) => Some(format!("(- {})", render(inner)?)),
        Expr::IsNull(inner) => Some(format!("({} IS NULL)", render(inner)?)),
        Expr::IsNotNull(inner) => Some(format!("({} IS NOT NULL)", render(inner)?)),
        Expr::IsTrue(inner) => Some(format!("({} IS TRUE)", render(inner)?)),
        Expr::IsFalse(inner) => Some(format!("({} IS FALSE)", render(inner)?)),
        Expr::IsNotTrue(inner) => Some(format!("({} IS NOT TRUE)", render(inner)?)),
        Expr::IsNotFalse(inner) => Some(format!("({} IS NOT FALSE)", render(inner)?)),
        Expr::Between(Between {
            expr,
            negated,
            low,
            high,
        }) => Some(format!(
            "({} {}BETWEEN {} AND {})",
            render(expr)?,
            if *negated { "NOT " } else { "" },
            render(low)?,
            render(high)?
        )),
        Expr::InList(in_list) => {
            let items = in_list
                .list
                .iter()
                .map(render)
                .collect::<Option<Vec<_>>>()?;
            Some(format!(
                "({} {}IN ({}))",
                render(&in_list.expr)?,
                if in_list.negated { "NOT " } else { "" },
                items.join(", ")
            ))
        }
        Expr::Like(Like {
            negated,
            expr,
            pattern,
            escape_char,
            case_insensitive,
        }) => {
            if escape_char.is_some() {
                return None;
            }
            Some(format!(
                "({} {}{} {})",
                render(expr)?,
                if *negated { "NOT " } else { "" },
                if *case_insensitive { "ILIKE" } else { "LIKE" },
                render(pattern)?
            ))
        }
        _ => None,
    }
}

/// Renders a column reference, rejecting names that would need quoting rules
/// this translation does not want to depend on.
fn render_identifier(name: &str) -> Option<String> {
    let mut characters = name.chars();
    let valid = characters
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && characters.all(|c| c.is_ascii_alphanumeric() || c == '_');
    valid.then(|| name.to_string())
}

/// Renders a literal, restricted to the types whose SQL spelling is the same in
/// DataFusion and in Lance. Temporal and decimal literals are left out on
/// purpose: they need a cast whose spelling differs between SQL dialects.
fn render_literal(value: &ScalarValue) -> Option<String> {
    match value {
        ScalarValue::Null => Some("NULL".to_string()),
        ScalarValue::Boolean(Some(value)) => Some(value.to_string()),
        ScalarValue::Int8(Some(value)) => Some(value.to_string()),
        ScalarValue::Int16(Some(value)) => Some(value.to_string()),
        ScalarValue::Int32(Some(value)) => Some(value.to_string()),
        ScalarValue::Int64(Some(value)) => Some(value.to_string()),
        ScalarValue::UInt8(Some(value)) => Some(value.to_string()),
        ScalarValue::UInt16(Some(value)) => Some(value.to_string()),
        ScalarValue::UInt32(Some(value)) => Some(value.to_string()),
        ScalarValue::UInt64(Some(value)) => Some(value.to_string()),
        ScalarValue::Float32(Some(value)) => render_float(f64::from(*value)),
        ScalarValue::Float64(Some(value)) => render_float(*value),
        ScalarValue::Utf8(Some(value))
        | ScalarValue::LargeUtf8(Some(value))
        | ScalarValue::Utf8View(Some(value)) => Some(render_string(value)),
        // A typed NULL compares like an untyped one in every predicate this
        // module renders.
        ScalarValue::Boolean(None)
        | ScalarValue::Int8(None)
        | ScalarValue::Int16(None)
        | ScalarValue::Int32(None)
        | ScalarValue::Int64(None)
        | ScalarValue::UInt8(None)
        | ScalarValue::UInt16(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::Float32(None)
        | ScalarValue::Float64(None)
        | ScalarValue::Utf8(None)
        | ScalarValue::LargeUtf8(None)
        | ScalarValue::Utf8View(None) => Some("NULL".to_string()),
        _ => None,
    }
}

fn render_float(value: f64) -> Option<String> {
    value.is_finite().then(|| {
        if value.fract() == 0.0 {
            format!("{value:.1}")
        } else {
            value.to_string()
        }
    })
}

fn render_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn render_operator(op: &Operator) -> Option<&'static str> {
    match op {
        Operator::Eq => Some("="),
        Operator::NotEq => Some("!="),
        Operator::Lt => Some("<"),
        Operator::LtEq => Some("<="),
        Operator::Gt => Some(">"),
        Operator::GtEq => Some(">="),
        Operator::And => Some("AND"),
        Operator::Or => Some("OR"),
        Operator::Plus => Some("+"),
        Operator::Minus => Some("-"),
        Operator::Multiply => Some("*"),
        Operator::Divide => Some("/"),
        Operator::Modulo => Some("%"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use datafusion_expr::{col, lit};

    use super::*;

    #[test]
    fn comparisons_and_conjunctions_are_rendered() {
        let expr = col("id").gt(lit(3_i64)).and(col("name").eq(lit("lance")));
        assert_eq!(
            to_lance_filter(&expr).as_deref(),
            Some("((id > 3) AND (name = 'lance'))")
        );
    }

    #[test]
    fn null_checks_in_lists_and_patterns_are_rendered() {
        assert_eq!(
            to_lance_filter(&col("id").is_null()).as_deref(),
            Some("(id IS NULL)")
        );
        assert_eq!(
            to_lance_filter(&col("id").in_list(vec![lit(1_i64), lit(2_i64)], true)).as_deref(),
            Some("(id NOT IN (1, 2))")
        );
        assert_eq!(
            to_lance_filter(&col("name").like(lit("lan%"))).as_deref(),
            Some("(name LIKE 'lan%')")
        );
    }

    #[test]
    fn string_literals_are_escaped() {
        assert_eq!(
            to_lance_filter(&col("name").eq(lit("o'brien"))).as_deref(),
            Some("(name = 'o''brien')")
        );
    }

    #[test]
    fn float_literals_keep_their_type() {
        assert_eq!(
            to_lance_filter(&col("score").gt(lit(1.0_f64))).as_deref(),
            Some("(score > 1.0)")
        );
        assert_eq!(
            to_lance_filter(&col("score").gt(lit(f64::NAN))),
            None,
            "a literal that has no SQL spelling must not be pushed down"
        );
    }

    #[test]
    fn unsupported_expressions_stay_with_datafusion() {
        let timestamp = col("at").gt(lit(ScalarValue::TimestampMicrosecond(Some(0), None)));
        assert_eq!(to_lance_filter(&timestamp), None);
        let unknown = col("id")
            .eq(lit(1_i64))
            .or(Expr::IsUnknown(Box::new(col("flag"))));
        assert_eq!(to_lance_filter(&unknown), None);
    }

    #[test]
    fn identifiers_that_need_quoting_are_not_pushed_down() {
        let quoted = Expr::Column(datafusion_common::Column::new_unqualified("with space"));
        assert_eq!(to_lance_filter(&quoted.eq(lit(1_i64))), None);
    }

    #[test]
    fn filters_are_combined_with_and() {
        assert_eq!(combine_lance_filters(&[]), None);
        assert_eq!(
            combine_lance_filters(&["(a = 1)".to_string()]).as_deref(),
            Some("(a = 1)")
        );
        assert_eq!(
            combine_lance_filters(&["(a = 1)".to_string(), "(b = 2)".to_string()]).as_deref(),
            Some("(a = 1) AND (b = 2)")
        );
    }
}
