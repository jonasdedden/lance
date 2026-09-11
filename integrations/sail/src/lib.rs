// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance data source for [Sail](https://github.com/lakehq/sail).
//!
//! Sail is a Rust implementation of the Spark Connect server built on
//! DataFusion. It reaches its formats through a `DataSource` trait, which this
//! crate implements for Lance: registering [`LanceDataSource`] makes
//! `spark.read.format("lance")`, `df.write.format("lance")` and
//! `CREATE TABLE ... USING lance` work in a Sail session without any JVM.
//!
//! ```text
//!   Spark client ──Spark Connect──▶ Sail ──DataSource──▶ sail-lance ──▶ lance::Dataset
//! ```
//!
//! The crate is built around three pieces:
//!
//! * [`LanceDataSource`] plans reads and writes for Sail.
//! * [`LanceTableProvider`] is the DataFusion table provider it hands back,
//!   with projection, filter, limit and vector search pushed into the Lance
//!   scanner.
//! * `bridge` moves Arrow data between the Arrow version Lance is built against
//!   and the one Sail is built against, through the Arrow C data interface.

mod bridge;
mod exec;
mod filter;
mod options;
mod provider;
mod sink;
mod source;
mod uri;
mod write;

/// Returns the message of a call that must fail.
#[cfg(test)]
pub(crate) fn error_message<T>(result: datafusion_common::Result<T>) -> String {
    match result {
        Ok(_) => "the call unexpectedly succeeded".to_string(),
        Err(error) => error.to_string(),
    }
}

pub use options::{
    DatasetRef, LanceReadOptions, LanceWriteMode, LanceWriteOptions, NearestOptions,
};
pub use provider::LanceTableProvider;
pub use source::LanceDataSource;
pub use write::{LancePhysicalPlanner, LanceWriteNode};
