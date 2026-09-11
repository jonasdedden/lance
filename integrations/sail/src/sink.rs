// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The physical write into a Lance dataset.

use std::fmt::Formatter;
use std::str::FromStr;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow_lance::array::{RecordBatch as LanceBatch, RecordBatchReader};
use arrow_lance::datatypes::SchemaRef as LanceSchemaRef;
use arrow_lance::error::ArrowError as LanceArrowError;
use async_trait::async_trait;
use datafusion::datasource::sink::DataSink;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType};
use datafusion_common::{DataFusionError, Result, plan_datafusion_err, plan_err};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance_file::version::LanceFileVersion;
use tokio::sync::mpsc::Receiver;

use crate::bridge;
use crate::exec::lance_error;
use crate::options::{LanceWriteMode, LanceWriteOptions};

/// Writes a DataFusion stream into a Lance dataset in one transaction.
#[derive(Debug)]
pub struct LanceDataSink {
    uri: String,
    mode: LanceWriteMode,
    options: LanceWriteOptions,
    schema: SchemaRef,
}

impl LanceDataSink {
    pub fn new(
        uri: String,
        mode: LanceWriteMode,
        options: LanceWriteOptions,
        schema: SchemaRef,
    ) -> Self {
        Self {
            uri,
            mode,
            options,
            schema,
        }
    }

    fn write_params(&self, mode: WriteMode) -> Result<WriteParams> {
        let defaults = WriteParams::default();
        let data_storage_version = self
            .options
            .file_format_version
            .as_deref()
            .map(|version| {
                LanceFileVersion::from_str(version).map_err(|e| {
                    plan_datafusion_err!("unsupported Lance file format version '{version}': {e}")
                })
            })
            .transpose()?;
        Ok(WriteParams {
            mode,
            max_rows_per_file: self
                .options
                .max_rows_per_file
                .unwrap_or(defaults.max_rows_per_file),
            max_rows_per_group: self
                .options
                .max_rows_per_group
                .unwrap_or(defaults.max_rows_per_group),
            max_bytes_per_file: self
                .options
                .max_bytes_per_file
                .unwrap_or(defaults.max_bytes_per_file),
            data_storage_version,
            enable_stable_row_ids: self
                .options
                .enable_stable_row_ids
                .unwrap_or(defaults.enable_stable_row_ids),
            ..defaults
        })
    }
}

impl DisplayAs for LanceDataSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "LanceDataSink: uri={}, mode={:?}", self.uri, self.mode)
    }
}

#[async_trait]
impl DataSink for LanceDataSink {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        mut data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> Result<u64> {
        let mode = resolve_write_mode(&self.uri, self.mode).await?;
        let Some(mode) = mode else {
            // `IgnoreIfExists` against an existing dataset: consume nothing and
            // leave the dataset untouched.
            return Ok(0);
        };
        let params = self.write_params(mode)?;
        let schema: LanceSchemaRef = Arc::new(bridge::schema_to_lance(self.schema.as_ref())?);

        // Lance drives its writer from a blocking `RecordBatchReader` on a
        // background thread, so batches are handed over through a channel
        // instead of being collected in memory.
        let (sender, receiver) = tokio::sync::mpsc::channel(2);
        let reader = ChannelBatchReader {
            schema: Arc::clone(&schema),
            receiver,
        };
        let uri = self.uri.clone();
        let write = tokio::spawn(async move {
            Dataset::write(reader, uri.as_str(), Some(params))
                .await
                .map_err(lance_error)
        });

        let mut rows = 0_u64;
        let mut input_error = None;
        let mut writer_stopped_early = false;
        loop {
            match data.try_next().await {
                Ok(Some(batch)) => {
                    rows += batch.num_rows() as u64;
                    // The reader side is an Arrow 58 `RecordBatchReader`, so a
                    // conversion failure has to travel as an Arrow 58 error.
                    let converted = bridge::batch_to_lance(batch, Arc::clone(&schema))
                        .map_err(|e| LanceArrowError::ExternalError(Box::new(e)));
                    if sender.send(converted).await.is_err() {
                        // The writer is gone; the error it failed with is the
                        // useful one.
                        writer_stopped_early = true;
                        break;
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    // Closing the channel here would look like the end of the
                    // input and commit the rows written so far, so the writer
                    // is failed on purpose instead.
                    let _ = sender.send(Err(input_failed(&error))).await;
                    input_error = Some(error);
                    break;
                }
            }
        }
        drop(sender);
        let written = write
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        if let Some(error) = input_error {
            return Err(error);
        }
        written?;
        if writer_stopped_early {
            return plan_err!(
                "the Lance writer for {} stopped before all rows were written",
                self.uri
            );
        }
        Ok(rows)
    }
}

/// Reports a failed input stream to the Lance writer, so that it aborts
/// instead of committing the rows it has already received.
fn input_failed(error: &DataFusionError) -> LanceArrowError {
    LanceArrowError::ExternalError(Box::new(std::io::Error::other(format!(
        "the input of the Lance write failed: {error}"
    ))))
}

/// Resolves the Lance write mode against the dataset that is already there.
///
/// Returns `None` when the write must be skipped entirely.
async fn resolve_write_mode(uri: &str, mode: LanceWriteMode) -> Result<Option<WriteMode>> {
    match mode {
        LanceWriteMode::Overwrite => Ok(Some(WriteMode::Overwrite)),
        LanceWriteMode::Append => {
            if dataset_exists(uri).await? {
                Ok(Some(WriteMode::Append))
            } else {
                Ok(Some(WriteMode::Create))
            }
        }
        LanceWriteMode::ErrorIfExists => {
            if dataset_exists(uri).await? {
                Err(plan_datafusion_err!("Lance dataset already exists: {uri}"))
            } else {
                Ok(Some(WriteMode::Create))
            }
        }
        LanceWriteMode::IgnoreIfExists => {
            if dataset_exists(uri).await? {
                Ok(None)
            } else {
                Ok(Some(WriteMode::Create))
            }
        }
    }
}

pub async fn dataset_exists(uri: &str) -> Result<bool> {
    match Dataset::open(uri).await {
        Ok(_) => Ok(true),
        Err(lance::Error::DatasetNotFound { .. }) => Ok(false),
        Err(e) => Err(lance_error(e)),
    }
}

/// Adapts the converted batches to the blocking reader interface Lance writes from.
struct ChannelBatchReader {
    schema: LanceSchemaRef,
    receiver: Receiver<std::result::Result<LanceBatch, LanceArrowError>>,
}

impl Iterator for ChannelBatchReader {
    type Item = std::result::Result<LanceBatch, LanceArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.blocking_recv()
    }
}

impl RecordBatchReader for ChannelBatchReader {
    fn schema(&self) -> LanceSchemaRef {
        Arc::clone(&self.schema)
    }
}
