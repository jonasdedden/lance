// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The physical scan over a Lance dataset.

use std::fmt::Formatter;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{DataFusionError, Result, internal_err, plan_datafusion_err};
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance_table::format::Fragment;

use crate::bridge;
use crate::options::LanceReadOptions;

/// Everything a [`LanceScanExec`] partition needs to build its Lance scanner.
#[derive(Debug, Clone)]
pub struct LanceScanConfig {
    /// Columns to read, in the order the scan emits them. Lance metadata
    /// columns such as `_rowid` are not listed here: they are produced by the
    /// scan options instead.
    pub columns: Vec<String>,
    /// Filter to evaluate inside the scan, as a Lance filter expression.
    pub filter: Option<String>,
    /// Row limit hint. Applied per partition, which is sound because
    /// DataFusion keeps its own limit above the scan.
    pub limit: Option<usize>,
    pub options: LanceReadOptions,
}

#[derive(Debug)]
pub struct LanceScanExec {
    dataset: Arc<Dataset>,
    config: Arc<LanceScanConfig>,
    /// Fragments assigned to each output partition. An empty assignment scans
    /// the whole dataset in one partition.
    partitions: Arc<Vec<Vec<Fragment>>>,
    schema: SchemaRef,
    metrics: ExecutionPlanMetricsSet,
    properties: Arc<PlanProperties>,
}

impl LanceScanExec {
    pub fn new(
        dataset: Arc<Dataset>,
        config: LanceScanConfig,
        partitions: Vec<Vec<Fragment>>,
        schema: SchemaRef,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(partitions.len()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            config: Arc::new(config),
            partitions: Arc::new(partitions),
            schema,
            metrics: ExecutionPlanMetricsSet::new(),
            properties,
        }
    }
}

impl DisplayAs for LanceScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "LanceScanExec: uri={}, version={}, columns=[{}]",
            self.dataset.uri(),
            self.dataset.version().version,
            self.config.columns.join(", ")
        )?;
        if let Some(filter) = &self.config.filter {
            write!(f, ", filter={filter}")?;
        }
        if let Some(limit) = self.config.limit {
            write!(f, ", limit={limit}")?;
        }
        if let Some(nearest) = &self.config.options.nearest {
            write!(
                f,
                ", nearest={{column: {}, k: {}}}",
                nearest.column, nearest.k
            )?;
        }
        Ok(())
    }
}

impl ExecutionPlan for LanceScanExec {
    fn name(&self) -> &str {
        "LanceScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            internal_err!("LanceScanExec does not have children")
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let Some(fragments) = self.partitions.get(partition) else {
            return internal_err!(
                "partition {partition} is out of range for a Lance scan with {} partitions",
                self.partitions.len()
            );
        };
        let output_rows = MetricBuilder::new(&self.metrics).output_rows(partition);
        let stream = futures::stream::once(open_scan(
            Arc::clone(&self.dataset),
            Arc::clone(&self.config),
            fragments.clone(),
            Arc::clone(&self.schema),
            output_rows,
        ))
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

async fn open_scan(
    dataset: Arc<Dataset>,
    config: Arc<LanceScanConfig>,
    fragments: Vec<Fragment>,
    output_schema: SchemaRef,
    output_rows: Count,
) -> Result<impl futures::Stream<Item = Result<RecordBatch>> + Send> {
    let mut scanner = dataset.scan();
    if !fragments.is_empty() {
        scanner.with_fragments(fragments);
    }
    if !config.columns.is_empty() {
        scanner.project(&config.columns).map_err(lance_error)?;
    }
    if let Some(filter) = &config.filter {
        scanner.filter(filter).map_err(lance_error)?;
    }
    if let Some(limit) = config.limit {
        let limit = i64::try_from(limit)
            .map_err(|_| plan_datafusion_err!("limit is too large for a Lance scan: {limit}"))?;
        scanner.limit(Some(limit), None).map_err(lance_error)?;
    }
    if let Some(batch_size) = config.options.batch_size {
        scanner.batch_size(batch_size);
    }
    if config.options.with_row_id {
        scanner.with_row_id();
    }
    if let Some(nearest) = &config.options.nearest {
        let query = arrow_lance::array::Float32Array::from(nearest.query.clone());
        scanner
            .nearest(&nearest.column, &query, nearest.k)
            .map_err(lance_error)?;
        if let Some(nprobes) = nearest.nprobes {
            scanner.nprobes(nprobes);
        }
        if let Some(refine_factor) = nearest.refine_factor {
            scanner.refine(refine_factor);
        }
        scanner.use_index(nearest.use_index);
    }

    let stream = scanner.try_into_stream().await.map_err(lance_error)?;
    // The schema of a Lance scan does not change between batches, so the
    // conversion of the schema itself is done once and reused.
    let mut converted_schema: Option<SchemaRef> = None;
    Ok(stream.map_err(lance_error).map(move |batch| {
        let batch = batch?;
        let schema = match &converted_schema {
            Some(schema) => Arc::clone(schema),
            None => {
                let schema: SchemaRef = Arc::new(bridge::schema_to_sail(batch.schema_ref())?);
                converted_schema = Some(Arc::clone(&schema));
                schema
            }
        };
        let batch = bridge::batch_to_sail(batch, schema)?;
        let batch = select_output_columns(batch, &output_schema)?;
        output_rows.add(batch.num_rows());
        Ok(batch)
    }))
}

/// Reorders a scanned batch into the column order DataFusion asked for.
///
/// Lance emits projected columns followed by the metadata columns the scan
/// options add (`_rowid`, `_distance`), which is not necessarily the order of
/// the projection DataFusion pushed down.
fn select_output_columns(batch: RecordBatch, output_schema: &SchemaRef) -> Result<RecordBatch> {
    if batch.schema_ref() == output_schema {
        return Ok(batch);
    }
    let columns = output_schema
        .fields()
        .iter()
        .map(|field| {
            batch.column_by_name(field.name()).cloned().ok_or_else(|| {
                plan_datafusion_err!(
                    "Lance scan did not return the requested column '{}'; it returned [{}]",
                    field.name(),
                    field_names(batch.schema_ref()).join(", ")
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let options = arrow::array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    RecordBatch::try_new_with_options(Arc::clone(output_schema), columns, &options)
        .map_err(DataFusionError::from)
}

fn field_names(schema: &Schema) -> Vec<&str> {
    schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect()
}

pub fn lance_error(error: lance::Error) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}
