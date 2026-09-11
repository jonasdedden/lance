# Lance for Sail

A proof of concept that makes [Sail](https://github.com/lakehq/sail), the
Rust implementation of the Spark Connect server, read and write Lance datasets.
Everything here is Rust: no JVM, no Spark, no JNI.

```python
# against a Sail server with this crate registered
spark.read.format("lance").load("s3://bucket/embeddings.lance")
df.write.format("lance").mode("overwrite").save("/data/embeddings.lance")
```

The existing [Apache Spark connector for Lance](https://github.com/lance-format/lance-spark)
gives Spark the same capability through the JVM: Spark DSv2 in Java, calling
Lance through JNI. Sail replaces the Spark JVM engine with DataFusion, so the
connector cannot be reused, but it also does not need to be: Sail reaches its
formats through a Rust trait, and Lance is a Rust library.

## What this crate is

`sail-lance` implements Sail's `DataSource` trait for Lance and provides the
physical planner for its write node. Registering it with a Sail session adds a
`lance` format to `spark.read.format(...)`, `df.write.format(...)` and
`CREATE TABLE ... USING lance`.

| Capability | Status |
| --- | --- |
| Read a dataset, schema from the dataset itself | yes |
| Column projection pushdown | yes |
| Filter pushdown (comparisons, `AND`/`OR`/`NOT`, `IS NULL`, `IN`, `BETWEEN`, `LIKE`) | yes, re-checked by DataFusion |
| Limit pushdown | yes |
| Parallel scans, one partition per fragment group | yes |
| Vector search (`nearest`) pushed into the scan | yes |
| Time travel by version or tag | yes |
| Write: append, overwrite, error-if-exists, ignore-if-exists | yes |
| Row level `MERGE`/`UPDATE`/`DELETE`, catalogs, index creation | not yet, see [Next steps](#next-steps) |

## The problem: two Arrow versions in one process

Sail and Lance are both built on Arrow and DataFusion, but not on the same
releases:

| | Sail 0.7.1 | Lance 11.0.0-beta.5 |
| --- | --- | --- |
| DataFusion | 55.0 | 54.1 |
| arrow-rs | 59.2 | 58.3 |
| object_store | 0.13.2 | 0.13.2 |

Cargo links both, but `arrow_array::RecordBatch` from Arrow 58 and from Arrow 59
are unrelated types, so a `lance::Dataset` cannot hand its batches to
DataFusion 55 directly, and Lance's own DataFusion 54 `TableProvider` is not a
DataFusion 55 `TableProvider`. Three ways out:

1. **Align the versions.** The cleanest end state and the one to aim for, but it
   is a change to one of the two projects, not something an integration can do.
2. **Serialize between them,** through Arrow IPC. Simple, and a copy of every
   byte on every batch.
3. **Hand data over through the Arrow C data interface,** which every Arrow
   version implements identically and which is designed for exactly this.

This crate does (3). `src/bridge.rs` exports a batch from one Arrow version,
moves the two `#[repr(C)]` C data interface structs field by field into the
other version's declaration of them, and imports it there. No buffer is copied:
the importing side ends up owning the exporting side's release callback and
calls it when the data is dropped. Field and schema metadata, which is how Lance
marks extension types such as blob and vector columns, travels with the schema.

Two smaller mismatches shape the rest of the design:

* Sail builds DataFusion **without the `sql` feature**, so DataFusion's SQL
  unparser is not available to render a filter for Lance. `src/filter.rs`
  renders the supported expressions itself, which also keeps the output inside
  the SQL subset Lance's own parser accepts.
* Sail's lint configuration **disallows `DmlStatement`**: writes are planned
  through user defined logical nodes and extension planners instead. So the
  write path is a `LanceWriteNode` plus a `LancePhysicalPlanner`, the same shape
  as Sail's Delta, Iceberg and listing table writes.

## How it fits together

```text
  Spark client
       │  Spark Connect (gRPC)
       ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Sail                                                                │
  │   DataSourceRegistry ──"lance"──►  LanceDataSource                  │
  │                                     │              │                │
  │                       create_source │              │ create_writer  │
  │                                     ▼              ▼                │
  │                          LanceTableProvider   LanceWriteNode        │
  │                                     │              │                │
  │   ExtensionQueryPlanner ────────────┼──────────────┤                │
  │                                     ▼              ▼                │
  │                              LanceScanExec    DataSinkExec          │
  │                                     │          (LanceDataSink)      │
  └─────────────────────────────────────┼──────────────┼────────────────┘
      DataFusion 55 / Arrow 59          │              │
  ────────────────────────────────────  │  ───────────  │  ─────────────
      Arrow C data interface       bridge::batch_to_sail │ bridge::batch_to_lance
  ────────────────────────────────────  │  ───────────  │  ─────────────
      DataFusion 54 / Arrow 58          ▼              ▼
                                 Dataset::scan    Dataset::write
```

| Module | What it does |
| --- | --- |
| `src/source.rs` | `LanceDataSource`, the type Sail's registry holds |
| `src/provider.rs` | `LanceTableProvider`: schema, pushdown decisions, fragment to partition assignment |
| `src/exec.rs` | `LanceScanExec`: runs a Lance scan per partition and converts its batches |
| `src/write.rs` | `LanceWriteNode` and `LancePhysicalPlanner` |
| `src/sink.rs` | `LanceDataSink`: streams converted batches into one Lance write transaction |
| `src/bridge.rs` | The Arrow version boundary |
| `src/filter.rs` | DataFusion expressions to Lance filter strings |
| `src/options.rs` | Read and write options, named after the Spark connector's |
| `src/uri.rs` | Table location handling |

## Options

Read options, as `spark.read.format("lance").option(...)`:

| Option | Meaning |
| --- | --- |
| `version` | Dataset version number, or a tag name |
| `batch_size` | Rows per batch the scan produces |
| `with_row_id` | Adds Lance's `_rowid` column |
| `nearest.column`, `nearest.query`, `nearest.k` | Vector search: column, query vector (`[1.0, 2.0]`), and neighbour count. All three are required together |
| `nearest.nprobes`, `nearest.refine_factor`, `nearest.use_index` | Vector index search tuning |

Time travel also comes through Sail's own option layers, so Spark's
`VERSION AS OF` and `TIMESTAMP AS OF` reach the data source: a version number
and a tag name are supported, a timestamp is rejected with a message that says
so.

Write options, as `df.write.format("lance").option(...)`:

| Option | Meaning |
| --- | --- |
| `max_rows_per_file`, `max_rows_per_group`, `max_bytes_per_file` | Write sizing |
| `file_format_version` | Lance file format version, for example `2.1` |
| `enable_stable_row_ids` | Stable row ids, which survive compaction |

Every option may also be given with a `lance.` prefix. An unknown option is an
error rather than a silent no-op; keys that arrive as catalog table properties
are left alone, because they are not addressed to this data source. The Spark
connector spells one of these options `max_row_per_file`; this crate uses the
name of the underlying Lance write parameter, `max_rows_per_file`.

## Running it

This crate is its own Cargo workspace, because it pins the Arrow and DataFusion
versions Sail uses while the Lance workspace is on the older ones. It also pins
its own toolchain, since Sail's MSRV is newer than the Lance one.

```bash
cd integrations/sail
cargo test            # unit tests, and end to end tests through Sail's traits
cargo clippy --all-targets -- -D warnings
```

The tests in `tests/lance_data_source.rs` go through the same entry points Sail
uses: `create_writer` for a write and `create_source` for a read, with the write
planned by `LancePhysicalPlanner` exactly as Sail's extension planner list does
it.

To put it into a Sail server:

```bash
git clone https://github.com/lakehq/sail.git
integrations/sail/scripts/install-into-sail.sh sail
cd sail
cargo test -p sail-lance
cargo run --bin sail -- spark server --port 50051
```

The script copies this crate into `sail/crates/sail-lance` with a manifest that
uses Sail's workspace dependencies, and applies
`patches/register-lance-data-source.patch`, which is the whole integration on
Sail's side:

* one dependency line in `crates/sail-session/Cargo.toml`,
* one `register_data_source(Arc::new(LanceDataSource))` in
  `crates/sail-session/src/formats.rs`,
* one `Arc::new(LancePhysicalPlanner)` in the extension planner list in
  `crates/sail-session/src/planner.rs`.

Then, from any Spark client:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote("sc://localhost:50051").getOrCreate()
spark.range(1000).write.format("lance").mode("overwrite").save("/tmp/ids.lance")
spark.read.format("lance").load("/tmp/ids.lance").filter("id > 900").show()
```

## Design notes

**Filter pushdown is inexact on purpose.** Lance parses and evaluates the filter
string itself, so a difference between DataFusion's and Lance's SQL semantics
could otherwise change a result. Pushed filters are reported as `Inexact`,
which keeps a `FilterExec` above the scan; the pushdown then only ever saves
IO. Before a filter is offered at all it is rendered from a restricted set of
expressions and literal types, and handed to Lance's parser during planning: if
Lance will not take it, the filter stays entirely with DataFusion.

**Partitioning follows fragments.** A scan splits the dataset's fragments
round-robin across `target_partitions` partitions, which is the unit Lance
itself reads in parallel. A vector search is the exception: the nearest `k` rows
of the whole dataset cannot be assembled from independent per-fragment top `k`
lists, so it plans as a single partition and lets Lance's own index search
parallelise inside it.

**A write is one transaction.** `LanceDataSink` streams batches into a single
`Dataset::write`, so a failed write leaves no new dataset version, and an input
that fails part way through fails the writer rather than letting it commit the
rows it has already received. Lance drives its writer from a blocking reader on
a background thread, so batches cross over through a bounded channel rather than
being collected in memory. Spark's four
save modes map onto Lance's create, append and overwrite: `append` creates the
dataset when it does not exist yet, and `ignore` against an existing dataset
writes nothing at all.

**The bridge is the only `unsafe` in the crate**, four functions that move a
C data interface struct between two declarations of it, each with the reasoning
next to it. `bridge::tests` round-trips a batch with nested structs, a
dictionary column, a fixed size list vector column, nulls, and Lance's field
metadata, and asserts the result equals the original.

## Next steps

* **Row level operations.** Lance supports `MERGE`, `UPDATE` and `DELETE`
  natively. Sail routes those through its `LakeSource` trait, which
  `LanceDataSource` does not implement yet; `as_lake_source` returning `None` is
  what makes `registry.get_lake_source("lance")` fail today.
* **Catalogs.** Sail has catalog providers for Glue, Hive, Unity and others.
  Lance has `lance-namespace`, which would slot in the same way and give
  `CREATE TABLE ... USING lance` a home beyond a bare path.
* **Distributed execution.** Sail's cluster mode serializes physical plans
  through a `PhysicalExtensionCodec`. `LanceScanExec` and `LanceDataSink` need a
  codec before they can run on Sail workers rather than in the driver.
* **More of Lance.** Full text search, vector index creation and compaction,
  blob columns, and schema evolution on write are all Lance features with no
  path through this data source yet.
* **Closing the version gap.** When Lance and Sail sit on the same Arrow and
  DataFusion release, `src/bridge.rs` and the duplicate Arrow dependency
  disappear, and `lance-datafusion` can be used directly.

## Status

This is a proof of concept, not a supported integration. It is built against
Lance at the revision it lives in and Sail at revision `67b3cc5e`, which is
pinned in `Cargo.toml`. What has been run:

* `cargo test` here: 34 tests, covering the Arrow bridge, option and filter
  translation, and the read and write paths end to end through Sail's traits.
* `cargo clippy --all-targets -- -D warnings` with Sail's lint configuration.
* `cargo check -p sail-lance --all-targets` and `cargo check -p sail-session`
  inside a Sail checkout with `scripts/install-into-sail.sh` applied, so both
  the crate and the registration compile as part of Sail itself.

Not run here: a Sail server against a real Spark client, and anything on object
storage rather than a local filesystem.
