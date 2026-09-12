# Lance for Sail

A proof of concept that makes [Sail](https://github.com/lakehq/sail), the Rust
implementation of the Spark Connect server, read and write Lance datasets.
Everything here is Rust: no JVM, no Spark, no JNI.

```python
# against a Sail server with this crate registered
spark.read.format("lance").load("s3://bucket/embeddings.lance")
df.write.format("lance").mode("overwrite").save("/data/embeddings.lance")
```

The existing [Apache Spark connector for Lance](https://github.com/lance-format/lance-spark)
gives Spark the same capability through the JVM: Spark DSv2 in Java, calling
Lance over JNI. Sail replaces the Spark JVM engine with DataFusion, so the
connector cannot be reused, but it also does not need to be: Sail reaches its
formats through a Rust trait, and Lance is a Rust library.

## What this is

`sail-lance/` is a Sail crate. It belongs at `crates/sail-lance` in a Sail
checkout, which is what `scripts/install-into-sail.sh` does, together with the
four-file patch in `patches/` that registers it.

| Capability | Status |
| --- | --- |
| Read a dataset, schema from the dataset itself | yes |
| Column projection pushdown | yes |
| Filter pushdown, as DataFusion expressions | yes, exact: no filter is re-applied above the scan |
| Limit pushdown | yes |
| Parallel scans, one partition per fragment group | yes |
| Vector search (`nearest`) pushed into the scan | yes |
| Time travel by version or tag | yes |
| Write: append, overwrite, error-if-exists, ignore-if-exists | yes |
| Row level `MERGE`/`UPDATE`/`DELETE`, catalogs, index creation | not yet, see [Next steps](#next-steps) |

## Why Sail v0.7.1

Both projects are built on DataFusion and Arrow, so the integration only works
where the versions line up:

| | DataFusion | arrow-rs | object_store |
| --- | --- | --- | --- |
| Lance 11.0.0 | ^54.0 | ^58.0 | ^0.13 |
| **Sail v0.7.1** (August 2026, the latest release) | **54.1** | **58.3** | **0.13.2** |
| Sail `main` (September 2026) | 55.0 | 59.2 | 0.13.2 |

Together they resolve to one DataFusion 54.1, one Arrow 58.4 and one
object_store 0.13.2 in the lockfile.

At v0.7.1 the two sides share every relevant crate, so a `RecordBatch` read from
Lance *is* a `RecordBatch` DataFusion can execute on, and a DataFusion filter
`Expr` *is* an expression Lance can evaluate. Nothing is converted, copied or
rendered at the boundary.

Sail `main` has since moved to DataFusion 55 and Arrow 59. Lance and Sail then
hold unrelated `RecordBatch` types in one process, and data has to cross through
the Arrow C data interface, filters have to be rendered as SQL strings for
Lance's parser, and the crate needs two Arrow versions in its dependency tree.
That version of this integration is in the history of this branch
(`integrations/sail`, commit `cf13015`); it works, and it needs no change to
Sail's own code, but it carries a bridge that this one does not need.

## Three Sail changes are required

Matching versions is necessary but not sufficient. Lance and Sail want
different DataFusion **features**, and a Cargo build has only one feature set
per crate:

* Lance needs DataFusion's `sql` feature. `lance`, `lance-datafusion` and
  `lance-index` all parse SQL filter strings.
* Sail is built with `sql` off, and its code depends on that. With the feature
  off, `datafusion_expr::sql` is a set of stand-in types DataFusion provides
  precisely for this case; with it on, they are the real `sqlparser` types, and
  `DataFusionError` gains an `SQL` variant.

So adding Lance to a Sail workspace breaks Sail before it breaks anything else:

```
error[E0004]: non-exhaustive patterns: `&DataFusionError::SQL(_, _)` not covered
   --> crates/sail-common-datafusion/src/error.rs:143:26
error[E0432]: unresolved import `datafusion_expr::sql`
   --> crates/sail-plan/src/resolver/expression/wildcard.rs:6:22
```

`patches/lance-table-format.patch` therefore does three things besides
registering the format:

1. turns on DataFusion's `sql` feature in the workspace manifest,
2. handles the `SQL` error variant in `sail-common-datafusion`, and
3. builds the Spark wildcard options in `sail-plan` from `sqlparser` types
   rather than from strings, which is what DataFusion's stand-in types are.

That is twenty-odd lines in `sail-plan`, one in `sail-common-datafusion` and
three in the workspace manifest. It is worth upstreaming on its own: DataFusion
ships those stand-in types so that code can switch between the two worlds, and
Sail only compiles in one of them today.

Note that this conflict is *created* by aligning the versions. Against Sail
`main`, where DataFusion 55 and DataFusion 54 coexist as separate crates, the
feature sets never meet — the price being the Arrow bridge described below.

## How it fits together

```text
  Spark client
       │  Spark Connect (gRPC)
       ▼
  ┌────────────────────────────────────────────────────────────────┐
  │ Sail                                                           │
  │   TableFormatRegistry ──"lance"──►  LanceTableFormat           │
  │                                      │             │           │
  │                        create_source │             │ create_writer
  │                                      ▼             ▼           │
  │                           LanceTableProvider  LanceWriteNode   │
  │                                      │             │           │
  │   ExtensionQueryPlanner ─────────────┼─────────────┤           │
  │                                      ▼             ▼           │
  │                               LanceScanExec   DataSinkExec     │
  │                                      │        (LanceDataSink)  │
  └──────────────────────────────────────┼─────────────┼───────────┘
                                         ▼             ▼
                                  Dataset::scan   InsertBuilder
                                         DataFusion 54.1 / Arrow 58.3
                                            on both sides
```

| Module | What it does |
| --- | --- |
| `src/format.rs` | `LanceTableFormat`, the type Sail's registry holds |
| `src/provider.rs` | `LanceTableProvider`: schema, pushdown decisions, fragment to partition assignment |
| `src/exec.rs` | `LanceScanExec`: runs a Lance scan per partition |
| `src/write.rs` | `LanceWriteNode` and `LancePhysicalPlanner` |
| `src/sink.rs` | `LanceDataSink`: streams the input into one Lance write transaction |
| `src/filter.rs` | Which filters are worth offering to a Lance scan |
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
`VERSION AS OF` and `TIMESTAMP AS OF` reach the format: a version number and a
tag name are supported, a timestamp is rejected with a message that says so.

Write options, as `df.write.format("lance").option(...)`:

| Option | Meaning |
| --- | --- |
| `max_rows_per_file`, `max_rows_per_group`, `max_bytes_per_file` | Write sizing |
| `file_format_version` | Lance file format version, for example `2.1` |
| `enable_stable_row_ids` | Stable row ids, which survive compaction |

Every option may also be given with a `lance.` prefix. An unknown option is an
error rather than a silent no-op; keys that arrive as catalog table properties
are left alone, because they are not addressed to this format. The Spark
connector spells one of these options `max_row_per_file`; this crate uses the
name of the underlying Lance write parameter, `max_rows_per_file`.

## Running it

```bash
git clone --branch v0.7.1 https://github.com/lakehq/sail.git
integrations/sail/scripts/install-into-sail.sh sail
cd sail
cargo test -p sail-lance
cargo clippy -p sail-lance --all-targets -- -D warnings
cargo run --bin sail -- spark server --port 50051
```

The crate is not a member of the Lance workspace and is not built from this
repository: it is a Sail crate that lives here, and the script is how it gets
to where it builds. Its tests go through the same entry points Sail uses —
`create_writer` for a write and `create_source` for a read, with the write
planned by `LancePhysicalPlanner` exactly as Sail's extension planner list does
it.

Then, from any Spark client:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote("sc://localhost:50051").getOrCreate()
spark.range(1000).write.format("lance").mode("overwrite").save("/tmp/ids.lance")
spark.read.format("lance").load("/tmp/ids.lance").filter("id > 900").show()
```

## Design notes

**Filter pushdown is exact.** Lance takes the DataFusion `Expr` itself through
`Scanner::filter_expr` and evaluates it with the same DataFusion version that
built it, so DataFusion drops its own `FilterExec`. Scalar functions work too:
`character_length(name) = 3` is evaluated inside the scan, and `name` is read
for the filter without being added to the output projection. Before a filter is
offered, Lance binds it against the dataset during planning; an expression it
cannot bind stays with DataFusion.

**Partitioning follows fragments.** A scan splits the dataset's fragments
round-robin across `target_partitions` partitions, which is the unit Lance
itself reads in parallel. A vector search is the exception: the nearest `k` rows
of the whole dataset cannot be assembled from independent per-fragment top `k`
lists, so it plans as a single partition and lets Lance's own index search
parallelise inside it.

**A write is one transaction.** `LanceDataSink` hands the input stream straight
to Lance's `InsertBuilder`, so a failure part way through fails the write rather
than committing the rows already received, and no batch is buffered in memory.
Spark's four save modes map onto Lance's create, append and overwrite:
`append` creates the dataset when it does not exist yet, and `ignore` against an
existing dataset writes nothing at all.

**There is no `unsafe` in the crate.** With the versions aligned there is
nothing to transmute or reinterpret at the boundary.

## Next steps

* **Row level operations.** Lance supports `MERGE`, `UPDATE` and `DELETE`
  natively, and Sail's `TableFormat` trait has `create_deleter` and
  `create_merger` hooks with default "not implemented" bodies. Filling those in
  is the natural next piece of work.
* **Catalogs.** Sail has catalog providers for Glue, Hive, Unity and others.
  Lance has `lance-namespace`, which would slot in the same way and give
  `CREATE TABLE ... USING lance` a home beyond a bare path.
* **Distributed execution.** Sail's cluster mode serializes physical plans
  through a `PhysicalExtensionCodec`. `LanceScanExec` and `LanceDataSink` need a
  codec before they can run on Sail workers rather than in the driver.
* **More of Lance.** Full text search, vector index creation and compaction,
  blob columns, and schema evolution on write are all Lance features with no
  path through this format yet.
* **Sail `main`.** Keeping this integration on a released Sail means waiting
  for Lance and Sail to agree on DataFusion 55 and Arrow 59. Until they do, the
  bridging version in this branch's history is what runs against `main`.

## Status

A proof of concept, not a supported integration, built against Lance 11.0.0 from
crates.io and Sail at tag v0.7.1. What has been run, in a Sail checkout prepared
by `scripts/install-into-sail.sh`:

* `cargo test -p sail-lance` — 27 tests, covering option and filter handling and
  the read and write paths end to end through Sail's traits.
* `cargo clippy -p sail-lance --all-targets -- -D warnings` and
  `cargo fmt -p sail-lance --check`, with Sail's own lint and format
  configuration.
* `cargo check -p sail-session` — the whole Sail session compiles with the
  format registered.
* `cargo test -p sail-plan` — Sail's own planner tests still pass with the
  `sql` feature on and the wildcard change applied.

Not run here: a Sail server against a real Spark client, and anything on object
storage rather than a local filesystem.
