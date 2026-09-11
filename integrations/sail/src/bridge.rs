// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Moves Arrow data between the two Arrow versions linked into this crate.
//!
//! Lance is built against `arrow` 58 while Sail and DataFusion 55 are built
//! against `arrow` 59. Cargo happily links both, but their `RecordBatch`,
//! `Schema` and `ArrayData` types are unrelated, so data has to cross an
//! explicit boundary.
//!
//! The boundary used here is the Arrow C data interface, which every Arrow
//! version implements identically:
//!
//! ```text
//!   lance::Dataset  --(arrow 58)-->  FFI_ArrowArray / FFI_ArrowSchema  --(arrow 59)-->  DataFusion
//! ```
//!
//! Exporting to the C structs and importing them on the other side is a move of
//! ownership, not a copy: the buffers stay where they are and the importing
//! Arrow version calls the exporting version's release callback when it is done
//! with them.

use std::mem::ManuallyDrop;

use arrow::array::{
    Array as _, RecordBatch as SailBatch, RecordBatchOptions, StructArray as SailStruct,
};
use arrow::datatypes::{Schema as SailSchema, SchemaRef as SailSchemaRef};
use arrow::ffi::{
    FFI_ArrowArray as SailFfiArray, FFI_ArrowSchema as SailFfiSchema, from_ffi as sail_from_ffi,
};
use arrow_lance::array::{
    Array as _, RecordBatch as LanceBatch, RecordBatchOptions as LanceBatchOptions,
    StructArray as LanceStruct,
};
use arrow_lance::datatypes::{Schema as LanceSchema, SchemaRef as LanceSchemaRef};
use arrow_lance::ffi::{
    FFI_ArrowArray as LanceFfiArray, FFI_ArrowSchema as LanceFfiSchema, from_ffi as lance_from_ffi,
    to_ffi as lance_to_ffi,
};
use datafusion_common::{DataFusionError, Result};

/// Converts a schema produced by Lance into the Arrow version DataFusion uses.
///
/// Field-level metadata, which Lance uses for extension types such as blobs and
/// for vector column annotations, is carried by the C data interface and
/// survives the conversion.
pub fn schema_to_sail(schema: &LanceSchema) -> Result<SailSchema> {
    let exported = LanceFfiSchema::try_from(schema).map_err(lance_arrow_error)?;
    Ok(SailSchema::try_from(&lance_schema_into_sail(exported))?)
}

/// Converts a DataFusion schema into the Arrow version Lance uses.
pub fn schema_to_lance(schema: &SailSchema) -> Result<LanceSchema> {
    let exported = SailFfiSchema::try_from(schema)?;
    LanceSchema::try_from(&sail_schema_into_lance(exported)).map_err(lance_arrow_error)
}

/// Converts a batch read from Lance into the Arrow version DataFusion uses.
///
/// `schema` must be the [`schema_to_sail`] conversion of the batch schema. It is
/// passed in so that a scan converts its schema once instead of once per batch.
pub fn batch_to_sail(batch: LanceBatch, schema: SailSchemaRef) -> Result<SailBatch> {
    if batch.num_columns() == 0 {
        // `StructArray` cannot represent a batch without columns, which is what
        // a `SELECT count(*)` scan asks for.
        let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
        return Ok(SailBatch::try_new_with_options(schema, vec![], &options)?);
    }
    let (array, data_type) =
        lance_to_ffi(&LanceStruct::from(batch).into_data()).map_err(lance_arrow_error)?;
    let data_type = lance_schema_into_sail(data_type);
    // SAFETY: `array` and `data_type` were just exported by Arrow 58 through the
    // C data interface and moved here without being released, so they are valid
    // and this is their only import.
    let data = unsafe { sail_from_ffi(lance_array_into_sail(array), &data_type) }?;
    Ok(SailBatch::try_new(
        schema,
        SailStruct::from(data).columns().to_vec(),
    )?)
}

/// Converts a batch produced by DataFusion into the Arrow version Lance uses.
///
/// `schema` must be the [`schema_to_lance`] conversion of the batch schema.
pub fn batch_to_lance(batch: SailBatch, schema: LanceSchemaRef) -> Result<LanceBatch> {
    if batch.num_columns() == 0 {
        let options = LanceBatchOptions::new().with_row_count(Some(batch.num_rows()));
        return LanceBatch::try_new_with_options(schema, vec![], &options)
            .map_err(lance_arrow_error);
    }
    let (array, data_type) = arrow::ffi::to_ffi(&SailStruct::from(batch).into_data())?;
    let data_type = sail_schema_into_lance(data_type);
    // SAFETY: as in `batch_to_sail`, with the two Arrow versions swapped.
    let data = unsafe { lance_from_ffi(sail_array_into_lance(array), &data_type) }
        .map_err(lance_arrow_error)?;
    LanceBatch::try_new(schema, LanceStruct::from(data).columns().to_vec())
        .map_err(lance_arrow_error)
}

/// Reports an Arrow 58 error, which DataFusion cannot convert itself.
fn lance_arrow_error(error: arrow_lance::error::ArrowError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

// The four functions below move an exported C data interface struct from one
// Arrow version's definition of it to the other's. Both versions declare the
// same `#[repr(C)]` struct from the Arrow C data interface specification, so the
// move is field by field; the release callback and the private data it owns
// travel with the struct. The source value is wrapped in `ManuallyDrop` so the
// buffers are released exactly once, by whoever imports the returned struct.

fn lance_array_into_sail(array: LanceFfiArray) -> SailFfiArray {
    let array = ManuallyDrop::new(array);
    SailFfiArray {
        length: array.length,
        null_count: array.null_count,
        offset: array.offset,
        n_buffers: array.n_buffers,
        n_children: array.n_children,
        buffers: array.buffers,
        children: array.children.cast(),
        dictionary: array.dictionary.cast(),
        // SAFETY: the release callback has the same C ABI in both Arrow
        // versions; only the name of the struct it takes differs.
        release: array.release.map(|release| unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(*mut LanceFfiArray),
                unsafe extern "C" fn(*mut SailFfiArray),
            >(release)
        }),
        private_data: array.private_data,
    }
}

fn sail_array_into_lance(array: SailFfiArray) -> LanceFfiArray {
    let array = ManuallyDrop::new(array);
    LanceFfiArray {
        length: array.length,
        null_count: array.null_count,
        offset: array.offset,
        n_buffers: array.n_buffers,
        n_children: array.n_children,
        buffers: array.buffers,
        children: array.children.cast(),
        dictionary: array.dictionary.cast(),
        // SAFETY: see `lance_array_into_sail`.
        release: array.release.map(|release| unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(*mut SailFfiArray),
                unsafe extern "C" fn(*mut LanceFfiArray),
            >(release)
        }),
        private_data: array.private_data,
    }
}

fn lance_schema_into_sail(schema: LanceFfiSchema) -> SailFfiSchema {
    let schema = ManuallyDrop::new(schema);
    SailFfiSchema {
        format: schema.format,
        name: schema.name,
        metadata: schema.metadata,
        flags: schema.flags,
        n_children: schema.n_children,
        children: schema.children.cast(),
        dictionary: schema.dictionary.cast(),
        // SAFETY: see `lance_array_into_sail`.
        release: schema.release.map(|release| unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(*mut LanceFfiSchema),
                unsafe extern "C" fn(*mut SailFfiSchema),
            >(release)
        }),
        private_data: schema.private_data,
    }
}

fn sail_schema_into_lance(schema: SailFfiSchema) -> LanceFfiSchema {
    let schema = ManuallyDrop::new(schema);
    LanceFfiSchema {
        format: schema.format,
        name: schema.name,
        metadata: schema.metadata,
        flags: schema.flags,
        n_children: schema.n_children,
        children: schema.children.cast(),
        dictionary: schema.dictionary.cast(),
        // SAFETY: see `lance_array_into_sail`.
        release: schema.release.map(|release| unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(*mut SailFfiSchema),
                unsafe extern "C" fn(*mut LanceFfiSchema),
            >(release)
        }),
        private_data: schema.private_data,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_lance::array::{
        ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Int32Array, StringArray,
        StructArray,
    };
    use arrow_lance::datatypes::{DataType, Field, Fields, Int8Type};

    use super::*;

    /// A batch that exercises the parts of the Arrow type system a Lance
    /// dataset uses: nested types, dictionaries, nulls, fixed size lists for
    /// vector columns, and the metadata Lance attaches to fields.
    fn sample_batch() -> LanceBatch {
        let ids = Int32Array::from(vec![Some(1), None, Some(3)]);
        let names = StringArray::from(vec![Some("a"), Some("b"), None]);
        let vectors =
            FixedSizeListArray::from_iter_primitive::<arrow_lance::datatypes::Float32Type, _, _>(
                vec![
                    Some(vec![Some(1.0), Some(2.0)]),
                    Some(vec![Some(3.0), Some(4.0)]),
                    None,
                ],
                2,
            );
        let categories: DictionaryArray<Int8Type> =
            vec![Some("x"), Some("y"), Some("x")].into_iter().collect();
        let nested = StructArray::from(vec![
            (
                Arc::new(Field::new("inner", DataType::Float32, true)),
                Arc::new(Float32Array::from(vec![Some(0.5), None, Some(2.5)])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("label", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![Some("p"), Some("q"), Some("r")])) as ArrayRef,
            ),
        ]);

        let vector_field = Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        )
        .with_metadata(HashMap::from([(
            "ARROW:extension:name".to_string(),
            "lance.vector".to_string(),
        )]));
        let schema = LanceSchema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
            vector_field,
            Field::new(
                "category",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new(
                "nested",
                DataType::Struct(Fields::from(vec![
                    Field::new("inner", DataType::Float32, true),
                    Field::new("label", DataType::Utf8, true),
                ])),
                true,
            ),
        ])
        .with_metadata(HashMap::from([(
            "dataset".to_string(),
            "sample".to_string(),
        )]));

        #[expect(clippy::expect_used, reason = "test fixture")]
        LanceBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(ids) as ArrayRef,
                Arc::new(names),
                Arc::new(vectors),
                Arc::new(categories),
                Arc::new(nested),
            ],
        )
        .expect("the sample batch is well formed")
    }

    #[test]
    fn a_batch_survives_a_round_trip_between_the_arrow_versions() -> Result<()> {
        let original = sample_batch();
        let sail_schema = Arc::new(schema_to_sail(original.schema_ref())?);
        let converted = batch_to_sail(original.clone(), Arc::clone(&sail_schema))?;

        assert_eq!(converted.num_rows(), original.num_rows());
        assert_eq!(converted.schema(), sail_schema);
        assert_eq!(
            converted
                .schema()
                .metadata()
                .get("dataset")
                .map(String::as_str),
            Some("sample")
        );
        assert_eq!(
            converted
                .schema()
                .field_with_name("vector")?
                .metadata()
                .get("ARROW:extension:name")
                .map(String::as_str),
            Some("lance.vector"),
            "field metadata must survive the conversion"
        );

        let lance_schema = Arc::new(schema_to_lance(converted.schema_ref())?);
        let round_tripped = batch_to_lance(converted, lance_schema)?;
        assert_eq!(round_tripped, original);
        Ok(())
    }

    #[test]
    fn a_batch_without_columns_keeps_its_row_count() -> Result<()> {
        let schema = Arc::new(LanceSchema::empty());
        let options = LanceBatchOptions::new().with_row_count(Some(7));
        let batch = LanceBatch::try_new_with_options(schema, vec![], &options)
            .map_err(lance_arrow_error)?;
        let converted = batch_to_sail(batch, Arc::new(SailSchema::empty()))?;
        assert_eq!(converted.num_rows(), 7);
        assert_eq!(converted.num_columns(), 0);
        Ok(())
    }
}
