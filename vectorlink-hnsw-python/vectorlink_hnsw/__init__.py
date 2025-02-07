import datafusion as df
import pyarrow as pa

from .vectorlink_hnsw import *
#__doc__ = vectorlink_hnsw.__doc__
#if hasattr(vectorlink_hnsw, "__all__"):
#    __all__ = vectorlink_hnsw.__all__

VECTOR_1536_SCHEMA = pa.schema([
    pa.field("embedding", pa.list_(pa.float32(), 1536), nullable=False)
])

LAYER_SCHEMA = pa.schema([
    pa.field("layer_id", pa.uint8(), nullable=False),
    pa.field("vector_id", pa.uint32(), nullable=False),
    pa.field("neighborhood", pa.list_(pa.uint32()), nullable=False)
])

def generate_hnsw_from_dataframe(bp: BuildParams, frame: df.DataFrame) -> Hnsw:
    """Generate a new HNSW from a dataframe that has at least the column 'embedding'.
    Currently only 1536-sized vectors are supported.
    Vectors are assumed to already be sorted in the order required."""
    stream = frame.__arrow_c_stream__(VECTOR_1536_SCHEMA.__arrow_c_schema__())
    vecs = Vectors.from_arrow(stream, frame.count())
    hnsw = Hnsw.generate_with_cosine_distance_1536(bp, vecs)

    return hnsw

def hnsw_to_record_batch_reader(hnsw: Hnsw) -> pa.RecordBatchReader:
    """Generate a RecordBatchReader from an HNSW with the columns layer_id, vector_id and neighborhood"""
    layer_batches = []
    for i in range(0,hnsw.layer_count()):
        layer_array = hnsw.layer_arrow_array(i)
        layer_id_array = pa.array([i for _ in range(0,len(layer_array))])
        vector_id_array = pa.array([v for v in range(0,len(layer_array))])
        dynsized_layer_array = layer_array.cast(pa.list_(pa.uint32()))
        batch = pa.RecordBatch.from_arrays([layer_id_array, vector_id_array, dynsized_layer_array], schema=LAYER_SCHEMA)
        layer_batches.append(batch)


    return pa.RecordBatchReader.from_batches(LAYER_SCHEMA, layer_batches)

def hnsw_to_dataframe(ctx: df.SessionContext, hnsw: Hnsw) -> df.DataFrame:
    """Generate a DataFrame from an HNSW with the columns layer_id, vector_id and neighborhood"""
    reader = hnsw_to_record_batch_reader(hnsw)

    return ctx.from_arrow(reader)

def hnsw_from_dataframe(frame: df.DataFrame) -> Hnsw:
    """Load an HNSW from a dataframe containing the columns 'layer_id', 'vector_id', and 'neighborhood'
    This will
    - group the dataframe by layer_id
    - load each group in its own layer, ensuring equal neighborhood sizes
    - compose this into an HNSW"""
    layer_ids = frame.select(df.col('layer_id')).distinct().sort(df.col('layer_id')).to_pydict()["layer_id"]
    expected_layer_ids = list([i for i in range(0, len(layer_ids))])
    assert layer_ids == expected_layer_ids

    layers = []
    for layer_id in layer_ids:
        neighborhoods = frame.filter(df.col('layer_id') == layer_id).sort(df.col('vector_id')).select(df.col('neighborhood'))
        neighborhood_size = neighborhoods.select(df.functions.array_length(df.col('neighborhood')).alias('size')).limit(1).to_pylist()[0]["size"]
        # silent assumption here is that all neighborhoods of a layer have the same size
        neighborhood_list_type = pa.list_(pa.uint32(), neighborhood_size)
        layer_schema = pa.schema([
            pa.field("neighborhood", neighborhood_list_type, nullable=False)
        ])
        cast_frame = neighborhoods.select(df.col('neighborhood').cast(neighborhood_list_type).alias('neighborhood'))
        stream = cast_frame.__arrow_c_stream__(layer_schema.__arrow_c_schema__())
        layer = Layer.from_arrow(stream, neighborhoods.count())
        layers.append(layer)

    return Hnsw(layers)
