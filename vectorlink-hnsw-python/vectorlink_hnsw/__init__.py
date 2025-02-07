import datafusion as df
import pyarrow as pa

from .vectorlink_hnsw import *
__doc__ = vectorlink_hnsw.__doc__
if hasattr(vectorlink_hnsw, "__all__"):
    __all__ = vectorlink_hnsw.__all__

def generate_hnsw_from_vecs(frame: df.DataFrame) -> Hnsw:
    """Generate a new HNSW from a dataframe that has t least the column 'vector'.
    Currently only 1536-sized vectors are supported"""
    vector_frame = frame.select(df.col("vector"))

    # more here..
