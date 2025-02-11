import datafusion as df
import pyarrow as pa

from .vectorlink_hnsw import *

v: Vectors = vectorlink_hnsw.Vectors(0)


def generate_hnsw_from_vecs(frame: df.DataFrame) -> Hnsw:
    """Generate a new HNSW from a dataframe that has t least the column 'vector'.
    Currently only 1536-sized vectors are supported"""
    vector_frame = frame.select(df.col("vector"))

    # more here..
