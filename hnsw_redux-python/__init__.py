# Allow importing using
#   import hnsw_redux
#   hnsw_redux.foo()
# rather than:
#   from hnsw_redux import hnsw_redux

from .hnsw_redux import *

__doc__ = hnsw_redux.__doc__
if hasattr(hnsw_redux, "__all__"):
    __all__ = hnsw_redux.__all__
