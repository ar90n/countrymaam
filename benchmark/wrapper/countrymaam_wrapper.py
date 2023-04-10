import ctypes
import numpy as np
import numpy.typing as npt

from typing import Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict


countrymaam = ctypes.CDLL(Path(__file__).parent.joinpath('libcountrymaam_wrapper.so').as_posix())

countrymaam.NewFlatIndex.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
countrymaam.NewFlatIndex.restype = ctypes.c_longlong

countrymaam.NewKdTreeIndex.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
countrymaam.NewKdTreeIndex.restype = ctypes.c_longlong

countrymaam.NewRpTreeIndex.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
countrymaam.NewRpTreeIndex.restype = ctypes.c_longlong

countrymaam.NewAKnnIndex.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool)
countrymaam.NewAKnnIndex.restype = ctypes.c_longlong

countrymaam.NewRpAKnnIndex.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool)
countrymaam.NewRpAKnnIndex.restype = ctypes.c_longlong

countrymaam.Search.argtypes = (ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p,  ctypes.c_int, ctypes.c_int, ctypes.c_int)
countrymaam.Search.restype = None

def _get_raw_features(features):
    if features.ndim == 1:
        features = features.reshape(1, -1)

    rows, cols = features.shape
    data_ptr = features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return data_ptr, rows, cols

def _create_neighbors_array(rows, k):
    neighbors = np.zeros((rows,k), dtype=np.int32)
    neighbors_ptr = neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    return neighbors, neighbors_ptr

def _get_profile_output_name_suffix(**kwargs: dict[str, Any]) -> str:
    ret = ""
    for key, value in kwargs.items():
        if value is None:
            continue
        if key == "use_profile":
            continue
        ret += f"_{key}_{value}"
    return ret

class IndexType(Enum):
    FLAT = "flat"
    KD_TREE = "kd_tree"
    RP_TREE = "rp_tree"
    AKNN = "aknn"
    RP_AKNN = "rp_aknn"

    @classmethod
    def from_str(cls, s):
        return {
            str(cls.FLAT): cls.FLAT,
            str(cls.KD_TREE): cls.KD_TREE,
            str(cls.RP_TREE): cls.RP_TREE,
            str(cls.AKNN): cls.AKNN,
            str(cls.RP_AKNN): cls.RP_AKNN,
        }[s]

    def __str__(self):
        return str(self.value)

def get_index_class(index_type: IndexType) -> type:
    return {
        IndexType.FLAT: FlatIndex,
        IndexType.KD_TREE: KdTreeIndex,
        IndexType.RP_TREE: RpTreeIndex,
        IndexType.AKNN: AKnnIndex,
        IndexType.RP_AKNN: RpAKnnIndex,
    }[index_type]

@dataclass
class CountrymaamParam:
    trees: int | None = None
    leafs: int | None = None
    sample_features: int | None = None
    top_k_candidates: int | None = None
    neighbors: int | None = None
    rho: float | None = None
    use_profile: bool = False

class FlatIndex:
    _symbol: ctypes.c_longlong
    _size: int

    def __init__(self, features: npt.NDArray[np.float32], use_profile: bool = False, **kwargs):
        data_ptr, rows, cols = _get_raw_features(features)
        self._size = rows
        self._symbol = countrymaam.NewFlatIndex(data_ptr, rows, cols, ctypes.c_bool(use_profile))

    def search(self, queries: npt.NDArray[np.float32], k: int, **kwargs) -> npt.NDArray[np.int32]:
        data_ptr, rows, cols = _get_raw_features(queries)
        neighbors, neighbors_ptr = _create_neighbors_array(rows, k)
        countrymaam.Search(self._symbol, data_ptr, neighbors_ptr, rows, cols, k, self._size)
        return neighbors

class KdTreeIndex:
    _symbol: ctypes.c_longlong
    _profile_output_name_suffix: str | None = None

    def __init__(self,
        features: npt.NDArray[np.float32],
        *,
        leafs: int | None = None,
        trees: int | None = None,
        sample_features: int | None = None,
        top_k_candidates: int | None = None,
        use_profile: bool = False,
        **kwargs
    ):
        if leafs is None:
            leafs = -1
        if trees is None:
            trees = -1
        if sample_features is None:
            sample_features = -1
        if top_k_candidates is None:
            top_k_candidates = -1

        self._features = np.copy(features)

        data_ptr, rows, cols = _get_raw_features(self._features)
        self._symbol = countrymaam.NewKdTreeIndex(data_ptr, rows, cols, leafs, trees, sample_features, top_k_candidates, ctypes.c_bool(use_profile))

    def search(self, queries: npt.NDArray[np.float32], k: int, *, search_k: int | None = None, **kwargs) -> npt.NDArray[np.int32]:
        if search_k is None:
            search_k = k

        data_ptr, rows, cols = _get_raw_features(queries)
        neighbors, neighbors_ptr = _create_neighbors_array(rows, k)
        countrymaam.Search(self._symbol, data_ptr, neighbors_ptr, rows, cols, k, search_k)
        return neighbors


class RpTreeIndex:
    _symbol: ctypes.c_longlong
    _profile_output_name_suffix: str | None = None

    def __init__(
        self,
        features: npt.NDArray[np.float32],
        *,
        leafs: int | None = None,
        trees: int | None = None,
        sample_features: int | None = None,
        use_profile: bool = False,
        **kwargs,
    ):
        if leafs is None:
            leafs = -1
        if trees is None:
            trees = -1
        if sample_features is None:
            sample_features = -1

        self._features = np.copy(features)

        data_ptr, rows, cols = _get_raw_features(self._features)
        self._symbol = countrymaam.NewRpTreeIndex(data_ptr, rows, cols, leafs, trees, sample_features, ctypes.c_bool(use_profile))

    def search(self, queries: npt.NDArray[np.int32], k: int, *, search_k: int | None = None, **kwargs) -> npt.NDArray[np.int32]:
        if search_k is None:
            search_k = k

        data_ptr, rows, cols = _get_raw_features(queries)
        neighbors, neighbors_ptr = _create_neighbors_array(rows, k)
        countrymaam.Search(self._symbol, data_ptr, neighbors_ptr, rows, cols, k, search_k)
        return neighbors

class AKnnIndex:
    _symbol: ctypes.c_longlong
    _profile_output_name_suffix: str | None = None

    def __init__(
        self,
        features: npt.NDArray[np.float32],
        *,
        neighbors: int | None = None,
        rho: float | None = None,
        use_profile: bool = False,
        **kwargs
    ):
        if neighbors is None:
            neighbors = -1
        if rho is None:
            rho = -1.0

        self._features = np.copy(features)

        data_ptr, rows, cols = _get_raw_features(self._features)
        self._symbol = countrymaam.NewAKnnIndex(data_ptr, rows, cols, neighbors, rho, ctypes.c_bool(use_profile))

    def search(self, queries: npt.NDArray[np.float32], k: int, **kwargs) -> npt.NDArray[np.int32]:
        data_ptr, rows, cols = _get_raw_features(queries)
        neighbors, neighbors_ptr = _create_neighbors_array(rows, k)
        countrymaam.Search(self._symbol, data_ptr, neighbors_ptr, rows, cols, k, k)
        return neighbors


class RpAKnnIndex:
    _symbol: ctypes.c_longlong
    _profile_output_name_suffix: str | None = None

    def __init__(
        self,
        features: npt.NDArray[np.float32],
        *,
        leafs: int | None = None,
        entries: int | None = None,
        neighbors: int | None = None,
        rho: float | None = None,
        use_profile: bool = False,
        **kwargs,
    ):
        if leafs is None:
            leafs = -1
        if entries is None:
            entries = -1
        if neighbors is None:
            neighbors = -1
        if rho is None:
            rho = -1.0

        self._features = np.copy(features)

        data_ptr, rows, cols = _get_raw_features(self._features)
        self._symbol = countrymaam.NewRpAKnnIndex(data_ptr, rows, cols, leafs, entries, neighbors, rho, ctypes.c_bool(use_profile))

    def search(self, queries: npt.NDArray[np.float32], k: int, **kwargs) -> npt.NDArray[np.int32]:
        data_ptr, rows, cols = _get_raw_features(queries)
        neighbors, neighbors_ptr = _create_neighbors_array(rows, k)
        countrymaam.Search(self._symbol, data_ptr, neighbors_ptr, rows, cols, k, k)
        return neighbors
