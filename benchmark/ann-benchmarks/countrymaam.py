from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN
from .countrymaam_wrapper import FlatIndex, KdTreeIndex, RpTreeIndex, AKnnIndex, RpAKnnIndex, get_index_class, IndexType, CountrymaamParam

import subprocess
from dataclasses import asdict

import struct
import subprocess
import sys
import os
import glob
import numpy as np
import random
import string

class Countrymaam(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self._index = None
        self._index_type =  IndexType.from_str(params["index"])
        self._param = CountrymaamParam(
            trees=params.get("trees"),
            leafs=params.get("leafs"),
            sample_features=params.get("sample_features"),
            top_k_candidates=params.get("top_k_candidates"),
            neighbors=params.get("neighbors"),
            rho=params.get("rho"),
            use_profile=params.get("use_profile", False), 
        )

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        clz = get_index_class(self._index_type)
        self._index = clz(X, **asdict(self._param))

    def set_query_arguments(self, search_k):
        self._query_param = {
            "search_k": search_k,
        } 

    def query(self, v, k):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        v = v.reshape(1, -1)

        res= self._index.search(v, k, **self._query_param)
        return res[0]

    def __str__(self):
        ret = []
        for k, v in asdict(self._param).items():
            if v is None:
                continue
            if k == "use_profile":
                continue
            ret.append(f"{k}={v}")
        param_str = ", ".join(ret)
        return f"Countrymaam(index={self._index_type}, {param_str})"
