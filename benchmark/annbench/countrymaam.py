from __future__ import absolute_import
from .base import BaseANN
from .countrymaam_wrapper import FlatIndex, KdTreeIndex, RpTreeIndex, AKnnIndex, RpAKnnIndex, get_index_class, IndexType, CountrymaamParam
from dataclasses import dataclass, asdict

import socket
import subprocess

import struct
import subprocess
import sys
import os
import glob
import numpy as np
import random
import string



class Countrymaam(BaseANN):
    def __init__(self):
        self._index = None
        self._index_type = None
        self._param = CountrymaamParam()

    def set_index_param(self, param):
        self._index_type = IndexType.from_str(param["index"])
        self._param.use_profile = param.get("use_profile", False)
        self._param.trees = param.get("trees")
        self._param.leafs = param.get("leafs")
        self._param.sample_features = param.get("sample_features")
        self._param.top_k_candidates = param.get("top_k_candidates")
        self._param.neighbors = param.get("neighbors")
        self._param.rho = param.get("rho")

    def has_train(self):
        return False

    def add(self, vecs):
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)

        clz = get_index_class(self._index_type)
        self._index = clz(vecs, **asdict(self._param))

    def query(self, vecs, topk, param):
        if vecs.dtype != np.uint8:
            vecs = vecs.astype(np.float32)

        res= self._index.search(vecs, topk, **param)
        return res

    def write(self, path):
        pass

    def stringify_index_param(self, param):
        ret = []
        for k, v in asdict(self._param).items():
            if v is None:
                continue
            if k == "use_profile":
                continue
            ret.append(f"{k}={v}")
        param_str = ", ".join(ret)
        return f"Countrymaam(index={self._index_type}, {param_str})"
