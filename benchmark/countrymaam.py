from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN

import subprocess

import struct
import subprocess
import sys
import os
import glob
import numpy as np
from pathlib import Path

class Countrymaam(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self._index = params.get("index", "kd-tree")
        self._n_trees = params.get("n_trees", 8)
        self._leaf_size = params.get("leaf_size", 8)

    def fit(self, X):
        X = X.astype(np.float64)
        p = subprocess.Popen([
            "countrymaam",
            "train",
            "--dim", str(len(X[0])),
            "--index", self._index,
            "--leaf-size", str(self._leaf_size),
            "--tree-num", str(self._n_trees),
        ], stdin=subprocess.PIPE)

        p.stdin.write(struct.pack(f"={X.size}d", *np.ravel(X)))
        p.communicate()
        p.stdin.close()

        self._pipe = subprocess.Popen([
            "countrymaam",
            "predict",
            "--dim", str(len(X[0])),
            "--index", self._index,
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def set_query_arguments(self, search_k):
        self._search_k = search_k

    def query(self, v, n):
        v = v.astype(np.float64)
        self._p.stdin.write(struct.pack(f"=i", self._search_k))
        self._p.stdin.write(struct.pack(f"=i", n))
        self._p.stdin.write(struct.pack(f"={v.size}d", *v))
        self._p.stdin.flush()

        rn = struct.unpack("=i", self._p.stdout.read(4))[0]
        ret = [0] * rn
        for i in range(rn):
            ret[i] = struct.unpack("=i", self._p.stdout.read(4))[0]
        return np.array(ret)

    def __str__(self):
        return f"Countrymaam(index={self._index_name}, leaf_size={self._leaf_size} n_trees={self._n_trees}, search_k={self._search_k})"
