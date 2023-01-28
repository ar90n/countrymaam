from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN

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
    def __init__(self, metric, params):
        self._metric = metric
        self._index = params.get("index", "kd-tree")
        self._n_trees = params.get("n_trees", 8)
        self._leaf_size = params.get("leaf_size", 8)

    def fit(self, X):
        X = X.astype(np.float64)
        suffix = "".join(random.choices(string.ascii_lowercase, k=16))
        index_file_path = f"index_{suffix}_{os.getpid()}.bin"
        p = subprocess.Popen([
            "countrymaam",
            "train",
            "--dim", str(len(X[0])),
            "--index", self._index,
            "--leaf-size", str(self._leaf_size),
            "--tree-num", str(self._n_trees),
       	    "--output", index_file_path
        ], stdin=subprocess.PIPE)

        p.stdin.write(struct.pack(f"={X.size}d", *np.ravel(X)))
        p.communicate()
        p.stdin.close()

        self._pipe = subprocess.Popen([
            "countrymaam",
            "predict",
            "--dim", str(len(X[0])),
            "--index", self._index,
       	    "--input", index_file_path
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def set_query_arguments(self, search_k):
        self._search_k = search_k

    def query(self, v, n):
        v = v.astype(np.float64)
        self._pipe.stdin.write(struct.pack(f"=i", self._search_k))
        self._pipe.stdin.write(struct.pack(f"=i", n))
        self._pipe.stdin.write(struct.pack(f"={v.size}d", *v))
        self._pipe.stdin.flush()

        rn = struct.unpack("=i", self._pipe.stdout.read(4))[0]
        ret = [0] * rn
        for i in range(rn):
            ret[i] = struct.unpack("=i", self._pipe.stdout.read(4))[0]
        return np.array(ret)

    def __str__(self):
        return f"Countrymaam(index={self._index}, leaf_size={self._leaf_size} n_trees={self._n_trees}, search_k={self._search_k})"
