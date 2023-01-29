from __future__ import absolute_import
from .base import BaseANN

import subprocess

import struct
import subprocess
import sys
import os
import glob
import numpy as np
import random
import string


def get_countrymaam_launch_cmd():
    if bin_path := os.environ.get("COUNTRYMAAM_BIN"):
        return [bin_path]

    return [
        "go",
        "run",
        "../../../../../cmd/countrymaam/main.go"
    ]

class Countrymaam(BaseANN):
    def __init__(self):
        self._index = None
        self._n_trees = None
        self._leaf_size = None
        self._pipe = None

    def set_index_param(self, param):
        self._index = param.get("index", "rkd-tree")
        self._n_trees = param.get("n_trees", 8)
        self._leaf_size = param.get("leaf_size", 8)

    def has_train(self):
        return False

    def add(self, vecs):
        vecs = vecs.astype(np.float32)
        D = len(vecs[0])

        suffix = "".join(random.choices(string.ascii_lowercase, k=16))
        index_file_path = f"index_{suffix}_{os.getpid()}.bin"
        p = subprocess.Popen([
            *get_countrymaam_launch_cmd(),
            "train",
            "--dim", str(D),
            "--index", self._index,
            "--leaf-size", str(self._leaf_size),
            "--tree-num", str(self._n_trees),
       	    "--output", index_file_path
        ], stdin=subprocess.PIPE)
        p.stdin.write(struct.pack(f"={vecs.size}f", *np.ravel(vecs)))
        p.communicate()
        p.stdin.close()

        self.read(index_file_path, D)

    def query(self, vecs, topk, param):
        res = []
        for v in vecs:
            v = v.astype(np.float32)
            self._pipe.stdin.write(struct.pack(f"=i", int(param["search_k"])))
            self._pipe.stdin.write(struct.pack(f"=i", int(topk)))
            self._pipe.stdin.write(struct.pack(f"={v.size}f", *v))
            self._pipe.stdin.flush()

            rn = struct.unpack("=i", self._pipe.stdout.read(4))[0]
            ret = [0] * rn
            for i in range(rn):
                ret[i] = struct.unpack("=i", self._pipe.stdout.read(4))[0]
            res.append(np.array(ret))
        return res


    def write(self, path):
        pass

    def read(self, path, D):
        self._pipe = subprocess.Popen([
            *get_countrymaam_launch_cmd(),
            "predict",
            "--dim", str(D),
            "--index", self._index,
       	    "--input", path
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def stringify_index_param(self, param):
        return f"Countrymaam(index={self._index}, leaf_size={self._leaf_size} n_trees={self._n_trees})"
