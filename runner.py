import struct
import subprocess
import sys
import os
import string
import random
from tempfile import TemporaryDirectory
from pathlib import Path

def main():
    features = [
        [-0.662, -0.405, 0.508, -0.991, -0.614, -1.639, 0.637, 0.715],
		[0.44, -1.795, -0.243, -1.375, 1.154, 0.142, -0.219, -0.711],
		[0.22, -0.029, 0.7, -0.963, 0.257, 0.419, 0.491, -0.87],
		[0.906, 0.551, -1.198, 1.517, 1.616, 0.014, -1.358, -1.004],
		[0.687, 0.818, 0.868, 0.688, 0.428, 0.582, -0.352, -0.269],
		[-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1],
		[1.606, 1.256, -0.644, -0.858, 0.743, -0.063, 0.042, -1.539],
		[0.255, 1.018, -0.835, -0.288, 0.992, -0.17, 0.764, -1.0],
		[1.061, -0.506, -1.467, 0.043, 1.121, 1.03, 0.596, -1.747],
		[-0.269, -0.346, -0.076, -0.392, 0.301, -1.097, 0.139, 1.692],
		[-1.034, -1.709, -2.693, 1.539, -1.186, 0.29, -0.935, -0.546],
		[1.954, -1.708, -0.423, -2.241, 1.272, -0.253, -1.013, -0.382],
    ]

    index_name = "rkd-tree"
    leaf_size = 5
    max_candidates = 32
    tree_num = 4
    with TemporaryDirectory() as tmpdir:
        output_name = str(Path(tmpdir) / "index.bin")
        print(output_name)
        p = subprocess.Popen([
            "./countrymaam",
            "train",
            "--dim", str(len(features[0])),
            "--index", index_name,
            "--leaf-size", str(leaf_size),
            "--tree-num", str(tree_num),
            "--output", output_name
        ], stdin=subprocess.PIPE)
        for v in sum(features, []):
            p.stdin.write(struct.pack("=d", float(v)))
        p.stdin.flush()
        p.stdin.close()

        p = subprocess.Popen([
            "./countrymaam",
            "predict",
            "--dim", str(len(features[0])),
            "--index", index_name,
            "--input", output_name,
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p.stdin.write(struct.pack(f"=i", max_candidates))
        p.stdin.write(struct.pack(f"=i", 3))
        for v in sum(features[2:3], []):
            p.stdin.write(struct.pack("=d", float(v)))
        p.stdin.flush()
        n = struct.unpack("=i", p.stdout.read(4))[0]
        for i in range(n):
            v = struct.unpack("=i", p.stdout.read(4))[0]
            print(v)


if __name__ == '__main__':
    main()