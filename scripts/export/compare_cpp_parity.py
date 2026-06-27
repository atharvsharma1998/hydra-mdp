#!/usr/bin/env python3
"""Stage-by-stage parity: C++ pipeline dumps vs PyTorch references.

    python compare_cpp_parity.py --ref <sample>/ref --cpp <cpp-dump-dir>

Both directories hold .tensor files in CUDA-BEVFusion's format. For each stage
present in both, report shape, max|a-b|, mean|a-b| and cosine similarity. Walk
the stages in pipeline order -- the FIRST stage that diverges is where the C++
port deviates from PyTorch; fix that before trusting any later stage.
"""
import argparse
import os

import numpy as np

# fork tensor.hpp DataType codes
_NP = {1: np.int32, 2: np.float16, 3: np.float32, 4: np.int64, 8: np.uint8}

# pipeline order: SCN -> camera LSS -> fuser -> heads
STAGES = [
    "lidar_bev", "cam_bev", "fenv",
    "scores", "trajectory", "agent_states", "agent_class_logits", "bev_semantic_logits",
]


def load_tensor(path):
    with open(path, "rb") as f:
        magic, ndim, code = np.frombuffer(f.read(12), dtype=np.int32)
        assert int(magic) == 0x33ff1101, f"bad magic in {path}"
        dims = np.frombuffer(f.read(4 * int(ndim)), dtype=np.int32).tolist()
        data = np.frombuffer(f.read(), dtype=_NP[int(code)])
    return data.reshape(dims).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True, help="PyTorch reference dir (<sample>/ref)")
    p.add_argument("--cpp", required=True, help="C++ per-stage dump dir")
    a = p.parse_args()

    print(f"{'stage':22s} {'shape':20s} {'max|d|':>11s} {'mean|d|':>11s} {'cos':>8s}  ref[absmax]")
    print("-" * 90)
    for s in STAGES:
        rp = os.path.join(a.ref, s + ".tensor")
        cp = os.path.join(a.cpp, s + ".tensor")
        if not os.path.exists(rp):
            continue
        if not os.path.exists(cp):
            print(f"{s:22s} (no C++ dump)")
            continue
        r, c = load_tensor(rp), load_tensor(cp)
        rf, cf = r.ravel(), c.ravel()
        if rf.shape != cf.shape:
            print(f"{s:22s} SHAPE MISMATCH ref={tuple(r.shape)} cpp={tuple(c.shape)}")
            continue
        d = np.abs(rf - cf)
        denom = np.linalg.norm(rf) * np.linalg.norm(cf) + 1e-9
        cos = float(np.dot(rf, cf) / denom)
        flag = "" if cos > 0.99 else "  <-- DIVERGES"
        print(f"{s:22s} {str(tuple(r.shape)):20s} {d.max():11.4f} {d.mean():11.5f} "
              f"{cos:8.4f}  {np.abs(rf).max():9.3f}{flag}")


if __name__ == "__main__":
    main()
