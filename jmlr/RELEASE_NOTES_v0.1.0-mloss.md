# Release notes — v0.1.0-mloss

JMLR MLOSS candidate release of SOPHI / GTRS-BEVFusion open stack.

## Highlights

- Open LiDAR SCN inference via spconv 2.x (replaces closed NVIDIA libspconv path)
- Multi-head ONNX/TensorRT deploy: planning + detection + BEV segmentation
- Train / PDM eval / export / C++ parity cookbook: `jmlr/REPRODUCIBILITY.md`
- Validated navtest PDM **0.7925** (12,149 scenarios)
- Python↔C++ trajectory max |Δ| **0.0017 m** (cosine 1.000)

## License

Apache-2.0. See `LICENSE` and `NOTICE` for CUDA/TensorRT and dataset boundaries.

## Tag commands

```bash
cd /path/to/hydra-mdp   # navsim repo root
git add -A   # after reviewing
git commit -m "docs: JMLR MLOSS paper, reproducibility cookbook, NOTICE"
git tag -a v0.1.0-mloss -m "JMLR MLOSS release v0.1.0-mloss"
git push origin HEAD
git push origin v0.1.0-mloss
```

Do **not** push until paper/docs commits are reviewed.
