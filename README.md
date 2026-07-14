# SOPHI

**Sparse-conv Offline Perception with Hydra-distillation Inference**

Open Apache-2.0 stack for training, evaluating, and deploying an end-to-end
camera–LiDAR planner on [NAVSIM](https://github.com/autonomousvision/navsim).
A fused BEV feature \(\mathbf{F}_{\mathrm{env}}\) is shared by a Hydra-MDP-style
trajectory decoder and auxiliary 3D detection + BEV segmentation heads.
The LiDAR sparse-convolution (SCN) path runs on open [`spconv`](https://github.com/traveller59/spconv) 2.x
instead of NVIDIA CUDA-BEVFusion’s closed `libspconv` binary.

| | |
|---|---|
| **Paper (JMLR MLOSS)** | [`jmlr/bevfusion_planner.pdf`](jmlr/bevfusion_planner.pdf) |
| **Version tag** | [`v0.1.0-mloss`](https://github.com/atharvsharma1998/hydra-mdp/releases) |
| **License** | [Apache-2.0](LICENSE) · [NOTICE](NOTICE) |
| **Install** | [`INSTALL.md`](INSTALL.md) |
| **Models / ONNX** | [`docs/MODELS.md`](docs/MODELS.md) |
| **Deploy (export → TRT → C++)** | [`docs/DEPLOY.md`](docs/DEPLOY.md) |
| **Full cookbook** | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |

**Validated results (checkpoint `gtrs_bevfusion_navtrain_v1_best.pth`):**
- NAVSIM `navtest` PDM **0.7925** (12,149 scenarios; NC 0.982, DAC 0.975, TTC 0.981)
- C++ FP16 end-to-end ≈ **391 ms** device time on RTX 3060 (open SCN ≈ 38 ms)
- Python ↔ C++ trajectory max |Δ| **0.0017 m** (cosine 1.000)

<p align="center">
  <img src="assets/demo/sophi_navtrain_v1.gif" width="720"
       alt="SOPHI Python inference on mini (det + BEV seg + planned trajectory)">
  <br/>
  <em>Python inference demo (eval mode) on OpenScene-mini; solid = GT, dashed = predicted.</em>
</p>

---

## Why this repo exists

NVIDIA’s [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution) is a strong detection runtime, but:

1. the LiDAR SCN depends on a **non-redistributable** `libspconv`,
2. the reference exports **detection only** (no planning / BEV seg),
3. there is no documented NAVSIM train → PDM eval → ONNX/TensorRT → C++ path.

SOPHI fills that gap so you can **train your own model**, score it with official
NAVSIM PDM evaluation, and deploy with open SCN + TensorRT.

This is a **software / systems** release: we reuse BEVFusion fusion and a
Hydra-MDP/GTRS-style scorer; the contribution is the open SCN runtime, multi-head
deploy graph, and end-to-end recipe.

---

## Quick start

```bash
git clone https://github.com/atharvsharma1998/hydra-mdp.git
cd hydra-mdp
git checkout v0.1.0-mloss   # reviewed MLOSS version
```

1. [`INSTALL.md`](INSTALL.md) — environment, maps, data layout  
2. [`docs/MODELS.md`](docs/MODELS.md) — download PyTorch checkpoint + ONNX bundle  
3. [`docs/DEPLOY.md`](docs/DEPLOY.md) — export (optional) → TensorRT → **C++ inference**  
4. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — metric cache, train, PDM scoring  

### Python inference

```bash
export BEVFUSION_ROOT=/path/to/CUDA-BEVFusion/bevfusion
export NUPLAN_MAPS_ROOT=/path/to/maps
export PYTHONPATH=$PWD:$PWD/scripts/training:$PYTHONPATH

python scripts/training/viz_gtrs_bevfusion_seg.py \
  --sensor-blobs-path /path/to/mini_sensor_blobs/mini \
  --checkpoint /path/to/gtrs_bevfusion_navtrain_v1_best.pth \
  --out /tmp/sophi_viz.png \
  --eval-mode --show-gt-boxes --num-frames 6
```

### C++ inference (after TensorRT engines are built)

```bash
cd deploy
./build/gtrs_bevfusion parity-data gtrs_bevfusion fp16 parity-data/cpp
# see docs/DEPLOY.md for export + engine build
```

---

## Repository layout

```
navsim/agents/gtrs_bevfusion/   # PyTorch model, losses, features
scripts/training/               # train_gtrs_bevfusion.py, viz_*.py
scripts/export/                 # ONNX export + compare_cpp_parity.py
deploy/                         # C++/CUDA + TensorRT engines + parity-data
docs/                           # install, agents, metrics, REPRODUCIBILITY.md
jmlr/                           # MLOSS paper sources + cover letter
cloud/                          # multi-GPU training helpers
```

Hydra agent config for official scoring:
`navsim/planning/script/config/common/agent/gtrs_bevfusion_agent.yaml`

---

## End-to-end workflow

```
NAVSIM data  →  metric cache  →  train  →  PDM score (navtest)
                                 ↓
                         ONNX export (+ SCN)
                                 ↓
                    TensorRT .plan engines (fp16)
                                 ↓
              C++ gtrs_bevfusion  →  parity vs PyTorch
```

Full copy-paste commands: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## Licensing and dependencies

| Component | Status |
|-----------|--------|
| This repository (train / export / deploy / docs) | **Apache-2.0** |
| Open `spconv` SCN runtime | Open source |
| CUDA-BEVFusion camera / BEVPool template | NVIDIA Apache-2.0; **closed `libspconv` replaced** |
| CUDA + TensorRT 8.6 | Proprietary NVIDIA runtime (required for documented C++ path) |
| NAVSIM / nuPlan / OpenScene data | Separate dataset licenses — download yourself |

See [`NOTICE`](NOTICE) for redistributable boundaries. We do **not** claim a
fully proprietary-free stack; the open claim is the **SCN path + full planning/seg deploy recipe**.

---

## Citation

If you use SOPHI, please cite the JMLR MLOSS paper (update volume/pages after acceptance)
and the upstream benchmarks it builds on:

```bibtex
@article{Sharma2026SOPHI,
  title   = {SOPHI: An Open End-to-End Camera--LiDAR Planning Stack
             with Sparse-Convolution Inference for {NAVSIM}},
  author  = {Sharma, Atharv},
  journal = {Journal of Machine Learning Research (MLOSS)},
  year    = {2026},
  note    = {Software available at https://github.com/atharvsharma1998/hydra-mdp},
}
```

Upstream (please also cite when using NAVSIM evaluation):

```bibtex
@inproceedings{Dauner2024NEURIPS,
  title  = {NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking},
  author = {Dauner, Daniel and Hallgarten, Marcel and Li, Tianyu and Weng, Xinshuo
            and Huang, Zhiyu and Yang, Zetong and Li, Hongyang and Gilitschenski, Igor
            and Ivanovic, Boris and Pavone, Marco and Geiger, Andreas and Chitta, Kashyap},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2024},
}
```

---

## Relationship to upstream NAVSIM

This repository is based on the open [NAVSIM](https://github.com/autonomousvision/navsim)
devkit (Apache-2.0) and adds the SOPHI / GTRS-BEVFusion agent, training, ONNX export,
and C++/TensorRT deploy stack. For the original NAVSIM challenge docs
(agents, splits, metrics, leaderboard), see [`docs/`](docs/) and the
[upstream project](https://github.com/autonomousvision/navsim).

---

## Issues and contributions

Open a [GitHub issue](https://github.com/atharvsharma1998/hydra-mdp/issues) for bugs
or questions. Pull requests that improve install docs, parity tests, or deploy portability
are welcome. Contribution notes and developer pointers are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) (“Extending the stack”).
