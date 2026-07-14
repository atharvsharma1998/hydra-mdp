# How the LiDAR Backbone ONNX Runs in C++

A plain-language guide to SOPHI’s **open SCN path**: why we don’t use ONNX Runtime / TensorRT for LiDAR, what a “custom ONNX interpreter” is, and how ours is written and executed.

**Code map**

| Piece | Location |
|--------|----------|
| Pipeline wiring | `navsim/deploy/src/gtrs/gtrs_pipeline.cu` |
| SCN wrapper (voxelize → ONNX model) | `CUDA-BEVFusion/src/bevfusion/lidar-scn.cpp` |
| Custom parser + executor | `CUDA-BEVFusion/src/bevfusion/lidar-scn-onnx-parser-custom.{hpp,cu}` |
| ONNX file | `deploy/model/gtrs_bevfusion/lidar.backbone.xyz.onnx` |
| Sparse kernels | open **spconv 2.x** (`libspconv.so`) |

---

## 1. The one-sentence mental model

```text
LiDAR points
  → range crop
  → voxelize
  → custom ONNX interpreter calls open spconv kernels
  → dense lidar_bev [1, 256, 100, 100]
  → TensorRT fuser (+ camera) → F_env → planning / det / seg
```

**Critical distinction**

| Graph | Runner |
|--------|--------|
| Camera, fuser, planning, det, seg | **TensorRT** (`.plan` engines) |
| LiDAR backbone SCN | **Not TensorRT.** Custom ONNX parser + **open spconv** |

So: the LiDAR file is still an **ONNX file**, but it is **not** executed by ONNX Runtime or `trtexec`.

---

## 2. Why not ONNX Runtime (or TensorRT) for LiDAR?

### What ONNX is

ONNX is a **file format**: a list of operators + weights (protobuf).  
Something must still **interpret** that file and run GPU kernels.

### What ONNX Runtime / TensorRT are good at

Standard dense ops: `Conv`, `MatMul`, `Relu`, `Add`, `Reshape`, attention-friendly patterns, etc.

### What’s in `lidar.backbone.xyz.onnx`

Custom / sparse ops, roughly:

- `SparseConvolution` (submanifold or strided 3D sparse conv)
- `Relu`, `Add` (residuals)
- `ScatterDense` / `ToDense` (sparse → dense volume)
- `Transpose`, `Reshape` (pack Z into BEV channels)

Those sparse ops are **not** in the default ONNX / TensorRT opsets.  
To use ORT or TRT you’d still need a **custom plugin** that implements sparse convolution — i.e. the same hard work, different packaging.

### Our choice

| Approach | Meaning |
|----------|---------|
| ORT + custom ops | Still write sparse kernels as ORT plugins |
| TensorRT + plugins | Same |
| NVIDIA closed `libspconv` | Proprietary SCN engine (what we replaced) |
| **Custom parser → open spconv** | Parse ONNX ourselves; call redistributable `spconv` |

**Analogy:** ONNX Runtime is a general media player. Our SCN ONNX uses a rare codec (`SparseConvolution`). The custom interpreter is a small special-purpose player that uses the `spconv` library as the decoder.

---

## 3. What “custom ONNX interpreter” means

It does **not** mean a new file format.

It means C++ code that:

1. **Reads** `lidar.backbone.xyz.onnx` (protobuf)
2. **Builds** an ordered list of layer configs + GPU weights (`load`)
3. **Each frame**, walks that list and runs the matching kernels (`forward`)

You are writing a **tiny domain-specific VM** that only understands this SCN graph — not a full ONNX Runtime clone.

```text
ONNX file
  → load(): protobuf → vector<LayerConfig> + device weights
  → forward(): for each layer, dispatch to spconv / CUDA
  → dense BEV feature
```

---

## 4. End-to-end steps (runtime)

### 4.0 Export (Python, once per checkpoint)

```bash
python scripts/export/export_gtrs_bevfusion_onnx.py ... --lidar-scn
```

Produces `lidar.backbone.xyz.onnx`.  
This file is **staged as-is**; it is **not** converted by `trtexec`.

### 4.1 Startup (`Pipeline` constructor)

In `gtrs_pipeline.cu`:

1. Set NAVSIM voxel geometry, e.g.:
   - range: XY ±32 m, Z ∈ [-3, 5]
   - voxel size: `0.08 × 0.08 × 0.2` → grid about `(800, 800, 40)`
2. Point `scn.model` at `.../lidar.backbone.xyz.onnx`
3. Call `bevfusion::lidar::create_scn(scn)`

Inside `SCNImplement::init` (`lidar-scn.cpp`):

1. Create the **voxelizer**
2. `load_onnx_lidar_model(path)` → construct `ONNXLiDARModel` and call `load()`

### 4.2 Per frame (`Pipeline::forward` → `scn_->forward`)

**A. Optional range crop** (`LidarFrontend`)  
Drop points outside the BEV range. Points stay **5 channels**: `x, y, z, intensity, t`.

**B. Voxelization** (CUDA, still not ONNX)  
Bins points into voxels →

- features `[N_vox, 5]` FP16  
- indices `[N_vox, 4]` int32 (batch + spatial coords)  
- spatial shape for spconv: `[Z, Y, X]` (e.g. `[40, 800, 800]`)

**C. `ONNXLiDARModel::forward(features, indices, …)`**  
Custom interpreter runs the SCN graph (next sections).

**D. Handoff**  
Output `lidar_bev` `[1, 256, 100, 100]` FP16 goes into the **TensorRT fuser** with `cam_bev` → `F_env` → heads.

---

## 5. How the custom parser is written

Two phases again: **`load()`** once, **`forward()`** every frame.

### 5.1 Data structures

```cpp
struct LayerConfig {
  std::string type;   // "SparseConvolution", "Relu", "Add", ...
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  // SparseConvolution:
  std::vector<int> kernel_size, stride, padding, dilation;
  bool subm;
  LayerWeights weights;  // FP16 weight + bias
  // ScatterDense / Transpose / Reshape fields...
};

std::vector<LayerConfig> layers_;
```

At runtime, tensors are looked up **by ONNX name**:

```cpp
struct SparseState {
  Tensor features;              // [N, C] fp16
  Tensor indices;               // [N, 4] int32
  int num_points;
  std::vector<int> spatial_shape;  // [Z, Y, X]
};

std::unordered_map<std::string, SparseState> reg;
```

### 5.2 `load()` — parse ONNX into an executable plan

Roughly:

1. Read file → `onnx::ModelProto`
2. Inspect `graph.node()`, `graph.initializer()`, outputs
3. For each node, branch on `op_type()`:

| `op_type` | What we store |
|-----------|----------------|
| `SparseConvolution` | kernel/stride/pad/dilation, `subm`, weight/bias, I/O names |
| `Relu` | I/O names |
| `Add` | two feature inputs + output |
| `ScatterDense` / dense | target dense shape |
| `Transpose` / `Reshape` | perm / dims |

4. Fix conventions that differ between export and runtime:
   - spatial order **XYZ (ONNX export)** vs **ZYX (spconv)**
   - weight layout permute into what the CUDA path expects
5. `allocate_weights_on_gpu()` — upload FP16 weights once
6. Prepare grow-only scratch buffers (pair tables, workspaces) to avoid per-frame `cudaMalloc`

After `load()`, the protobuf is no longer needed every frame — only `layers_` + device weights.

**Ops we deliberately do *not* support:** anything else. This is not a general ORT.

### 5.3 `forward()` — interpret the plan each frame

```text
1. Take voxel features + indices from the voxelizer
2. Convert indices XYZ → ZYX if needed
3. Insert them into `reg` under the graph’s first input name
4. for each layer in layers_:
     - look up inputs in `reg` by name
     - run the kernel for layer.type
     - write outputs into `reg` under output name(s)
5. Final dense tensor = lidar_bev (return device pointer)
```

**SparseConvolution dispatch (conceptual):**

```text
lookup input sparse tensor
compute output spatial size
build indice pairs
  - SubM: reuse cached pair tables when topology matches
call spconv (prefer kMaskImplicitGemm; fallback kNative)
optional ReLU
store output SparseState in registry
```

**Add:** `out = a + b` (same active sites).  
**ScatterDense:** sparse features → dense 5D `[N,C,H,W,Z]`.  
**Transpose + Reshape:** collapse Z into channels → BEV `[1, 256, 100, 100]`.

That loop **is** the custom ONNX interpreter.

---

## 6. Checklist: writing one from scratch

1. Dump the ONNX node list once (`op_type`, attributes, I/O names).
2. Define `LayerConfig` covering **only** those ops.
3. Implement `load()`: protobuf → `layers_` + weight upload + layout fixes.
4. Helpers: read initializers, int-array attributes, weight permute.
5. Implement `forward()`: name registry + per-op dispatch.
6. Implement each op by calling an existing library (`spconv`), don’t reinvent GEMM.
7. Validate vs PyTorch (`scripts/export/compare_cpp_parity.py` on `lidar_bev`).

---

## 7. What you are *not* writing

- A general ONNX Runtime  
- Autodiff / training  
- Support for arbitrary ONNX models  
- A TensorRT engine for sparse LiDAR  

You **are** writing a **domain-specific executor** for one exported SCN graph, so the same ONNX can be:

- produced from Python training, and  
- run in C++ with open kernels (no closed `libspconv`).

---

## 8. Where this sits in the full SOPHI stack

```text
┌─────────────────────────────────────────────────────────┐
│  Camera branch                                          │
│    images → (optional frontend) → TRT backbone          │
│            → BEVPool / LSS → TRT vtransform → cam_bev   │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  LiDAR branch  ← THIS DOC                               │
│    points → crop → voxelize                             │
│           → custom ONNX parser → open spconv            │
│           → lidar_bev                                   │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  Fusion + heads (TensorRT)                              │
│    fuser(cam_bev, lidar_bev) → F_env                     │
│    → planning_head / det_head / seg_head                │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Related reading in this repo

- Deploy overview: `navsim/deploy/README.md`
- Full train → eval → export → TRT → parity cookbook: `docs/REPRODUCIBILITY.md` (also `jmlr/REPRODUCIBILITY.md`)
- Licensing note: open SCN path vs CUDA/TensorRT still required for dense engines — see `NOTICE`

---

## 10. Glossary

| Term | Meaning |
|------|---------|
| **SCN** | Sparse Convolution Network (LiDAR backbone) |
| **ONNX** | Portable graph file (ops + weights) |
| **ONNX Runtime** | Generic executor for standard ONNX ops |
| **TensorRT** | NVIDIA optimizer/runtime for dense engines (`.plan`) |
| **Custom interpreter** | Our C++ `load`+`forward` over SCN ONNX |
| **spconv** | Open library providing sparse conv CUDA kernels |
| **SubM** | Submanifold sparse conv (keeps same active sites) |
| **Voxelization** | Points → sparse voxels (features + indices) |
| **lidar_bev** | Dense BEV feature from SCN, input to fuser |
