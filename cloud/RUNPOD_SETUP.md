# RunPod setup — GTRS-BEVFusion / Hydra-MDP training

Reference for the **actual** directory layout on our RunPod network volume.
Use this whenever you spin up a new pod so you don't point at the wrong paths.

---

## Workspace layout

```
/workspace/
├── hydra-mdp/                          # GitHub repo root (NAVSIM_REPO)
│   ├── navsim/                         # Python package: `import navsim`
│   ├── scripts/training/               # train_gtrs_bevfusion.py
│   ├── cloud/                          # preflight, download, train scripts
│   ├── traj_final/8192.npy             # planning vocabulary (do NOT rebuild)
│   └── deploy/                         # C++ / TensorRT (post-training)
│
├── bev-inference/                      # NVIDIA Lidar_AI_Solution fork
│   └── CUDA-BEVFusion/
│       └── bevfusion/                  # mmdet3d Python fork (BEVFUSION_ROOT) ✅
│           ├── setup.py
│           └── mmdet3d/
│
├── bevfusion/                          # ⚠️ EMPTY STUB — do NOT use for pip install
│
├── nuplan-devkit/
├── venv/                               # Python 3.8 + torch 1.10.2 + spconv
└── navsim_workspace/                   # data + checkpoints (NAVSIM_WS)
    ├── maps/
    ├── trainval_navsim_logs/trainval/
    ├── trainval_sensor_blobs/trainval/
    ├── navtrain_8192.pkl               # teacher scores (download before train)
    └── checkpoints/
```

**Common mistake:** `/workspace/bevfusion` is not the mmdet3d package. The installable
fork lives at `/workspace/bev-inference/CUDA-BEVFusion/bevfusion`.

---

## One-time env (every new shell / pod)

```bash
export NAVSIM_REPO=/workspace/hydra-mdp
export NAVSIM_WS=/workspace/navsim_workspace
export BEVFUSION_ROOT=/workspace/bev-inference/CUDA-BEVFusion/bevfusion
export NUPLAN_SRC=/workspace/nuplan-devkit
export PYTHON=/workspace/venv/bin/python
export TEACHER_PKL=/workspace/navsim_workspace/navtrain_8192.pkl

source $NAVSIM_REPO/cloud/env.sh
```

`cloud/env.sh` defaults `BEVFUSION_ROOT` to the path above; override only if your
layout differs.

---

## First-time Python install (fresh venv or new pod)

Only needed if `import mmdet3d` fails in preflight.

```bash
export BEVFUSION_ROOT=/workspace/bev-inference/CUDA-BEVFusion/bevfusion
export NAVSIM_REPO=/workspace/hydra-mdp
export PYTHON=/workspace/venv/bin/python

# mmdet3d fork (compiles CUDA ops — ~5–15 min first time)
$PYTHON -m pip install -e "$BEVFUSION_ROOT" --no-deps

# navsim / hydra-mdp
$PYTHON -m pip install -e "$NAVSIM_REPO" --no-deps

# sanity
$PYTHON -c "import mmdet3d, navsim, spconv, mmcv; print('imports OK')"
```

If the venv doesn't exist yet, run `bash $NAVSIM_REPO/cloud/setup_env.sh` first
(set `VENV=/workspace/venv` and the same `BEVFUSION_ROOT` / `NAVSIM_REPO` exports).

---

## Preflight (run before every training launch)

```bash
source $NAVSIM_REPO/cloud/env.sh
bash $NAVSIM_REPO/cloud/00_preflight.sh
```

Must pass:
- `mmdet3d` import OK
- backbone builds from scratch
- `traj_final/8192.npy` present
- maps + trainval logs + sensor blobs present

---

## Data (if not already on volume)

```bash
bash $NAVSIM_REPO/cloud/01_download_data.sh
```

Resumable. Needs ~425 GB for full navtrain sensor blobs.

---

## Teacher scores (required for planner distillation)

**Do not rebuild the vocab.** Reuse `traj_final/8192.npy`.

Download GTRS precomputed scores (~30 GB):

```bash
cd /workspace/navsim_workspace
wget -c https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_8192.pkl
export TEACHER_PKL=/workspace/navsim_workspace/navtrain_8192.pkl
```

The trainer loads this via `--teacher-pkl` (see `cloud/03_train.sh`).
A partial per-token cache (`teacher_scores_cache/` with hundreds of shards) is **not**
enough for full navtrain — use the big pkl.

Alternative (slow, days of CPU): `bash $NAVSIM_REPO/cloud/02_precompute_teacher_scores.sh`

---

## Launch training

```bash
source $NAVSIM_REPO/cloud/env.sh
export TEACHER_PKL=/workspace/navsim_workspace/navtrain_8192.pkl

export RUN_NAME=gtrs_bevfusion_navtrain_v1
export EPOCHS=20          # HydraMDP paper; script default is 30
export BATCH=4            # per-GPU; reduce to 2 if OOM
export NGPU=1             # or 2/4/8 for multi-GPU
export AMP=1              # safe with fp32 focal-loss fix in bevfusion_loss.py
export WORKERS=10

bash $NAVSIM_REPO/cloud/03_train.sh
tail -f $NAVSIM_WS/logs/${RUN_NAME}_*.log
```

Watch the first ~10 epochs for `loss_total=nan` (should not happen after the
fp32 CenterPoint focal-loss fix). Training auto-resumes from
`checkpoints/${RUN_NAME}_latest.pth` if the pod restarts.

---

## Verify latest code is on the pod

After `git pull` on hydra-mdp:

```bash
grep -n "MUST be computed in fp32" \
  $NAVSIM_REPO/navsim/agents/gtrs_bevfusion/bevfusion_loss.py
grep -n "Learned upsampling decoder" \
  $NAVSIM_REPO/navsim/agents/gtrs_bevfusion/bevfusion_model.py
```

Both greps should match (NaN fix + seg decoder + CenterPoint head).

---

## Option C — Prebuilt Docker image (skip manual pip install)

Build once on your machine, push to Docker Hub, use as the RunPod pod image.
The stack (torch, mmcv, mmdet3d, nuplan, navsim) is baked under `/opt/` so the
network volume at `/workspace` only needs data.

```bash
# On your machine (CUDA-BEVFusion repo):
export HYDRA_MDP_REPO=https://github.com/<you>/hydra-mdp.git
bash /path/to/CUDA-BEVFusion/cloud/build_train_image.sh <you>/gtrs-bevfusion-train:cu113
docker push <you>/gtrs-bevfusion-train:cu113
```

RunPod: select that image, attach volume at `/workspace`, then:

```bash
export TEACHER_PKL=/workspace/navsim_workspace/navtrain_8192.pkl
bash $NAVSIM_REPO/cloud/00_preflight.sh
bash $NAVSIM_REPO/cloud/03_train.sh
```

Full details: `bev-inference/CUDA-BEVFusion/cloud/README.md` (in the
Lidar_AI_Solution repo).

---

## What we do NOT rebuild for each run

| Asset | Rebuild? |
|---|---|
| `traj_final/8192.npy` | **No** — same GTRS vocabulary |
| `navtrain_8192.pkl` | **No** — download once |
| Maps / sensor blobs | Only if missing |
| mmdet3d pip install | Only if venv is fresh |
| Seg ONNX/TRT engine | **Yes**, after training (seg head architecture changed) |
