# Training and PDM evaluation

Deploy-only users: stop at [`QUICKSTART.md`](../QUICKSTART.md) — you do **not**
need the full NAVSIM dataset. The repo ships `deploy/example-data/` for C++ inference.

This page is for training and official `navtest` scoring.

## Data

Download NAVSIM logs/sensors and nuPlan maps using scripts under `download/`
(see also [`install.md`](install.md)). Example layout:

```
$WORKSPACE/
  maps/
  mini_navsim_logs/mini/          # or trainval / test
  mini_sensor_blobs/mini/
  checkpoints/
  metric_cache/
```

```bash
export NUPLAN_MAPS_ROOT=$WORKSPACE/maps
export PYTHONPATH=$REPO:$REPO/scripts/training:$PYTHONPATH
```

## Metric cache (before PDM scoring)

```bash
cd $REPO
$PYTHON navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  worker=single_machine_thread_pool \
  worker.use_process_pool=True \
  worker.max_workers=$(nproc) \
  force_feature_computation=False \
  cache.cache_path=$WORKSPACE/metric_cache
```

## Train

See `scripts/training/train_gtrs_bevfusion.py` and `cloud/03_train.sh` for multi-GPU.

## Official PDM score (`navtest`)

```bash
cd $REPO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PYTHONWARNINGS=ignore $PYTHON navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=gtrs_bevfusion_agent \
  agent.checkpoint_path=$WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth \
  worker=single_machine_thread_pool \
  worker.use_process_pool=True \
  worker.max_workers=10 \
  experiment_name=gtrs_bevfusion_navtest \
  metric_cache_path=$WORKSPACE/metric_cache
```

Agent config: `navsim/planning/script/config/common/agent/gtrs_bevfusion_agent.yaml`.

Published checkpoint scores **PDM 0.7925** on 12,149 `navtest` scenarios
(see benchmark table in the root [`README.md`](../README.md)).
