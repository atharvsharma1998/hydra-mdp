"""Train / overfit the GTRS-BEVFusion agent on NAVSIM (sensor-driven).

Local validation: overfit a small subset of mini scenes to confirm the full
sensor -> F_env -> {planning, detection, BEV-seg} graph learns (losses drop).
Then scale to navtrain on cloud.

Example (local overfit smoke):
    python scripts/training/train_gtrs_bevfusion.py \
        --workspace ./navsim_workspace \
        --sensor-blobs-path ./navsim_workspace/openscene-v1.1/sensor_blobs/mini \
        --num-scenes 32 --epochs 60 --batch-size 2

Teacher scores (PDM distillation): two options
  * --teacher-pkl  : GTRS-style single big pickle {token: {metric: (8192,)}},
                     loaded ONCE in the main process and indexed by token in the
                     training loop (DataLoader workers stay light). Preferred.
  * --teacher-cache-path : legacy dir of per-token <token>.pkl files (read in
                     the Dataset workers).
"""
import os
import argparse
import warnings
from pathlib import Path

# nuplan's map code spams "invalid value encountered in cast" (NaN->int) once per
# map query, which floods stdout/logs over 100k scenes. It's harmless; silence it.
warnings.filterwarnings("ignore", message="invalid value encountered in cast")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

if __name__ == "__main__" or "NUPLAN_MAPS_ROOT" not in os.environ:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", type=str, default=os.environ.get("NAVSIM_EXP_ROOT", "./navsim_workspace"))
    parser.add_argument("--maps-path", type=str, default=None)
    _args, _ = parser.parse_known_args()
    _ws = Path(_args.workspace)
    if "NUPLAN_MAPS_ROOT" not in os.environ:
        os.environ["NUPLAN_MAPS_ROOT"] = str(Path(_args.maps_path) if _args.maps_path else _ws / "maps")

import pickle
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# PDM metric heads that receive distillation targets (must exist in both the
# planning head preds and the teacher score dict). 'imi' is computed separately.
DISTILL_METRICS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
]

from navsim.common.dataclasses import SensorConfig, Scene, SceneFilter
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.agents.gtrs_bevfusion.bevfusion_features import BEVFusionFeatureBuilder
from navsim.agents.gtrs_bevfusion.bevfusion_collate import bevfusion_collate
from navsim.agents.gtrs_bevfusion.bevfusion_model import GTRSBevfusionModel
from navsim.agents.gtrs_bevfusion.bevfusion_loss import gtrs_bevfusion_loss
from navsim.agents.gtrs_bevfusion.bevfusion_target import BEVFusionTargetBuilder
# train_modular_planner.py sits in this same dir (on sys.path[0] when run as a
# script). Its argparse header is skipped because NUPLAN_MAPS_ROOT is set above.
from train_modular_planner import LazySceneLoader


class BevfusionDataset(Dataset):
    def __init__(self, scene_loader, cache_path, config: GTRSBevfusionConfig, trajectory_sampling,
                 big_pkl_mode: bool = False):
        self.scene_loader = scene_loader
        self.cache_path = cache_path
        self.big_pkl_mode = big_pkl_mode  # if True, teacher scores are injected in the train loop
        self.feature_builder = BEVFusionFeatureBuilder(config)
        self.target_builder = BEVFusionTargetBuilder(trajectory_sampling=trajectory_sampling)

    def __len__(self):
        return len(self.scene_loader.tokens)

    def __getitem__(self, idx):
        token = self.scene_loader.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        features = self.feature_builder.compute_features(agent_input)
        targets = self.target_builder.compute_targets(scene)
        targets["token"] = token  # for teacher-score lookup (legacy cache or big pkl)

        # Legacy per-token cache path. In big-pkl mode the scores are looked up
        # once in the main process (avoids holding the ~30 GB dict per worker).
        if not self.big_pkl_mode:
            cache_file = self.cache_path / f"{token}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        teacher_scores = pickle.load(f)
                    targets["gt_scores"] = {k: torch.tensor(v, dtype=torch.float32) for k, v in teacher_scores.items()}
                except Exception:
                    pass
        return features, targets


def _log_has_sensors(sensor_blobs_path: Path, log_name: str) -> bool:
    """Require both camera (CAM_F0) and LiDAR (MergedPointCloud) for the log."""
    base = sensor_blobs_path / log_name
    return (base / "CAM_F0").is_dir() and (base / "MergedPointCloud").is_dir()


def main():
    parser = argparse.ArgumentParser(description="Train/overfit GTRS-BEVFusion on NAVSIM")
    parser.add_argument("--workspace", type=str, default=os.environ.get("NAVSIM_EXP_ROOT", "./navsim_workspace"))
    parser.add_argument("--maps-path", type=str, default=None)
    parser.add_argument("--sensor-blobs-path", type=str, default=None,
                        help="dir containing <log>/CAM_*/*.jpg (e.g. .../sensor_blobs/mini)")
    parser.add_argument("--scene-filter-yaml", type=str,
                        default=str(Path(__file__).resolve().parents[2] /
                                    "navsim/planning/script/config/common/train_test_split"
                                    "/scene_filter/navtrain.yaml"),
                        help="navsim SceneFilter hydra yaml (default: official navtrain split). "
                             "Pass '' to fall back to a generic frame filter.")
    parser.add_argument("--teacher-cache-path", type=str, default=None,
                        help="legacy: dir of per-token <token>.pkl score files")
    parser.add_argument("--teacher-pkl", type=str, default=None,
                        help="GTRS-style single big pickle (e.g. navtrain_8192.pkl) "
                             "mapping token -> {metric: (8192,) scores}; loaded once in main process")
    parser.add_argument("--num-scenes", type=int, default=32, help="overfit subset size (0 = all)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="batches each worker prefetches (deeper queue rides through dataloader stalls)")
    # ---- production / cloud flags (defaults preserve the local overfit flow) ----
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm (0 = off)")
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0,
                        help="also save epoch{N}.pth every N epochs (0 = off; only _latest + _best kept)")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume (model+opt+epoch)")
    parser.add_argument("--amp", action="store_true", help="mixed-precision training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--tb-dir", type=str, default=None,
                        help="TensorBoard log dir (default: <workspace>/tb/<run_name>). "
                             "Pass '' to disable.")
    parser.add_argument("--run-name", type=str, default="gtrs_bevfusion")
    parser.add_argument("--no-find-unused-params", dest="find_unused_params",
                        action="store_false", default=True,
                        help="disable DDP find_unused_parameters (faster, but only safe if every "
                             "param gets a grad every step; head/vocab selection usually doesn't)")
    parser.add_argument("--no-sync-bn", dest="sync_bn", action="store_false", default=True,
                        help="disable SyncBatchNorm conversion under DDP (BEVFusion backbone has "
                             "many BN layers; syncing across GPUs helps with small per-GPU batch)")
    args = parser.parse_args()

    # ---- distributed (launched via torchrun); falls back to single-GPU ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1
    is_main = rank == 0
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)

    ws = Path(args.workspace)
    maps_path = Path(args.maps_path) if args.maps_path else ws / "maps"
    cache_path = Path(args.teacher_cache_path) if args.teacher_cache_path else ws / "teacher_scores_cache"
    sensor_blobs_path = Path(args.sensor_blobs_path) if args.sensor_blobs_path else ws / "mini_sensor_blobs" / "mini"
    ckpt_dir = ws / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Device: {device} | world_size: {world_size} | sensor blobs: {sensor_blobs_path}")

    # GTRS-style teacher scores: load the single big pickle ONCE per process.
    # NOTE: under DDP each rank loads its own copy (~30 GB), so a 4-GPU node
    # needs ~4x that in system RAM. Token lookup happens in the train loop so
    # DataLoader workers never receive a copy.
    teacher_full = None
    if args.teacher_pkl:
        if is_main:
            print(f"Loading teacher scores (big pkl): {args.teacher_pkl} ...", flush=True)
        with open(args.teacher_pkl, "rb") as f:
            teacher_full = pickle.load(f)
        if is_main:
            print(f"  loaded teacher scores for {len(teacher_full)} tokens", flush=True)

    config = GTRSBevfusionConfig()
    trajectory_sampling = __import__("nuplan.planning.simulation.trajectory.trajectory_sampling",
                                     fromlist=["TrajectorySampling"]).TrajectorySampling(
        time_horizon=4, interval_length=0.5)

    log_paths = [ws / "mini_navsim_logs" / "mini", ws / "trainval_navsim_logs" / "trainval"]
    log_paths = [p for p in log_paths if p.exists()]

    # Use NAVSIM's official navtrain SceneFilter (explicit log_names + tokens) so
    # we enumerate EXACTLY the tokens GTRS trains on (and that the teacher pkl
    # covers). A generic frame filter scans all of trainval and mostly produces
    # tokens that aren't in navtrain -> tiny overlap with the teacher scores.
    if args.scene_filter_yaml and os.path.exists(args.scene_filter_yaml):
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        scene_filter = instantiate(OmegaConf.load(args.scene_filter_yaml))
        if is_main:
            print(f"SceneFilter: navtrain yaml "
                  f"({len(scene_filter.log_names or [])} logs, "
                  f"{len(scene_filter.tokens or [])} tokens) <- {args.scene_filter_yaml}")
    else:
        scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True)
        if is_main:
            print("SceneFilter: generic (no navtrain yaml found)")
    scene_loader = LazySceneLoader(
        original_sensor_path=sensor_blobs_path,
        data_paths=log_paths,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )

    # keep only tokens whose log sensor folder is actually downloaded
    tokens = [t for t in scene_loader.tokens
              if _log_has_sensors(sensor_blobs_path, scene_loader.token_to_slice[t][0].name.replace(".pkl", ""))]
    if is_main:
        print(f"Tokens with downloaded sensors: {len(tokens)} / {len(scene_loader.tokens)}")

    # align to tokens that actually have teacher scores. In big-pkl mode this is an
    # in-memory dict lookup; in per-token cache mode (low-RAM, local) we require the
    # <token>.pkl shard to exist. Without this filter, cache-mode batches mix samples
    # with/without gt_scores and the collate breaks (it keys off sample 0).
    if teacher_full is not None:
        before = len(tokens)
        tokens = [t for t in tokens if t in teacher_full]
        if is_main:
            print(f"Tokens with teacher scores (big pkl): {len(tokens)} / {before}")
    elif cache_path.exists():
        before = len(tokens)
        tokens = [t for t in tokens if (cache_path / f"{t}.pkl").exists()]
        if is_main:
            print(f"Tokens with teacher cache shard: {len(tokens)} / {before} ({cache_path})")

    if args.num_scenes > 0:
        tokens = tokens[: args.num_scenes]
    scene_loader.token_to_slice = {t: scene_loader.token_to_slice[t] for t in tokens}
    scene_loader.tokens_list = tokens
    if is_main:
        print(f"Training scenes: {len(tokens)}")

    dataset = BevfusionDataset(scene_loader, cache_path, config, trajectory_sampling,
                               big_pkl_mode=teacher_full is not None)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    # persistent_workers keeps the (expensive) worker processes + their warm
    # nuplan map caches alive across epochs; prefetch_factor deepens the queue so
    # the GPU rides through the bursty per-sample map-query/decode stalls.
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=(sampler is None),
                         sampler=sampler, num_workers=args.num_workers,
                         collate_fn=bevfusion_collate, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    model = GTRSBevfusionModel(config, num_poses=trajectory_sampling.num_poses).to(device).train()
    core_model = model  # unwrapped handle for .planning_head / state_dict / save
    if ddp:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.find_unused_params)
        core_model = model.module
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine schedule with linear warmup (per-epoch stepping)
    scheduler = None
    if args.lr_schedule == "cosine":
        import math
        warmup = max(args.warmup_epochs, 0)

        def lr_lambda(ep):
            if warmup and ep < warmup:
                return float(ep + 1) / float(warmup)
            progress = (ep - warmup) / max(1, args.epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        core_model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        if is_main:
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_fh = open(args.log_file, "a") if (args.log_file and is_main) else None

    # TensorBoard (rank-0 only). The torch 1.10 SummaryWriter trips over modern
    # setuptools (`distutils.version` not auto-loaded); pre-importing it fixes that.
    # Guarded so a missing/broken tensorboard never kills a training run.
    tb_writer = None
    if is_main and args.tb_dir != "":
        tb_dir = args.tb_dir or str(ws / "tb" / args.run_name)
        try:
            import distutils.version  # noqa: F401  (must precede tensorboard import)
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(tb_dir)
            print(f"TensorBoard logging to {tb_dir}  (tensorboard --logdir {ws / 'tb'})", flush=True)
        except Exception as e:  # pragma: no cover - optional dep
            print(f"[warn] TensorBoard disabled ({e})", flush=True)

    def log(line: str):
        if not is_main:
            return
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    def to_dev(feats, tgts):
        feats = {k: ([p.to(device) for p in v] if k == "lidar" else v.to(device)) for k, v in feats.items()}
        tgts = {k: ({sk: sv.to(device) for sk, sv in v.items()} if k == "gt_scores" else v.to(device))
                for k, v in tgts.items()}
        return feats, tgts

    def build_gt_scores(tokens):
        """Index the big teacher pkl for a batch of tokens -> {metric: (B, 8192)}."""
        gt = {}
        for m in DISTILL_METRICS:
            arr = np.stack([np.asarray(teacher_full[t][m], dtype=np.float32) for t in tokens], axis=0)
            gt[m] = torch.from_numpy(arr).to(device)
        return gt

    global_step = start_epoch * len(loader)
    best_loss = float("inf")  # track lowest mean epoch loss for the "best" checkpoint
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        if sampler is not None:
            sampler.set_epoch(epoch)  # reshuffle differently each epoch across ranks
        agg = {}
        iterable = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}") if is_main else loader
        for feats, tgts in iterable:
            batch_tokens = tgts.pop("token", None)  # list[str]; not a tensor
            feats, tgts = to_dev(feats, tgts)
            if teacher_full is not None and batch_tokens is not None:
                tgts["gt_scores"] = build_gt_scores(batch_tokens)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds = model(feats)
                losses = gtrs_bevfusion_loss(tgts, preds, config, core_model.planning_head)
            scaler.scale(losses["loss_total"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            if tb_writer is not None:
                tb_writer.add_scalar("step/loss_total", float(losses["loss_total"]), global_step)
            global_step += 1
        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        msg = " | ".join(f"{k}={agg[k]/len(loader):.4f}" for k in sorted(agg))
        log(f"[epoch {epoch+1}/{args.epochs}] lr={cur_lr:.2e} | {msg}")
        if tb_writer is not None:
            for k in agg:
                tb_writer.add_scalar(f"epoch/{k}", agg[k] / len(loader), epoch + 1)
            tb_writer.add_scalar("epoch/lr", cur_lr, epoch + 1)
            tb_writer.flush()

        if is_main:
            state = {
                "epoch": epoch,
                "state_dict": core_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if args.amp else None,
                "config": vars(args),
            }
            torch.save(state, ckpt_dir / f"{args.run_name}_latest.pth")
            # keep only the BEST checkpoint (lowest mean epoch loss); overwrites a single
            # file so disk stays bounded. Intermediate epoch{N}.pth snapshots are only
            # written when --save-every > 0 is explicitly requested (default: off).
            epoch_loss = agg["loss_total"] / len(loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(state, ckpt_dir / f"{args.run_name}_best.pth")
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(state, ckpt_dir / f"{args.run_name}_epoch{epoch+1}.pth")
        if ddp:
            dist.barrier()  # keep ranks in lock-step across the epoch boundary

    if tb_writer is not None:
        tb_writer.close()
    if log_fh:
        log_fh.close()
    if ddp:
        dist.destroy_process_group()
    if is_main:
        print("=== training run complete ===")


if __name__ == "__main__":
    main()
