"""Train / overfit the GTRS-BEVFusion agent on NAVSIM (sensor-driven).

Local validation: overfit a small subset of mini scenes to confirm the full
sensor -> F_env -> {planning, detection, BEV-seg} graph learns (losses drop).
Then scale to navtrain on cloud.

Example (local overfit smoke):
    python scripts/training/train_gtrs_bevfusion.py \
        --workspace ./navsim_workspace \
        --sensor-blobs-path ./navsim_workspace/openscene-v1.1/sensor_blobs/mini \
        --num-scenes 32 --epochs 60 --batch-size 2
"""
import os
import argparse
from pathlib import Path

if __name__ == "__main__" or "NUPLAN_MAPS_ROOT" not in os.environ:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", type=str, default=os.environ.get("NAVSIM_EXP_ROOT", "./navsim_workspace"))
    parser.add_argument("--maps-path", type=str, default=None)
    _args, _ = parser.parse_known_args()
    _ws = Path(_args.workspace)
    if "NUPLAN_MAPS_ROOT" not in os.environ:
        os.environ["NUPLAN_MAPS_ROOT"] = str(Path(_args.maps_path) if _args.maps_path else _ws / "maps")

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from navsim.common.dataclasses import SensorConfig, Scene, SceneFilter
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.agents.gtrs_bevfusion.bevfusion_features import BEVFusionFeatureBuilder
from navsim.agents.gtrs_bevfusion.bevfusion_collate import bevfusion_collate
from navsim.agents.gtrs_bevfusion.bevfusion_model import GTRSBevfusionModel
from navsim.agents.gtrs_bevfusion.bevfusion_loss import gtrs_bevfusion_loss
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserTargetBuilder
# train_modular_planner.py sits in this same dir (on sys.path[0] when run as a
# script). Its argparse header is skipped because NUPLAN_MAPS_ROOT is set above.
from train_modular_planner import LazySceneLoader


class BevfusionDataset(Dataset):
    def __init__(self, scene_loader, cache_path, config: GTRSBevfusionConfig, trajectory_sampling):
        self.scene_loader = scene_loader
        self.cache_path = cache_path
        self.feature_builder = BEVFusionFeatureBuilder(config)
        self.target_builder = TransfuserTargetBuilder(
            trajectory_sampling=trajectory_sampling, config=TransfuserConfig()
        )

    def __len__(self):
        return len(self.scene_loader.tokens)

    def __getitem__(self, idx):
        token = self.scene_loader.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        features = self.feature_builder.compute_features(agent_input)
        targets = self.target_builder.compute_targets(scene)

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
    parser.add_argument("--teacher-cache-path", type=str, default=None)
    parser.add_argument("--num-scenes", type=int, default=32, help="overfit subset size (0 = all)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    # ---- production / cloud flags (defaults preserve the local overfit flow) ----
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm (0 = off)")
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1, help="also save epoch{N}.pth every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume (model+opt+epoch)")
    parser.add_argument("--amp", action="store_true", help="mixed-precision training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="gtrs_bevfusion")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ws = Path(args.workspace)
    maps_path = Path(args.maps_path) if args.maps_path else ws / "maps"
    cache_path = Path(args.teacher_cache_path) if args.teacher_cache_path else ws / "teacher_scores_cache"
    sensor_blobs_path = Path(args.sensor_blobs_path) if args.sensor_blobs_path else ws / "mini_sensor_blobs" / "mini"
    ckpt_dir = ws / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | sensor blobs: {sensor_blobs_path}")

    config = GTRSBevfusionConfig()
    trajectory_sampling = __import__("nuplan.planning.simulation.trajectory.trajectory_sampling",
                                     fromlist=["TrajectorySampling"]).TrajectorySampling(
        time_horizon=4, interval_length=0.5)

    log_paths = [ws / "mini_navsim_logs" / "mini", ws / "trainval_navsim_logs" / "trainval"]
    log_paths = [p for p in log_paths if p.exists()]

    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True)
    scene_loader = LazySceneLoader(
        original_sensor_path=sensor_blobs_path,
        data_paths=log_paths,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )

    # keep only tokens whose log sensor folder is actually downloaded
    tokens = [t for t in scene_loader.tokens
              if _log_has_sensors(sensor_blobs_path, scene_loader.token_to_slice[t][0].name.replace(".pkl", ""))]
    print(f"Tokens with downloaded sensors: {len(tokens)} / {len(scene_loader.tokens)}")
    if args.num_scenes > 0:
        tokens = tokens[: args.num_scenes]
    scene_loader.token_to_slice = {t: scene_loader.token_to_slice[t] for t in tokens}
    scene_loader.tokens_list = tokens
    print(f"Overfit subset: {len(tokens)} scenes")

    dataset = BevfusionDataset(scene_loader, cache_path, config, trajectory_sampling)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=bevfusion_collate, pin_memory=True)

    model = GTRSBevfusionModel(config, num_poses=trajectory_sampling.num_poses).to(device).train()
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
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_fh = open(args.log_file, "a") if args.log_file else None

    def log(line: str):
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    def to_dev(feats, tgts):
        feats = {k: ([p.to(device) for p in v] if k == "lidar" else v.to(device)) for k, v in feats.items()}
        tgts = {k: ({sk: sv.to(device) for sk, sv in v.items()} if k == "gt_scores" else v.to(device))
                for k, v in tgts.items()}
        return feats, tgts

    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        agg = {}
        for feats, tgts in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            feats, tgts = to_dev(feats, tgts)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds = model(feats)
                losses = gtrs_bevfusion_loss(tgts, preds, config, model.planning_head)
            scaler.scale(losses["loss_total"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        msg = " | ".join(f"{k}={agg[k]/len(loader):.4f}" for k in sorted(agg))
        log(f"[epoch {epoch+1}/{args.epochs}] lr={cur_lr:.2e} | {msg}")

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if args.amp else None,
            "config": vars(args),
        }
        torch.save(state, ckpt_dir / f"{args.run_name}_latest.pth")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(state, ckpt_dir / f"{args.run_name}_epoch{epoch+1}.pth")

    if log_fh:
        log_fh.close()
    print("=== training run complete ===")


if __name__ == "__main__":
    main()
