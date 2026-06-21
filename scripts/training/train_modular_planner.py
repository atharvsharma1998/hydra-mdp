import os
import argparse
from pathlib import Path

# Parse path configuration before other imports to set environment variables
if __name__ == "__main__" or "NUPLAN_MAPS_ROOT" not in os.environ:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", type=str, default=os.environ.get("NAVSIM_EXP_ROOT", "./navsim_workspace"))
    parser.add_argument("--maps-path", type=str, default=None)
    args, _ = parser.parse_known_args()

    navsim_workspace = Path(args.workspace)
    if "NUPLAN_MAPS_ROOT" not in os.environ:
        maps_path = Path(args.maps_path) if args.maps_path else navsim_workspace / "maps"
        os.environ["NUPLAN_MAPS_ROOT"] = str(maps_path)

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List

from navsim.common.dataloader import SceneFilter
from navsim.common.dataclasses import SensorConfig, Scene
from navsim.agents.modular_planner import ModularPlanner
from navsim.agents.modular_features import ModularFeatureBuilder, ModularTargetBuilder
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario


class LazySceneLoader:
    def __init__(
        self,
        data_paths: List[Path],
        original_sensor_path: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
    ):
        self._original_sensor_path = original_sensor_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config
        
        self.token_to_slice = {}
        num_frames = scene_filter.num_frames
        frame_interval = scene_filter.frame_interval

        print("Scanning pickle logs for tokens (lazy scanning)...")
        for data_path in data_paths:
            log_files = list(data_path.iterdir())
            if scene_filter.log_names is not None:
                log_files = [log_file for log_file in log_files if log_file.name.replace(".pkl", "") in scene_filter.log_names]

            if scene_filter.tokens is not None:
                filter_tokens = True
                tokens = set(scene_filter.tokens)
            else:
                filter_tokens = False

            for log_pickle_path in tqdm(log_files, desc=f"Scanning {data_path.name}"):
                if not log_pickle_path.name.endswith(".pkl"):
                    continue
                try:
                    with open(log_pickle_path, "rb") as f:
                        scene_dict_list = pickle.load(f)
                    for i in range(0, len(scene_dict_list), frame_interval):
                        frame_list = scene_dict_list[i : i + num_frames]
                        if len(frame_list) < num_frames:
                            continue
                        if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                            continue
                        token = frame_list[scene_filter.num_history_frames - 1]["token"]
                        if filter_tokens and token not in tokens:
                            continue
                        self.token_to_slice[token] = (log_pickle_path, i, i + num_frames)
                except Exception as e:
                    print(f"Error scanning {log_pickle_path}: {e}")

        self.tokens_list = list(self.token_to_slice.keys())
        self.synthetic_scenes = {}
        self.synthetic_scenes_tokens = set()

    @property
    def tokens(self):
        return self.tokens_list

    def get_scene_from_token(self, token: str) -> Scene:
        assert token in self.token_to_slice
        log_pickle_path, start_idx, end_idx = self.token_to_slice[token]
        with open(log_pickle_path, "rb") as f:
            scene_dict_list = pickle.load(f)
        frame_list = scene_dict_list[start_idx:end_idx]
        return Scene.from_scene_dict_list(
            frame_list,
            self._original_sensor_path,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config,
        )


class ModularPlannerDataset(Dataset):
    def __init__(self, scene_loader: LazySceneLoader, cache_path: Path, maps_path: Path):
        self.scene_loader = scene_loader
        self.cache_path = cache_path
        self.maps_path = maps_path
        
        self.feature_builder = ModularFeatureBuilder()
        self.target_builder = ModularTargetBuilder()
        
    def __len__(self):
        return len(self.scene_loader.tokens)
        
    def __getitem__(self, idx):
        token = self.scene_loader.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        
        # 1. Compute ego status history (status_feature)
        agent_input = scene.get_agent_input()
        status_feature = self.feature_builder._get_status_feature(agent_input)
        
        # 2. Compute BEV grid
        scenario = NavSimScenario(scene, map_root=str(self.maps_path), map_version="nuplan-maps-v1.0")
        ego_pose = scenario.initial_ego_state.rear_axle
        map_api = scenario.map_api
        annotations = scene.frames[-1].annotations
        
        bev_grid = self.feature_builder.compute_bev_grid(annotations, map_api, ego_pose)
        
        # 3. Compute target trajectory
        targets = self.target_builder.compute_targets(scene)
        
        # 4. Load teacher scores from cache if they exist
        cache_file = self.cache_path / f"{token}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    teacher_scores = pickle.load(f)
                gt_scores = {}
                for k, v in teacher_scores.items():
                    gt_scores[k] = torch.tensor(v, dtype=torch.float32)
                targets["gt_scores"] = gt_scores
            except Exception as e:
                pass # If cache is corrupted or incomplete, skip
            
        features = {
            "bev_grid": bev_grid,
            "status_feature": status_feature
        }
        
        return features, targets


def collate_fn(batch):
    features_batch = {}
    targets_batch = {}
    
    # Collate features
    features_batch["bev_grid"] = torch.stack([b[0]["bev_grid"] for b in batch], dim=0)
    features_batch["status_feature"] = torch.stack([b[0]["status_feature"] for b in batch], dim=0)
    
    # Collate targets
    targets_batch["trajectory"] = torch.stack([b[1]["trajectory"] for b in batch], dim=0)
    
    # Check if gt_scores exist in the batch
    if "gt_scores" in batch[0][1]:
        gt_scores_batch = {}
        keys = batch[0][1]["gt_scores"].keys()
        for k in keys:
            gt_scores_batch[k] = torch.stack([b[1]["gt_scores"][k] for b in batch], dim=0)
        targets_batch["gt_scores"] = gt_scores_batch
        
    return features_batch, targets_batch


def main():
    parser = argparse.ArgumentParser(description="Train Modular Planner on NAVSIM")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--workspace",
        type=str,
        default=os.environ.get("NAVSIM_EXP_ROOT", "./navsim_workspace"),
        help="Path to the NAVSIM workspace directory (defaults to NAVSIM_EXP_ROOT environment variable or ./navsim_workspace)",
    )
    parser.add_argument(
        "--maps-path",
        type=str,
        default=None,
        help="Path to NuPlan maps directory (defaults to <workspace>/maps)",
    )
    parser.add_argument(
        "--teacher-cache-path",
        type=str,
        default=None,
        help="Path to precomputed teacher scores cache directory (defaults to <workspace>/teacher_scores_cache)",
    )
    args = parser.parse_args()
    
    # Paths configuration
    navsim_workspace = Path(args.workspace)
    log_paths = [
        navsim_workspace / "trainval_navsim_logs" / "trainval",
        navsim_workspace / "mini_navsim_logs" / "mini"
    ]
    # Filter logs directories to only include directories that actually exist to avoid errors
    log_paths = [p for p in log_paths if p.exists()]
    if len(log_paths) == 0:
        log_paths = [navsim_workspace]
        print(f"Warning: Default log paths not found. Using workspace path directly: {navsim_workspace}")

    maps_path = Path(args.maps_path) if args.maps_path else navsim_workspace / "maps"
    cache_path = Path(args.teacher_cache_path) if args.teacher_cache_path else navsim_workspace / "teacher_scores_cache"
    checkpoint_dir = navsim_workspace / "checkpoints"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Initializing ModularPlanner...")
    model = ModularPlanner(lr=args.lr)
    model.to(device)
    
    # Dataloader setup
    print("Loading NAVSIM splits...")
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True)
    scene_loader = LazySceneLoader(
        original_sensor_path=navsim_workspace,
        data_paths=log_paths,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors()
    )
    
    # Shuffle splits randomly for train / val (80% train, 20% val)
    import random
    random.seed(42)
    all_tokens = scene_loader.tokens
    random.shuffle(all_tokens)
    
    num_train = int(len(all_tokens) * 0.8)
    train_tokens = all_tokens[:num_train]
    val_tokens = all_tokens[num_train:]
    
    # Segregate train tokens into distilled and imitation-only
    train_tokens_distill = [t for t in train_tokens if (cache_path / f"{t}.pkl").exists()]
    train_tokens_imi = [t for t in train_tokens if not (cache_path / f"{t}.pkl").exists()]
    
    print(f"Train dataset split: {len(train_tokens_distill)} distilled scenes, {len(train_tokens_imi)} imitation-only scenes.")
    assert len(train_tokens_distill) > 0, "No distilled training scenes found!"
    
    # Create copies of scene loader for splits and filter their dictionaries
    import copy
    train_loader_distill_obj = copy.copy(scene_loader)
    train_loader_distill_obj.token_to_slice = {t: scene_loader.token_to_slice[t] for t in train_tokens_distill if t in scene_loader.token_to_slice}
    train_loader_distill_obj.tokens_list = list(train_loader_distill_obj.token_to_slice.keys())
    train_loader_distill_obj.synthetic_scenes = {}
    train_loader_distill_obj.synthetic_scenes_tokens = set()

    train_loader_imi_obj = copy.copy(scene_loader)
    train_loader_imi_obj.token_to_slice = {t: scene_loader.token_to_slice[t] for t in train_tokens_imi if t in scene_loader.token_to_slice}
    train_loader_imi_obj.tokens_list = list(train_loader_imi_obj.token_to_slice.keys())
    train_loader_imi_obj.synthetic_scenes = {}
    train_loader_imi_obj.synthetic_scenes_tokens = set()
    
    val_loader_obj = copy.copy(scene_loader)
    val_loader_obj.token_to_slice = {t: scene_loader.token_to_slice[t] for t in val_tokens if t in scene_loader.token_to_slice}
    val_loader_obj.tokens_list = list(val_loader_obj.token_to_slice.keys())
    val_loader_obj.synthetic_scenes = {}
    val_loader_obj.synthetic_scenes_tokens = set()
    
    print(f"Validation dataset summary: {len(val_tokens)} val scenes.")
    
    train_dataset_distill = ModularPlannerDataset(train_loader_distill_obj, cache_path, maps_path)
    train_dataset_imi = ModularPlannerDataset(train_loader_imi_obj, cache_path, maps_path)
    val_dataset = ModularPlannerDataset(val_loader_obj, cache_path, maps_path)
    
    train_loader_distill = DataLoader(
        train_dataset_distill,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    train_loader_imi = DataLoader(
        train_dataset_imi,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = model.get_optimizers()
    best_val_loss = float("inf")
    
    from itertools import cycle
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Training loop
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0
        train_imi_loss = 0.0
        train_dist_loss = 0.0
        
        distill_iter = cycle(train_loader_distill)
        
        for step, (features_imi, targets_imi) in enumerate(tqdm(train_loader_imi, desc="Training Steps")):
            # 1. Imitation batch step
            features_imi = {k: v.to(device) for k, v in features_imi.items()}
            targets_imi["trajectory"] = targets_imi["trajectory"].to(device)
            
            optimizer.zero_grad()
            preds_imi = model(features_imi)
            losses_imi = model.planning_head.loss(preds_imi, targets_imi)
            loss_imi = losses_imi["loss_total"]
            
            loss_imi.backward()
            optimizer.step()
            
            train_loss += loss_imi.item()
            train_imi_loss += losses_imi["loss_imitation"].item()
            
            # 2. Distillation batch step (interleaved on every step)
            try:
                features_dist, targets_dist = next(distill_iter)
                features_dist = {k: v.to(device) for k, v in features_dist.items()}
                targets_dist["trajectory"] = targets_dist["trajectory"].to(device)
                targets_dist["gt_scores"] = {k: v.to(device) for k, v in targets_dist["gt_scores"].items()}
                
                optimizer.zero_grad()
                preds_dist = model(features_dist)
                losses_dist = model.planning_head.loss(preds_dist, targets_dist)
                loss_dist = losses_dist["loss_total"]
                
                loss_dist.backward()
                optimizer.step()
                
                train_loss += loss_dist.item()
                train_imi_loss += losses_dist["loss_imitation"].item()
                if "loss_distill" in losses_dist:
                    train_dist_loss += losses_dist["loss_distill"].item()
            except Exception as e:
                # Fallback / safe skip if cycle has issues
                pass
                
        # Average total steps is 2 * num_imi_steps
        avg_train_loss = train_loss / (2.0 * len(train_loader_imi))
        avg_train_imi = train_imi_loss / (2.0 * len(train_loader_imi))
        avg_train_dist = train_dist_loss / len(train_loader_imi)
        
        print(f"Train Loss: {avg_train_loss:.4f} (Imitation: {avg_train_imi:.4f}, Distill: {avg_train_dist:.4f})")
        
        # Validation loop
        torch.cuda.empty_cache()
        model.eval()
        val_loss = 0.0
        val_imi_loss = 0.0
        val_dist_loss = 0.0
        val_dist_count = 0
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validation Steps"):
                features = {k: v.to(device) for k, v in features.items()}
                targets["trajectory"] = targets["trajectory"].to(device)
                if "gt_scores" in targets:
                    targets["gt_scores"] = {k: v.to(device) for k, v in targets["gt_scores"].items()}
                    val_dist_count += 1
                    
                preds = model(features)
                losses = model.planning_head.loss(preds, targets)
                loss = losses["loss_total"]
                
                val_loss += loss.item()
                val_imi_loss += losses["loss_imitation"].item()
                if "loss_distill" in losses:
                    val_dist_loss += losses["loss_distill"].item()
                    
        avg_val_loss = val_loss / len(val_loader)
        avg_val_imi = val_imi_loss / len(val_loader)
        avg_val_dist = val_dist_loss / val_dist_count if val_dist_count > 0 else 0.0
        
        print(f"Val Loss: {avg_val_loss:.4f} (Imitation: {avg_val_imi:.4f}, Distill: {avg_val_dist:.4f})")
        
        # Save checkpoints
        state_dict = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": avg_val_loss
        }
        
        torch.save(state_dict, checkpoint_dir / "latest.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(state_dict, checkpoint_dir / "best.pth")
            print("New best validation loss! Saved best.pth.")
            
    print("\n=== Training Completed Successfully ===")


if __name__ == "__main__":
    main()
