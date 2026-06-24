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
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")

from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array


# Global variables for workers
_MAPS_PATH = None
_VOCAB = None
_SIMULATOR = None
_SCORER = None
_PROCESSOR = None
_OUTPUT_CACHE_PATH = None


def init_worker(maps_path_str, vocab_path_str, output_cache_path_str):
    global _MAPS_PATH, _VOCAB, _SIMULATOR, _SCORER, _PROCESSOR, _OUTPUT_CACHE_PATH
    _MAPS_PATH = Path(maps_path_str)
    _OUTPUT_CACHE_PATH = Path(output_cache_path_str)
    
    os.environ["NUPLAN_MAPS_ROOT"] = maps_path_str
    _VOCAB = np.load(vocab_path_str)
    
    sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    _SIMULATOR = PDMSimulator(sampling)
    _SCORER = PDMScorer(sampling, config=PDMScorerConfig(human_penalty_filter=False))
    _PROCESSOR = MetricCacheProcessor(cache_path=None, force_feature_computation=True, proposal_sampling=sampling)


def process_scene_worker(args):
    token, log_pickle_path_str, start_idx, end_idx = args
    
    out_file = _OUTPUT_CACHE_PATH / f"{token}.pkl"
    if out_file.exists():
        return True
        
    try:
        with open(log_pickle_path_str, "rb") as f:
            scene_dict_list = pickle.load(f)
        frame_list = scene_dict_list[start_idx:end_idx]
        
        from navsim.common.dataclasses import Scene, SensorConfig
        original_sensor_path = _MAPS_PATH.parent
        scene = Scene.from_scene_dict_list(
            frame_list,
            original_sensor_path,
            num_history_frames=4,
            num_future_frames=10,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        
        scenario = NavSimScenario(scene, map_root=str(_MAPS_PATH), map_version="nuplan-maps-v1.0")
        
        # Build metric cache
        metric_cache = _PROCESSOR.compute_metric_cache(scenario)
        initial_ego_state = metric_cache.ego_state
        
        num_vocab = _VOCAB.shape[0]
        
        # Vectorized coordinate transform
        x_ego = _VOCAB[..., 0]
        y_ego = _VOCAB[..., 1]
        yaw_ego = _VOCAB[..., 2]
        
        yaw_0 = initial_ego_state.rear_axle.heading
        x_0 = initial_ego_state.rear_axle.x
        y_0 = initial_ego_state.rear_axle.y
        
        x_global = x_ego * np.cos(yaw_0) - y_ego * np.sin(yaw_0) + x_0
        y_global = x_ego * np.sin(yaw_0) + y_ego * np.cos(yaw_0) + y_0
        yaw_global = yaw_ego + yaw_0
        
        # Construct trajectory states [8192, 41, 11]
        trajectory_states = np.zeros((num_vocab, 41, 11), dtype=np.float64)
        initial_state_arr = ego_state_to_state_array(initial_ego_state)
        trajectory_states[:, 0] = initial_state_arr
        
        trajectory_states[:, 1:, 0] = x_global
        trajectory_states[:, 1:, 1] = y_global
        trajectory_states[:, 1:, 2] = yaw_global
        
        dx = np.diff(trajectory_states[:, :, 0], axis=1) # [8192, 40]
        dy = np.diff(trajectory_states[:, :, 1], axis=1) # [8192, 40]
        dist = np.hypot(dx, dy)
        speed = dist / 0.5 # dt = 0.5s
        
        trajectory_states[:, 1:, 3] = speed * np.cos(yaw_global) # VELOCITY_X
        trajectory_states[:, 1:, 4] = speed * np.sin(yaw_global) # VELOCITY_Y
        
        dvx = np.diff(trajectory_states[:, :, 3], axis=1) # [8192, 40]
        dvy = np.diff(trajectory_states[:, :, 4], axis=1) # [8192, 40]
        trajectory_states[:, 1:-1, 5] = dvx[:, 1:] / 0.5 # ACCELERATION_X
        trajectory_states[:, 1:-1, 6] = dvy[:, 1:] / 0.5 # ACCELERATION_Y
        trajectory_states[:, -1, 5] = trajectory_states[:, -2, 5]
        trajectory_states[:, -1, 6] = trajectory_states[:, -2, 6]
        
        # Run simulation
        simulated_states = _SIMULATOR.simulate_proposals(trajectory_states, initial_ego_state)
        
        # Score proposals
        pdm_results = _SCORER.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            map_parameters=metric_cache.map_parameters,
            simulated_agent_detections_tracks=metric_cache.future_tracked_objects,
            human_past_trajectory=metric_cache.past_human_trajectory
        )
        
        # Extract sub-scores
        scores_dict = {
            'no_at_fault_collisions': np.zeros(num_vocab, dtype=np.float32),
            'drivable_area_compliance': np.zeros(num_vocab, dtype=np.float32),
            'time_to_collision_within_bound': np.zeros(num_vocab, dtype=np.float32),
            'ego_progress': np.zeros(num_vocab, dtype=np.float32),
            'driving_direction_compliance': np.zeros(num_vocab, dtype=np.float32),
            'lane_keeping': np.zeros(num_vocab, dtype=np.float32),
            'traffic_light_compliance': np.zeros(num_vocab, dtype=np.float32),
        }
        
        for p_idx in range(num_vocab):
            res = pdm_results[p_idx]
            scores_dict['no_at_fault_collisions'][p_idx] = float(res['no_at_fault_collisions'].iloc[0])
            scores_dict['drivable_area_compliance'][p_idx] = float(res['drivable_area_compliance'].iloc[0])
            scores_dict['time_to_collision_within_bound'][p_idx] = float(res['time_to_collision_within_bound'].iloc[0])
            scores_dict['ego_progress'][p_idx] = float(res['ego_progress'].iloc[0])
            scores_dict['driving_direction_compliance'][p_idx] = float(res['driving_direction_compliance'].iloc[0])
            scores_dict['lane_keeping'][p_idx] = float(res['lane_keeping'].iloc[0])
            scores_dict['traffic_light_compliance'][p_idx] = float(res['traffic_light_compliance'].iloc[0])
            
        with open(out_file, "wb") as f:
            pickle.dump(scores_dict, f)
            
        return True
    except Exception as e:
        print(f"Error processing token {token}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Precompute Teacher Scores")
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
        "--vocab-path",
        type=str,
        default=None,
        help="Path to vocabulary trajectory npy file (defaults to relative 'traj_final/8192.npy')",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to output teacher scores cache directory (defaults to <workspace>/teacher_scores_cache)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker processes (0 = auto = min(10, cpu_count())). Set to your pod's vCPU count; "
        "cpu_count() reports the HOST cores inside containers, which oversubscribes small pods.",
    )
    args = parser.parse_args()

    print("=== Precomputing Hydra-MDP Teacher Scores (Multiprocessing) ===")
    
    navsim_workspace = Path(args.workspace)
    log_path = navsim_workspace / "trainval_navsim_logs" / "trainval"
    if not log_path.exists():
        log_path = navsim_workspace / "mini_navsim_logs" / "mini"
    if not log_path.exists():
        log_path = navsim_workspace
        print(f"Warning: Default log paths not found. Scanning workspace directly: {log_path}")

    maps_path = Path(args.maps_path) if args.maps_path else navsim_workspace / "maps"
    
    # Try resolving vocab_path relative to this file or from traj_final
    if args.vocab_path:
        vocab_path = Path(args.vocab_path)
    else:
        vocab_path = Path("traj_final/8192.npy")
        if not vocab_path.exists():
            vocab_path = Path(os.path.dirname(__file__)).parent.parent / "traj_final" / "8192.npy"

    output_cache_path = Path(args.output_path) if args.output_path else navsim_workspace / "teacher_scores_cache"
    
    os.makedirs(output_cache_path, exist_ok=True)
    
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True)
    num_frames = scene_filter.num_frames
    frame_interval = scene_filter.frame_interval
    
    print("Scanning pickle logs for tokens (lazy scanning)...")
    log_files = list(log_path.iterdir())
    
    worker_args = []
    for log_pickle_path in tqdm(log_files, desc="Scanning pickle files"):
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
                
                worker_args.append((token, str(log_pickle_path), i, i + num_frames))
        except Exception as e:
            print(f"Error scanning {log_pickle_path}: {e}")
            
    print(f"Found {len(worker_args)} scenes.")
    
    num_workers = args.num_workers if args.num_workers > 0 else min(10, cpu_count())
    print(f"Starting multiprocessing pool with {num_workers} workers (cpu_count={cpu_count()})...")
    
    init_args = (str(maps_path), str(vocab_path), str(output_cache_path))
    
    success_count = 0
    with Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_scene_worker, worker_args),
            total=len(worker_args),
            desc="Precomputing Teacher Scores"
        ))
        success_count = sum(1 for r in results if r)
        
    print(f"=== Caching Completed: {success_count}/{len(worker_args)} successful ===")


if __name__ == "__main__":
    main()
