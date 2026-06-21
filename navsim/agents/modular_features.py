import cv2
import numpy as np
import torch
from typing import Any, Dict, List, Tuple

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import affinity
from shapely.geometry import LineString, Polygon

from navsim.common.dataclasses import AgentInput, Annotations, Scene
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class ModularFeatureBuilder(AbstractFeatureBuilder):
    """Computes a 6-channel BEV semantic grid and ego status history."""

    def __init__(
        self,
        bev_pixel_width: int = 256,
        bev_pixel_height: int = 256,
        pixels_per_meter: float = 4.0,
        lidar_min_x: float = -32.0,
        lidar_max_x: float = 32.0,
        lidar_min_y: float = -32.0,
        lidar_max_y: float = 32.0,
        num_ego_status_frames: int = 3,
    ):
        super().__init__()
        self.bev_pixel_width = bev_pixel_width
        self.bev_pixel_height = bev_pixel_height
        self.pixels_per_meter = pixels_per_meter
        self.lidar_min_x = lidar_min_x
        self.lidar_max_x = lidar_max_x
        self.lidar_min_y = lidar_min_y
        self.lidar_max_y = lidar_max_y
        self.num_ego_status_frames = num_ego_status_frames

        # Define grid semantic mapping
        self.bev_semantic_classes = {
            0: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road/drivable area
            1: ("polygon", [SemanticMapLayer.WALKWAYS]),                            # walkway
            2: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]) # centerline
        }

    def get_unique_name(self) -> str:
        return "modular_features"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # During pure training on cache, compute_features only receives AgentInput.
        # But AgentInput does NOT have annotations/map_api.
        # So we construct the dynamic features here (ego status history)
        # and we will handle the BEV grid in compute_targets or in a custom dataset wrapper
        # to ensure it has access to the full Scene object.
        
        status_feature = self._get_status_feature(agent_input)
        return {"status_feature": status_feature}

    def _get_status_feature(self, agent_input: AgentInput) -> torch.Tensor:
        # Get last num_ego_status_frames (default 3)
        history_statuses = agent_input.ego_statuses[-self.num_ego_status_frames:]
        
        status_list = []
        for status in history_statuses:
            vel_x = float(status.ego_velocity[0])
            vel_y = float(status.ego_velocity[1])
            vel_mag = float(np.hypot(vel_x, vel_y))
            acc_x = float(status.ego_acceleration[0])
            acc_y = float(status.ego_acceleration[1])
            
            # command is a 3-element one-hot vector: [left, right, straight]
            cmd = status.driving_command
            cmd_left = float(cmd[0])
            cmd_right = float(cmd[1])
            cmd_straight = float(cmd[2])
            
            status_list.extend([vel_x, vel_y, vel_mag, acc_x, acc_y, cmd_left, cmd_right, cmd_straight])
            
        # If we have fewer frames than requested, pad with zeros
        while len(status_list) < self.num_ego_status_frames * 8:
            status_list = [0.0] * 8 + status_list
            
        return torch.tensor(status_list, dtype=torch.float32)

    def compute_bev_grid(
        self,
        annotations: Annotations,
        map_api: AbstractMap,
        ego_pose: StateSE2,
        traffic_lights: List[Tuple[str, bool]] = None,
    ) -> torch.Tensor:
        """
        Creates a 6-channel one-hot BEV semantic grid.
        Channel 0: Road/Drivable Area
        Channel 1: Walkways
        Channel 2: Centerline
        Channel 3: Static Obstacles (Barrier, Cone, etc.) & Red Traffic Lights
        Channel 4: Dynamic Vehicles
        Channel 5: Dynamic Pedestrians
        """
        grid = np.zeros((6, self.bev_pixel_height, self.bev_pixel_width), dtype=np.float32)
        
        # 1. Map features (drivable area, walkways, centerline)
        if map_api is not None:
            map_object_dict = map_api.get_proximal_map_objects(
                point=ego_pose.point, radius=self.bev_radius, layers=list(self.bev_semantic_classes[0][1]) + list(self.bev_semantic_classes[1][1])
            )
            
            # Channel 0: Road/Drivable area
            road_mask = self._compute_map_polygon_mask(map_object_dict, ego_pose, self.bev_semantic_classes[0][1])
            grid[0] = road_mask.astype(np.float32)
            
            # Channel 1: Walkways
            walkway_mask = self._compute_map_polygon_mask(map_object_dict, ego_pose, self.bev_semantic_classes[1][1])
            grid[1] = walkway_mask.astype(np.float32)
            
            # Channel 2: Centerline
            centerline_mask = self._compute_map_linestring_mask(map_object_dict, ego_pose, self.bev_semantic_classes[2][1])
            grid[2] = centerline_mask.astype(np.float32)
            
        # 2. Bounding box features (static obstacles, vehicles, pedestrians)
        if annotations is not None:
            # Channel 3: Static Obstacles & Red Traffic Lights
            static_types = [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ]
            static_mask = self._compute_box_mask(annotations, static_types)
            
            # Add red traffic light polygons to Channel 3 static mask
            if traffic_lights is not None and map_api is not None:
                red_light_mask = self._compute_traffic_light_mask(map_api, ego_pose, traffic_lights)
                static_mask = static_mask | red_light_mask
                
            grid[3] = static_mask.astype(np.float32)
            
            # Channel 4: Vehicles
            grid[4] = self._compute_box_mask(annotations, [TrackedObjectType.VEHICLE]).astype(np.float32)
            
            # Channel 5: Pedestrians
            grid[5] = self._compute_box_mask(annotations, [TrackedObjectType.PEDESTRIAN]).astype(np.float32)
            
        return torch.from_numpy(grid)

    @property
    def bev_radius(self) -> float:
        return max(abs(self.lidar_min_x), abs(self.lidar_max_x), abs(self.lidar_min_y), abs(self.lidar_max_y))

    def _coords_to_pixel(self, coords):
        pixel_center = np.array([[self.bev_pixel_width / 2.0, self.bev_pixel_height / 2.0]])
        coords_idcs = (coords / (1.0 / self.pixels_per_meter)) + pixel_center
        # Note: Flip y-axis to match image space coordinate system (y-down)
        coords_idcs[:, 1] = self.bev_pixel_height - coords_idcs[:, 1]
        return coords_idcs.astype(np.int32)

    def _geometry_local_coords(self, geometry: Any, origin: StateSE2) -> Any:
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
        return rotated_geometry

    def _compute_map_polygon_mask(self, map_object_dict: Dict, ego_pose: StateSE2, layers: List[SemanticMapLayer]) -> np.ndarray:
        mask = np.zeros((self.bev_pixel_height, self.bev_pixel_width), dtype=np.uint8)
        for layer in layers:
            if layer in map_object_dict:
                for map_object in map_object_dict[layer]:
                    polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                    exterior = np.array(polygon.exterior.coords).reshape((-1, 2))
                    exterior = self._coords_to_pixel(exterior).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [exterior], color=1)
        return mask > 0

    def _compute_map_linestring_mask(self, map_object_dict: Dict, ego_pose: StateSE2, layers: List[SemanticMapLayer]) -> np.ndarray:
        mask = np.zeros((self.bev_pixel_height, self.bev_pixel_width), dtype=np.uint8)
        for layer in layers:
            if layer in map_object_dict:
                for map_object in map_object_dict[layer]:
                    linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                    points = np.array(linestring.coords).reshape((-1, 2))
                    points = self._coords_to_pixel(points).reshape((-1, 1, 2))
                    cv2.polylines(mask, [points], isClosed=False, color=1, thickness=2)
        return mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers: List[TrackedObjectType]) -> np.ndarray:
        mask = np.zeros((self.bev_pixel_height, self.bev_pixel_width), dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 2))
                exterior = self._coords_to_pixel(exterior).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [exterior], color=1)
        return mask > 0


class ModularTargetBuilder(AbstractTargetBuilder):
    """Computes future ego trajectory."""

    def __init__(self, trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)):
        super().__init__()
        self.trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "modular_targets"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self.trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses, dtype=torch.float32)}
