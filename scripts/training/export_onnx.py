import os
import torch
import torch.nn as nn
from pathlib import Path
from navsim.agents.modular_planner import ModularPlanner

class ModularPlannerONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bev_grid, status_feature):
        features = {
            "bev_grid": bev_grid,
            "status_feature": status_feature
        }
        predictions = self.model(features)
        return predictions["trajectory"], predictions["scores"]

import argparse

def main():
    parser = argparse.ArgumentParser(description="Export ModularPlanner to ONNX")
    parser.add_argument(
        "--workspace",
        type=str,
        default=os.environ.get("NAVSIM_EXP_ROOT", "./navsim_workspace"),
        help="Path to the NAVSIM workspace directory (defaults to NAVSIM_EXP_ROOT environment variable or ./navsim_workspace)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Custom path to the PyTorch checkpoint (defaults to <workspace>/checkpoints/best.pth)",
    )
    parser.add_argument(
        "--onnx-output-path",
        type=str,
        default=None,
        help="Custom path to output the ONNX model (defaults to <workspace>/checkpoints/modular_planner.onnx)",
    )
    args = parser.parse_args()

    navsim_workspace = Path(args.workspace)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else navsim_workspace / "checkpoints" / "best.pth"
    onnx_output_path = Path(args.onnx_output_path) if args.onnx_output_path else navsim_workspace / "checkpoints" / "modular_planner.onnx"

    print(f"Workspace path: {navsim_workspace}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"ONNX output path: {onnx_output_path}")

    print("Initializing ModularPlanner...")
    model = ModularPlanner()
    
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        # Remove "agent." prefix if it exists in state dict keys
        cleaned_state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_state_dict)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Exporting untrained model.")

    model.eval()
    wrapper = ModularPlannerONNXWrapper(model)

    # Dummy inputs
    dummy_bev_grid = torch.randn(1, 6, 256, 256, dtype=torch.float32)
    dummy_status_feature = torch.randn(1, 24, dtype=torch.float32)

    # Perform a dummy forward pass to check
    with torch.no_grad():
        traj, scores = wrapper(dummy_bev_grid, dummy_status_feature)
        print(f"Dummy forward pass succeeded. Trajectory shape: {traj.shape}, Scores shape: {scores.shape}")

    print("Exporting model to ONNX...")
    torch.onnx.export(
        wrapper,
        (dummy_bev_grid, dummy_status_feature),
        str(onnx_output_path),
        input_names=["bev_grid", "status_feature"],
        output_names=["trajectory", "scores"],
        dynamic_axes={
            "bev_grid": {0: "batch_size"},
            "status_feature": {0: "batch_size"},
            "trajectory": {0: "batch_size"},
            "scores": {0: "batch_size"}
        },
        opset_version=13,
        verbose=False
    )
    print(f"Model exported successfully to {onnx_output_path}")

if __name__ == "__main__":
    main()
