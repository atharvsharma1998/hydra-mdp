# SPDX-License-Identifier: Apache-2.0
"""Export the GTRS-BEVFusion model to ONNX, mirroring CUDA-BEVFusion's multi-graph
deployment decomposition.

CUDA-BEVFusion splits the network into several ONNX graphs + custom CUDA/TRT
plugins (the spconv LiDAR backbone is *parsed* into a TensorRT SCN plugin rather
than run as native ONNX ops). We follow the same split:

  camera.backbone (+ vtransform)  ── reuse qat/export-camera.py on encoders.camera
  lidar.backbone (SCN)            ── reuse qat/export-scn.py   on encoders.lidar.backbone
                                     (custom exptool ONNX -> TRT SCN plugin)
  ┌──────────────── THIS SCRIPT (the new, NAVSIM-side graphs) ────────────────┐
  fuser.onnx        : ConvFuser([cam_bev, lidar_bev]) -> decoder.backbone/neck -> F_env
  planning_head.onnx: PlanningHead(F_env, status)  -> per-vocab scores (+ trajectory)
  det_head.onnx     : det decoder + MultiClassAgentHead(F_env, status) -> boxes, class logits
  seg_head.onnx     : BEVSegHead(F_env) -> BEV semantic logits
  └───────────────────────────────────────────────────────────────────────────┘

Everything here is plain conv / transformer / MLP / interpolate -> TensorRT-friendly.
The LSS BEV-pool (camera) and the spconv SCN stay as CUDA/TRT plugins, exactly as
in CUDA-BEVFusion.

Run (uses a real mini-split sample so traced shapes + parity are exact):
    python scripts/export/export_gtrs_bevfusion_onnx.py \
        --checkpoint navsim_workspace/checkpoints/overfit_mc_latest.pth \
        --sensor-blobs-path navsim_workspace/mini_sensor_blobs/mini \
        --out-dir navsim_workspace/onnx
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.agents.gtrs_bevfusion.bevfusion_features import BEVFusionFeatureBuilder
from navsim.agents.gtrs_bevfusion.bevfusion_model import GTRSBevfusionModel
from navsim.agents.gtrs_bevfusion.bevfusion_target import BEVFusionTargetBuilder


OPSET = 14


def _neck_first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _adaptive_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """Averaging matrix [n_out, n_in] that reproduces ``adaptive_avg_pool`` window
    boundaries (start=floor(i*n_in/n_out), end=ceil((i+1)*n_in/n_out))."""
    P = torch.zeros(n_out, n_in)
    for i in range(n_out):
        s = (i * n_in) // n_out
        e = -(-(i + 1) * n_in // n_out)  # ceil division
        P[i, s:e] = 1.0 / (e - s)
    return P


class FixedAdaptiveAvgPool2d(nn.Module):
    """Exact, ONNX/TRT-exportable replacement for ``nn.AdaptiveAvgPool2d`` at a
    FIXED input size: pools H then W via two precomputed MatMuls. Numerically
    identical to adaptive avg pool (no learned params) and avoids the
    "output size not a factor of input size" ONNX export error."""

    def __init__(self, in_hw, out_hw):
        super().__init__()
        in_h, in_w = in_hw
        out_h, out_w = out_hw
        self.register_buffer("Ph", _adaptive_matrix(in_h, out_h))      # [out_h, in_h]
        self.register_buffer("Pw", _adaptive_matrix(in_w, out_w).t())  # [in_w, out_w]

    def forward(self, x):  # x: [B, C, H, W]
        x = torch.matmul(self.Ph[None, None], x)   # [B, C, out_h, W]
        x = torch.matmul(x, self.Pw[None, None])   # [B, C, out_h, out_w]
        return x


def _out_size(adaptive_pool) -> tuple:
    os_ = adaptive_pool.output_size
    return (os_, os_) if isinstance(os_, int) else tuple(os_)


# --------------------------------------------------------------------------- #
# Export wrappers: each takes F_env (and status) so the heads decode the shared
# feature exactly as in GTRSBevfusionModel.forward.
# --------------------------------------------------------------------------- #
class FenvProducer(nn.Module):
    """ConvFuser + decoder backbone + neck: [cam_bev, lidar_bev] -> F_env."""

    def __init__(self, bevfusion):
        super().__init__()
        self.fuser = bevfusion.fuser
        self.decoder_backbone = bevfusion.decoder["backbone"]
        self.decoder_neck = bevfusion.decoder["neck"]

    def forward(self, camera_bev, lidar_bev):
        x = self.fuser([camera_bev, lidar_bev])
        x = self.decoder_backbone(x)
        x = self.decoder_neck(x)
        return _neck_first(x)


class PlanningHeadExport(nn.Module):
    """PlanningHead(F_env, status) -> per-vocab scores + best trajectory.

    Host side does the final argmax + vocab lookup if desired; we also emit the
    selected trajectory for convenience. randperm vocab-dropout is training-only
    and skipped in eval, so the traced graph is the full-vocab inference path.
    """

    def __init__(self, planning_head, fenv_hw):
        super().__init__()
        # swap adaptive pool -> fixed MatMul pool (exact, exportable) for this input size
        planning_head.bev_pool = FixedAdaptiveAvgPool2d(fenv_hw, _out_size(planning_head.bev_pool))
        self.head = planning_head

    def forward(self, fenv, status):
        out = self.head(fenv, status)
        return out["scores"], out["selected_trajectory"]


class DetHeadExport(nn.Module):
    """det pool/downscale + transformer decoder + MultiClassAgentHead."""

    def __init__(self, model, fenv_hw):
        super().__init__()
        self.det_pool = FixedAdaptiveAvgPool2d(fenv_hw, _out_size(model.det_pool))
        self.det_downscale = model.det_downscale
        self.det_status_encoding = model.det_status_encoding
        self.det_keyval_embedding = model.det_keyval_embedding
        self.det_query_embedding = model.det_query_embedding
        self.det_decoder = model.det_decoder
        self.agent_head = model.agent_head

    def forward(self, fenv, status):
        B = fenv.shape[0]
        tokens = self.det_downscale(self.det_pool(fenv)).flatten(-2, -1).permute(0, 2, 1)
        status_enc = self.det_status_encoding(status)
        keyval = torch.cat([tokens, status_enc[:, None]], dim=1)
        keyval = keyval + self.det_keyval_embedding.weight[None]
        query = self.det_query_embedding.weight[None].repeat(B, 1, 1)
        agent_q = self.det_decoder(query, keyval)
        out = self.agent_head(agent_q)
        return out["agent_states"], out["agent_class_logits"]


class SegHeadExport(nn.Module):
    def __init__(self, seg_head):
        super().__init__()
        self.seg_head = seg_head

    def forward(self, fenv):
        return self.seg_head(fenv)


class CameraBackboneExport(nn.Module):
    """Image backbone + neck + LSS depthnet (mirrors CUDA-BEVFusion
    qat/export-camera.py SubclassCameraModule). Outputs per-camera context
    features + depth weights; the LSS BEV-pool scatter stays a CUDA/TRT plugin."""

    def __init__(self, bevfusion):
        super().__init__()
        cam = bevfusion.encoders["camera"] if not hasattr(bevfusion.encoders, "camera") \
            else bevfusion.encoders.camera
        self.backbone = cam["backbone"] if isinstance(cam, dict) else cam.backbone
        self.neck = cam["neck"] if isinstance(cam, dict) else cam.neck
        self.vtransform = cam["vtransform"] if isinstance(cam, dict) else cam.vtransform

    def forward(self, img, depth):
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        feat = self.neck(self.backbone(img))
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]
        BN, C, fH, fW = map(int, feat.size())
        feat = feat.view(B, BN // B, C, fH, fW)

        vt = self.vtransform
        d = depth.view(B * N, *depth.shape[2:])
        x = feat.view(B * N, C, fH, fW)
        d = vt.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = vt.depthnet(x)
        depth_weights = x[:, : vt.D].softmax(dim=1)
        context = x[:, vt.D : (vt.D + vt.C)].permute(0, 2, 3, 1)
        return context, depth_weights


class CameraVTransformExport(nn.Module):
    """The post-BEV-pool downsample conv stack (camera.vtransform.onnx)."""

    def __init__(self, vtransform):
        super().__init__()
        self.downsample = vtransform.downsample

    def forward(self, x):
        return self.downsample(x)


# --------------------------------------------------------------------------- #
def get_intermediates(model, feats, device):
    """Run the backbone in fp32 (autocast off) and return cam_bev, lidar_bev,
    F_env, status_feature -- the exact tensors the export graphs consume."""
    m = model.backbone.bevfusion
    status = feats["status_feature"].to(torch.float32)

    # capture the real input to the camera vtransform downsample (pre-downsample
    # BEV grid) so the camera.vtransform.onnx dummy shape is exact.
    cap = {}
    cam_enc = m.encoders["camera"] if not hasattr(m.encoders, "camera") else m.encoders.camera
    vt = cam_enc["vtransform"] if isinstance(cam_enc, dict) else cam_enc.vtransform
    ds = getattr(vt, "downsample", None)
    h = None
    if ds is not None and not isinstance(ds, nn.Identity):
        h = ds.register_forward_pre_hook(lambda mod, inp: cap.setdefault("ds_in", inp[0].detach()))

    with torch.cuda.amp.autocast(enabled=False), torch.no_grad():
        kw = dict(
            img=feats["img"].float(),
            points=[p.float() for p in feats["lidar"]],
            camera2ego=feats["camera2ego"].float(),
            lidar2ego=feats["lidar2ego"].float(),
            lidar2camera=feats["lidar2camera"].float(),
            lidar2image=feats["lidar2image"].float(),
            camera_intrinsics=feats["camera_intrinsics"].float(),
            camera2lidar=feats["camera2lidar"].float(),
            img_aug_matrix=feats["img_aug_matrix"].float(),
            lidar_aug_matrix=feats["lidar_aug_matrix"].float(),
        )
        metas = [{} for _ in range(kw["img"].shape[0])]
        cam_bev = lidar_bev = None
        for sensor in m.encoders:
            if sensor == "camera":
                cam_bev = m.extract_camera_features(
                    kw["img"], kw["points"], kw["camera2ego"], kw["lidar2ego"],
                    kw["lidar2camera"], kw["lidar2image"], kw["camera_intrinsics"],
                    kw["camera2lidar"], kw["img_aug_matrix"], kw["lidar_aug_matrix"], metas,
                )
            elif sensor == "lidar":
                lidar_bev = m.extract_lidar_features(kw["points"])
        x = m.fuser([cam_bev, lidar_bev]) if m.fuser is not None else cam_bev
        x = m.decoder["backbone"](x)
        x = m.decoder["neck"](x)
        fenv = _neck_first(x)
    if h is not None:
        h.remove()
    return cam_bev, lidar_bev, fenv, status, cap.get("ds_in")


def export_lidar_scn(model, out_dir, in_channel, inverse):
    """Export the spconv LiDAR backbone with the project's CUSTOM spconv-2.x
    exporter (bevfusion/tools/export_lidar_onnx_spconv2.py).

    This emits SparseConvolution/ScatterDense nodes in the spconv-2.x KRSC weight
    layout consumed by the custom C++ parser (lidar-scn-onnx-parser-custom) running
    on the OPEN-SOURCE libspconv -- NOT TensorRT and NOT NVIDIA's closed engine.
    Imported lazily because it pulls in spconv.pytorch (CUDA)."""
    tools_dir = os.path.join(
        os.environ.get("BEVFUSION_ROOT", "/home/atharv/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion"),
        "tools",
    )
    sys.path.insert(0, tools_dir)
    from export_lidar_onnx_spconv2 import export_onnx as export_scn_onnx  # noqa: E402

    scn = model.backbone.bevfusion.encoders.lidar.backbone.eval().cuda()
    voxels = torch.zeros(1, in_channel).cuda()
    coors = torch.zeros(1, 4).int().cuda()
    save = str(Path(out_dir) / "lidar.backbone.onnx")
    export_scn_onnx(scn, voxels, coors, 1, inverse, save)


def _to_numpy(x):
    return x.detach().cpu().numpy()


# fork tensor.hpp DataType codes: Int32=1, Float16=2, Float32=3, Int64=4, UInt8=8
_TENSOR_DTYPE = {"float16": 2, "float32": 3, "int32": 1, "int64": 4, "uint8": 8}


def _save_tensor(path, arr):
    """Write a numpy array in CUDA-BEVFusion's .tensor format (magic, ndim, dtype,
    int32 dims, raw data) consumed by nv::Tensor::load."""
    a = np.ascontiguousarray(arr)
    code = _TENSOR_DTYPE[str(a.dtype)]
    with open(path, "wb") as f:
        np.array([0x33ff1101, a.ndim, code], dtype=np.int32).tofile(f)
        np.array(a.shape, dtype=np.int32).tofile(f)
        a.tofile(f)
    print(f"    {os.path.basename(str(path)):28s} {str(a.dtype):8s} {tuple(a.shape)}")


def dump_cpp_sample(out_dir, model, feats, cam_bev, lidar_bev, fenv, status, fenv_hw, use_det,
                    img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)):
    """Dump one real NAVSIM sample for procedural C++ parity.

    Inputs feed the C++ binary directly. The camera is dumped PRE-NORMALIZED (the
    exact backbone input feats['img']) so the C++ run bypasses roiconvert and we
    isolate LSS/backbone/fuser/head numerics from preprocessing. References are
    PyTorch's per-stage outputs computed with the SAME wrappers as the ONNX graphs,
    so a stage that matches ONNX must match here too -- any C++ divergence is in
    the TRT engine or C++ wiring, not the graph."""
    out_dir = Path(out_dir)
    ref = out_dir / "ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref.mkdir(parents=True, exist_ok=True)

    print("inputs (C++ consumes):")
    _save_tensor(out_dir / "camera.tensor", _to_numpy(feats["img"][0]).astype(np.float16))
    # de-normalized resized RGB (uint8, [N,H,W,3]) so the C++ binary can draw the
    # exact images the model saw (camera.tensor is normalized -> not displayable).
    _mean = np.asarray(img_mean, np.float32).reshape(3, 1, 1)
    _std = np.asarray(img_std, np.float32).reshape(3, 1, 1)
    _img = _to_numpy(feats["img"][0]).astype(np.float32)  # [N,3,H,W] normalized
    _rgb = ((_img * _std + _mean) * 255.0).clip(0, 255).astype(np.uint8)  # [N,3,H,W]
    _save_tensor(out_dir / "camera_rgb.tensor", np.ascontiguousarray(_rgb.transpose(0, 2, 3, 1)))
    _save_tensor(out_dir / "points.tensor", _to_numpy(feats["lidar"][0]).astype(np.float16))
    for name in ("camera2lidar", "camera_intrinsics", "lidar2image", "img_aug_matrix"):
        _save_tensor(out_dir / f"{name}.tensor", _to_numpy(feats[name][0]).astype(np.float32))
    _save_tensor(out_dir / "status.tensor", _to_numpy(status[0]).astype(np.float32))

    print("references (PyTorch per-stage):")
    dev = fenv.device  # FixedAdaptiveAvgPool2d registers its pool buffers on CPU
    with torch.no_grad():
        scores, traj = PlanningHeadExport(model.planning_head, fenv_hw).to(dev)(fenv, status)
        seg_logits = SegHeadExport(model.seg_head).to(dev)(fenv)
        det_out = DetHeadExport(model, fenv_hw).to(dev)(fenv, status) if use_det else None

    _save_tensor(ref / "lidar_bev.tensor", _to_numpy(lidar_bev).astype(np.float32))
    _save_tensor(ref / "cam_bev.tensor", _to_numpy(cam_bev).astype(np.float32))
    _save_tensor(ref / "fenv.tensor", _to_numpy(fenv).astype(np.float32))
    _save_tensor(ref / "scores.tensor", _to_numpy(scores).astype(np.float32))
    _save_tensor(ref / "trajectory.tensor", _to_numpy(traj).astype(np.float32))
    _save_tensor(ref / "bev_semantic_logits.tensor", _to_numpy(seg_logits).astype(np.float32))
    if det_out is not None:
        _save_tensor(ref / "agent_states.tensor", _to_numpy(det_out[0]).astype(np.float32))
        _save_tensor(ref / "agent_class_logits.tensor", _to_numpy(det_out[1]).astype(np.float32))


def export_and_check(wrapper, inputs, in_names, out_names, path, simplify=True):
    """Export one wrapper to ONNX and verify numerical parity with onnxruntime."""
    import onnx
    import onnxruntime as ort

    dev = inputs[0].device
    wrapper = wrapper.to(dev).eval()
    with torch.no_grad():
        ref = wrapper(*inputs)
    ref = (ref,) if isinstance(ref, torch.Tensor) else tuple(ref)

    torch.onnx.export(
        wrapper, tuple(inputs), path,
        input_names=in_names, output_names=out_names,
        opset_version=OPSET, do_constant_folding=True,
    )

    if simplify:
        try:
            from onnxsim import simplify as _simplify
            model_simp, ok = _simplify(onnx.load(path))
            if ok:
                onnx.save(model_simp, path)
        except Exception as e:
            print(f"    (onnxsim skipped: {e})")

    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {n: _to_numpy(t) for n, t in zip(in_names, inputs)}
    ort_out = sess.run(out_names, feed)

    max_err = 0.0
    for r, o in zip(ref, ort_out):
        max_err = max(max_err, float(np.abs(_to_numpy(r) - o).max()))
    status = "OK " if max_err < 1e-2 else "WARN"
    print(f"  [{status}] {os.path.basename(path)}  max|torch-ort|={max_err:.2e}  "
          f"out={[tuple(o.shape) for o in ort_out]}")
    return max_err


def main():
    p = argparse.ArgumentParser(description="Export GTRS-BEVFusion F_env + heads to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--workspace", default="/home/atharv/Downloads/hydramdp/navsim_workspace")
    p.add_argument("--sensor-blobs-path", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--lidar-scn", action="store_true",
                   help="also export the spconv LiDAR backbone via the custom spconv-2.x "
                        "exporter (SparseConvolution/ScatterDense ONNX for the open-source "
                        "libspconv C++ parser; needs spconv.pytorch + CUDA)")
    p.add_argument("--lidar-inverse", action="store_true", help="export SCN in zyx coord order")
    p.add_argument("--lidar-in-channel", type=int, default=None,
                   help="SCN input channels (default: config.lidar_in_channels)")
    p.add_argument("--dump-cpp-inputs", default=None, metavar="DIR",
                   help="dump one real NAVSIM token (pre-normalized camera, points, calib, "
                        "status) + PyTorch per-stage references as .tensor for C++ parity, "
                        "then exit (skips ONNX export)")
    args = p.parse_args()

    ws = Path(args.workspace)
    out_dir = Path(args.out_dir) if args.out_dir else ws / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    simplify = not args.no_simplify

    config = GTRSBevfusionConfig()
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)

    # one real mini sample -> exact traced shapes + meaningful parity check
    from train_gtrs_bevfusion import LazySceneLoader
    sb = Path(args.sensor_blobs_path)
    loader = LazySceneLoader(
        original_sensor_path=sb,
        data_paths=[ws / "mini_navsim_logs" / "mini"],
        scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True),
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )
    tok = next(t for t in loader.tokens
               if (sb / loader.token_to_slice[t][0].name.replace(".pkl", "") / "CAM_F0").is_dir())
    scene = loader.get_scene_from_token(tok)
    feats = BEVFusionFeatureBuilder(config).compute_features(scene.get_agent_input())
    feats = {k: ([v.to(device)] if k == "lidar" else v.unsqueeze(0).to(device)) for k, v in feats.items()}

    model = GTRSBevfusionModel(config, num_poses=ts.num_poses).to(device).eval()
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.checkpoint} (missing={len(missing)} unexpected={len(unexpected)})")

    cam_bev, lidar_bev, fenv, status, ds_in = get_intermediates(model, feats, device)
    fenv_hw = (int(fenv.shape[-2]), int(fenv.shape[-1]))
    print(f"intermediates: cam_bev={tuple(cam_bev.shape)} lidar_bev={tuple(lidar_bev.shape)} "
          f"F_env={tuple(fenv.shape)} status={tuple(status.shape)}\n")

    if args.dump_cpp_inputs:
        dump_cpp_sample(args.dump_cpp_inputs, model, feats, cam_bev, lidar_bev, fenv,
                        status, fenv_hw, config.use_detection_head,
                        config.img_norm_mean, config.img_norm_std)
        (Path(args.dump_cpp_inputs) / "token.txt").write_text(tok)
        print(f"    token.txt                    {tok}")
        # ground-truth boxes + trajectory for the C++ visualizer (GT solid, pred dotted)
        _dump_dir = Path(args.dump_cpp_inputs)
        _tgts = BEVFusionTargetBuilder(trajectory_sampling=ts).compute_targets(scene)
        _gt_states = _to_numpy(_tgts["agent_states"])
        _gt_mask = _to_numpy(_tgts["agent_labels"]).astype(bool)
        _gt_cls = _to_numpy(_tgts["agent_classes"]).astype(int) if "agent_classes" in _tgts \
            else np.zeros(len(_gt_states), int)
        _gt_states, _gt_cls = _gt_states[_gt_mask], _gt_cls[_gt_mask]
        with open(_dump_dir / "gt_detections.txt", "w") as f:
            f.write("# cls x y heading length width\n")
            for s, c in zip(_gt_states, _gt_cls):
                f.write(f"{int(c)} {s[0]} {s[1]} {s[2]} {s[3]} {s[4]}\n")
        _gt_traj = _to_numpy(_tgts["trajectory"]).reshape(-1, 3) if "trajectory" in _tgts else np.zeros((0, 3))
        with open(_dump_dir / "gt_trajectory.txt", "w") as f:
            for p in _gt_traj:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        print(f"    gt_detections.txt            {len(_gt_states)} boxes")
        print(f"    gt_trajectory.txt            {len(_gt_traj)} poses")
        print(f"\nDone (dump-only). C++ sample in {args.dump_cpp_inputs}")
        return

    print("Exporting + validating ONNX graphs:")
    # 0) camera backbone + neck + depthnet (LSS BEV-pool stays a CUDA/TRT plugin)
    bevfusion = model.backbone.bevfusion
    img = feats["img"]
    depth = torch.zeros(img.shape[0], img.shape[1], 1, img.shape[-2], img.shape[-1], device=device)
    try:
        export_and_check(
            CameraBackboneExport(bevfusion),
            (img, depth), ["img", "depth"], ["camera_feature", "camera_depth_weights"],
            str(out_dir / "camera.backbone.onnx"), simplify,
        )
    except Exception as e:
        print(f"  [FAIL] camera.backbone.onnx: {e}")
    if ds_in is not None:
        cam_enc = bevfusion.encoders["camera"] if not hasattr(bevfusion.encoders, "camera") else bevfusion.encoders.camera
        vt = cam_enc["vtransform"] if isinstance(cam_enc, dict) else cam_enc.vtransform
        try:
            export_and_check(
                CameraVTransformExport(vt),
                (ds_in,), ["feat_in"], ["feat_out"],
                str(out_dir / "camera.vtransform.onnx"), simplify,
            )
        except Exception as e:
            print(f"  [FAIL] camera.vtransform.onnx: {e}")

    # 1) F_env producer (fuser + decoder)
    export_and_check(
        FenvProducer(model.backbone.bevfusion),
        (cam_bev, lidar_bev), ["camera_bev", "lidar_bev"], ["fenv"],
        str(out_dir / "fuser.onnx"), simplify,
    )
    # 2) planning head
    try:
        export_and_check(
            PlanningHeadExport(model.planning_head, fenv_hw),
            (fenv, status), ["fenv", "status"], ["scores", "trajectory"],
            str(out_dir / "planning_head.onnx"), simplify,
        )
    except Exception as e:
        print(f"  [FAIL] planning_head.onnx: {e}")
    # 3) detection head
    if config.use_detection_head:
        try:
            export_and_check(
                DetHeadExport(model, fenv_hw),
                (fenv, status), ["fenv", "status"], ["agent_states", "agent_class_logits"],
                str(out_dir / "det_head.onnx"), simplify,
            )
        except Exception as e:
            print(f"  [FAIL] det_head.onnx: {e}")
    # 4) segmentation head
    if config.use_bev_seg_head:
        export_and_check(
            SegHeadExport(model.seg_head),
            (fenv,), ["fenv"], ["bev_semantic_logits"],
            str(out_dir / "seg_head.onnx"), simplify,
        )

    # 5) LiDAR SCN backbone (custom spconv-2.x exporter -> open-source libspconv parser)
    if args.lidar_scn:
        in_ch = args.lidar_in_channel or getattr(config, "lidar_in_channels", 5)
        try:
            export_lidar_scn(model, out_dir, in_ch, args.lidar_inverse)
            print(f"  [OK ] lidar.backbone{'.zyx' if args.lidar_inverse else '.xyz'}.onnx "
                  f"(SparseConvolution/ScatterDense; run via custom parser + open-source libspconv)")
        except Exception as e:
            print(f"  [FAIL] lidar.backbone.onnx: {e}")

    print(f"\nDone. ONNX graphs in {out_dir}")


if __name__ == "__main__":
    main()
