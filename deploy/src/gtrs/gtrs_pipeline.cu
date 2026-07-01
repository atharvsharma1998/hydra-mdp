// SPDX-License-Identifier: MIT
#include "gtrs/gtrs_pipeline.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <fstream>
#include <vector>

#include "bevfusion/lidar-voxelization.hpp"
#include "common/check.hpp"
#include "common/dtype.hpp"
#include "common/tensor.hpp"

namespace gtrs {

// Write a .tensor (fork format: magic, ndim, Float32 code, int32 dims, raw f32)
// from a device fp16 buffer, for stage-by-stage PyTorch parity diffs.
static void dump_dev_half(const std::string& path, const void* dev, const std::vector<int>& dims,
                          cudaStream_t s) {
  size_t n = 1;
  for (int d : dims) n *= static_cast<size_t>(d);
  std::vector<__half> hbuf(n);
  checkRuntime(cudaMemcpyAsync(hbuf.data(), dev, n * sizeof(__half), cudaMemcpyDeviceToHost, s));
  checkRuntime(cudaStreamSynchronize(s));
  std::vector<float> f(n);
  for (size_t i = 0; i < n; ++i) f[i] = __half2float(hbuf[i]);
  std::ofstream of(path, std::ios::binary);
  int head[3] = {0x33ff1101, static_cast<int>(dims.size()), 3 /*Float32*/};
  of.write(reinterpret_cast<const char*>(head), sizeof(head));
  of.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(int));
  of.write(reinterpret_cast<const char*>(f.data()), f.size() * sizeof(float));
}

// Write a .tensor from a host float vector (already-decoded head outputs).
static void dump_host_float(const std::string& path, const std::vector<float>& f,
                            const std::vector<int>& dims) {
  std::ofstream of(path, std::ios::binary);
  int head[3] = {0x33ff1101, static_cast<int>(dims.size()), 3 /*Float32*/};
  of.write(reinterpret_cast<const char*>(head), sizeof(head));
  of.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(int));
  of.write(reinterpret_cast<const char*>(f.data()), f.size() * sizeof(float));
}

using bevfusion::camera::GeometryParameter;
using bevfusion::lidar::CoordinateOrder;
using bevfusion::lidar::Precision;
using bevfusion::lidar::SCNParameter;
using bevfusion::lidar::VoxelizationParameter;

static std::string join(const std::string& dir, const std::string& f) { return dir + "/" + f; }

Pipeline::Pipeline(const PipelineParam& param) : param_(param) {
  const std::string& m = param_.model_dir;

  // ---- LiDAR SCN (custom spconv parser; NAVSIM geometry) ----
  VoxelizationParameter vp;
  vp.min_range = nvtype::Float3(-32.0f, -32.0f, -3.0f);
  vp.max_range = nvtype::Float3(+32.0f, +32.0f, +5.0f);
  vp.voxel_size = nvtype::Float3(0.08f, 0.08f, 0.2f);
  vp.grid_size = vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);  // (800,800,40)
  vp.max_points_per_voxel = 10;
  vp.max_points = 300000;
  vp.max_voxels = 120000;
  vp.num_feature = 5;

  SCNParameter scn;
  scn.voxelization = vp;
  scn.model = join(m, "lidar.backbone.xyz.onnx");  // raw ONNX, run by custom parser
  scn.order = CoordinateOrder::XYZ;
  scn.precision = param_.fp16_only ? Precision::Float16 : Precision::Int8;
  scn_ = bevfusion::lidar::create_scn(scn);

  // ---- camera LSS geometry (matches export: bevpool grid 200x200, C=80) ----
  GeometryParameter g;
  g.xbound = nvtype::Float3(-32.0f, 32.0f, 0.32f);  // -> 200
  g.ybound = nvtype::Float3(-32.0f, 32.0f, 0.32f);  // -> 200
  g.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  g.dbound = nvtype::Float3(1.0f, 60.0f, 0.5f);  // -> D = 118
  g.geometry_dim = nvtype::Int3(200, 200, 80);   // (x, y, channels)
  g.feat_width = 88;
  g.feat_height = 32;
  g.image_width = 704;
  g.image_height = 256;
  g.num_camera = param_.camera_frontend.num_camera;
  geometry_ = bevfusion::camera::create_geometry(g);

  depth_ = bevfusion::camera::create_depth(g.image_width, g.image_height, g.num_camera);
  backbone_ = bevfusion::camera::create_backbone(join(m, "build/camera.backbone.plan"));
  // camera_shape = {N, C, D, H, W}; bevpool grid = xbound/ybound resolution.
  bevpool_ = bevfusion::camera::create_bevpool(backbone_->camera_shape(), 200, 200);
  vtransform_ = bevfusion::camera::create_vtransform(join(m, "build/camera.vtransform.plan"));

  // ---- fuser + heads (name-agnostic TRT modules) ----
  fuser_ = std::make_unique<TrtModule>(join(m, "build/fuser.plan"));
  fuser_->declare_outputs({"fenv"});

  planning_ = std::make_unique<TrtModule>(join(m, "build/planning_head.plan"));
  planning_->declare_outputs({"scores", "trajectory"});

  // CenterPoint detection head: 4 dense conv maps; peak decode runs host-side.
  det_ = std::make_unique<TrtModule>(join(m, "build/det_head.plan"));
  det_->declare_outputs({"det_heatmap", "det_offset", "det_size", "det_heading"});

  seg_ = std::make_unique<TrtModule>(join(m, "build/seg_head.plan"));
  seg_->declare_outputs({"bev_semantic_logits"});

  // ---- front-ends ----
  if (param_.use_camera_frontend) camera_frontend_ = std::make_unique<CameraFrontend>(param_.camera_frontend);
  LidarFrontendParam lp;  // defaults already match pc-range
  lidar_frontend_ = std::make_unique<LidarFrontend>(lp);

  checkRuntime(cudaMalloc(&status_dev_, 24 * sizeof(half)));

  if (param_.enable_timer) {
    tl_.resize(kNumStages + 1);
    for (auto& e : tl_) checkRuntime(cudaEventCreate(&e));
  }

  // optional cam_bev injection (PyTorch reference) to isolate SCN+fuser+heads
  if (!param_.cam_bev_override.empty()) {
    auto t = nv::Tensor::load(param_.cam_bev_override, false);
    Assertf(!t.empty(), "cam_bev_override not found: %s", param_.cam_bev_override.c_str());
    auto dev = t.to_device().to_half();  // host fp32 -> device fp32 -> device fp16
    int n = 1;
    for (int i = 0; i < (int)dev.shape.size(); ++i) n *= (int)dev.shape[i];
    checkRuntime(cudaMalloc(&cam_bev_override_, n * sizeof(half)));
    checkRuntime(cudaMemcpy(cam_bev_override_, dev.ptr<half>(), n * sizeof(half), cudaMemcpyDeviceToDevice));
    printf("cam_bev_override: loaded %s (%d elems) -> skipping camera LSS branch\n",
           param_.cam_bev_override.c_str(), n);
  }
}

Pipeline::~Pipeline() {
  for (auto& e : tl_) cudaEventDestroy(e);
}

void Pipeline::reset_timing() {
  for (int i = 0; i < kNumStages; ++i) acc_ms_[i] = 0;
  timer_iters_ = 0;
}

void Pipeline::report_timing() const {
  if (timer_iters_ == 0) {
    printf("[timing] no timed iterations (enable_timer=false or forward not called)\n");
    return;
  }
  static const char* names[kNumStages] = {"LiDAR (range-crop+SCN)", "Camera (LSS)", "Fuser (F_env)",
                                          "Heads (plan+det+seg)"};
  double total = 0;
  printf("\n================ Latency (GPU, mean over %d iters) ================\n", timer_iters_);
  for (int i = 0; i < kNumStages; ++i) {
    double ms = acc_ms_[i] / timer_iters_;
    total += ms;
    printf("  %-26s %7.3f ms\n", names[i], ms);
  }
  printf("  %-26s %7.3f ms\n", "-- total (device) --", total);
  printf("  %-26s %7.1f FPS\n", "throughput", total > 0 ? 1000.0 / total : 0.0);
  printf("==================================================================\n");
}

void Pipeline::update(const float* camera2lidar, const float* camera_intrinsics, const float* lidar2image,
                      const float* img_aug_matrix, void* stream) {
  geometry_->update(camera2lidar, camera_intrinsics, img_aug_matrix, stream);
  depth_->update(img_aug_matrix, lidar2image, stream);
}

PipelineOutput Pipeline::forward(const std::vector<const void*>& camera_planes, const half* lidar_points,
                                 int num_points, const float status[24], void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  auto h = [](const half* p) { return reinterpret_cast<const nvtype::half*>(p); };
  // when dumping, sync + check after each stage so an async illegal access is
  // attributed to the right stage instead of surfacing at a later enqueue.
  const bool dbg = !param_.dump_dir.empty();
  auto stage = [&](const char* name) {
    if (!dbg) return;
    cudaError_t e = cudaStreamSynchronize(s);
    if (e == cudaSuccess) e = cudaGetLastError();
    printf("  [stage] %-12s %s\n", name, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    Assertf(e == cudaSuccess, "stage %s failed: %s", name, cudaGetErrorString(e));
  };
  // per-stage timeline (async event records; one sync at the end). Boundaries:
  // 0=start, 1=after lidar, 2=after camera, 3=after fuser, 4=after heads.
  auto tmark = [&](int i) {
    if (param_.enable_timer) checkRuntime(cudaEventRecord(tl_[i], s));
  };

  // upload ego status [24] fp16
  half status_h[24];
  for (int i = 0; i < 24; ++i) status_h[i] = __float2half(status[i]);
  checkRuntime(cudaMemcpyAsync(status_dev_, status_h, 24 * sizeof(half), cudaMemcpyHostToDevice, s));

  tmark(0);
  // ---- LiDAR branch ----
  const half* pts = lidar_points;
  int npts = num_points;
  if (param_.use_range_crop) {
    unsigned int kept = 0;
    pts = lidar_frontend_->range_crop(lidar_points, num_points, &kept, stream);
    npts = static_cast<int>(kept);
  }
  const nvtype::half* lidar_bev = scn_->forward(h(pts), npts, stream);  // [1,256,100,100]
  stage("scn");
  tmark(1);

  // ---- camera branch (or injected reference cam_bev) ----
  const half* cam_bev;
  if (cam_bev_override_) {
    cam_bev = cam_bev_override_;
    stage("cam_bev_inject");
  } else {
    const half* normed = camera_planes.empty()
                             ? nullptr
                             : (param_.use_camera_frontend ? camera_frontend_->forward(camera_planes, stream)
                                                           : reinterpret_cast<const half*>(camera_planes[0]));
    stage("camera_pre");
    // depth map from lidar points projected into images
    nvtype::half* depthmap = depth_->forward(h(lidar_points), num_points, 5, stream);
    stage("depth");
    backbone_->forward(h(normed), depthmap, stream);
    stage("backbone");
    if (dbg) {
      unsigned int ni = geometry_->num_intervals();
      unsigned int nidx = geometry_->num_indices();  // = N*D*H*W frustum points
      std::vector<int> iv(static_cast<size_t>(ni) * 3);
      checkRuntime(cudaMemcpy(iv.data(), geometry_->intervals(), iv.size() * sizeof(int), cudaMemcpyDeviceToHost));
      long badx = 0, bady = 0, badz = 0;
      int minx = 1 << 30, maxy = -(1 << 30), maxz = -(1 << 30), minz = 1 << 30;
      const int bev_cells = 200 * 200;
      for (unsigned int i = 0; i < ni; ++i) {
        int x = iv[i * 3], y = iv[i * 3 + 1], z = iv[i * 3 + 2];
        if (x < 0 || x > (int)nidx) ++badx;
        if (y < x || y > (int)nidx) ++bady;
        if (z < 0 || z >= bev_cells) ++badz;
        minx = std::min(minx, x); maxy = std::max(maxy, y);
        maxz = std::max(maxz, z); minz = std::min(minz, z);
      }
      printf("  [geom] n_intervals=%u num_indices=%u  badx=%ld bady=%ld badz=%ld  "
             "minx=%d maxy=%d z=[%d,%d] bev_cells=%d\n",
             ni, nidx, badx, bady, badz, minx, maxy, minz, maxz, bev_cells);
    }
    if (dbg) {
      dump_dev_half(param_.dump_dir + "/dbg_depthmap.tensor", depthmap, {6, 1, 256, 704}, s);
      dump_dev_half(param_.dump_dir + "/dbg_context.tensor", backbone_->feature(), {6, 32, 88, 80}, s);
      dump_dev_half(param_.dump_dir + "/dbg_depthw.tensor", backbone_->depth(), {6, 118, 32, 88}, s);
    }
    nvtype::half* cam_bev_200 = bevpool_->forward(backbone_->feature(), backbone_->depth(), geometry_->indices(),
                                                  geometry_->intervals(), geometry_->num_intervals(), stream);
    stage("bevpool");
    if (dbg) dump_dev_half(param_.dump_dir + "/dbg_cambev200.tensor", cam_bev_200, {1, 80, 200, 200}, s);
    cam_bev = reinterpret_cast<const half*>(vtransform_->forward(cam_bev_200, stream));  // [1,80,100,100]
    stage("vtransform");
  }
  tmark(2);

  // ---- fuser -> F_env ----
  fuser_->forward({{"camera_bev", cam_bev}, {"lidar_bev", lidar_bev}}, stream);
  const half* fenv = fuser_->output("fenv");  // [1,512,100,100]
  stage("fuser");
  tmark(3);

  if (!param_.dump_dir.empty()) {
    auto sh = scn_->shape();
    std::vector<int> ldims(sh.begin(), sh.end());
    dump_dev_half(param_.dump_dir + "/lidar_bev.tensor", lidar_bev, ldims, s);
    dump_dev_half(param_.dump_dir + "/cam_bev.tensor", cam_bev, vtransform_->feat_shape(), s);
    dump_dev_half(param_.dump_dir + "/fenv.tensor", fenv, fuser_->output_dims("fenv"), s);
  }

  // ---- heads ----
  planning_->forward({{"fenv", fenv}, {"status", status_dev_}}, stream);
  stage("planning");
  det_->forward({{"fenv", fenv}}, stream);  // CenterPoint head reads F_env only
  stage("det");
  seg_->forward({{"fenv", fenv}}, stream);
  stage("seg");
  tmark(4);

  if (param_.enable_timer) {
    checkRuntime(cudaEventSynchronize(tl_[kNumStages]));
    for (int k = 0; k < kNumStages; ++k) {
      float ms = 0.f;
      checkRuntime(cudaEventElapsedTime(&ms, tl_[k], tl_[k + 1]));
      acc_ms_[k] += ms;
    }
    ++timer_iters_;
  }

  std::vector<float> det_hm, det_off, det_sz, det_hd, scores, traj, seg_logits;
  det_->copy_output_to_host("det_heatmap", det_hm, stream);
  det_->copy_output_to_host("det_offset", det_off, stream);
  det_->copy_output_to_host("det_size", det_sz, stream);
  det_->copy_output_to_host("det_heading", det_hd, stream);
  planning_->copy_output_to_host("scores", scores, stream);
  planning_->copy_output_to_host("trajectory", traj, stream);
  seg_->copy_output_to_host("bev_semantic_logits", seg_logits, stream);

  if (!param_.dump_dir.empty()) {
    dump_host_float(param_.dump_dir + "/det_heatmap.tensor", det_hm, det_->output_dims("det_heatmap"));
    dump_host_float(param_.dump_dir + "/det_offset.tensor", det_off, det_->output_dims("det_offset"));
    dump_host_float(param_.dump_dir + "/det_size.tensor", det_sz, det_->output_dims("det_size"));
    dump_host_float(param_.dump_dir + "/det_heading.tensor", det_hd, det_->output_dims("det_heading"));
    dump_host_float(param_.dump_dir + "/scores.tensor", scores, planning_->output_dims("scores"));
    dump_host_float(param_.dump_dir + "/trajectory.tensor", traj, planning_->output_dims("trajectory"));
    dump_host_float(param_.dump_dir + "/bev_semantic_logits.tensor", seg_logits,
                    seg_->output_dims("bev_semantic_logits"));
  }

  PipelineOutput out;
  auto hm_dims = det_->output_dims("det_heatmap");  // [1, K, H, W]
  int det_k = hm_dims[1], det_h = hm_dims[2], det_w = hm_dims[3];
  out.detections = decode_detections_centerpoint(det_hm, det_off, det_sz, det_hd, det_k, det_h, det_w,
                                                 -32.0f, -32.0f, 32.0f, 32.0f);
  out.plan = decode_plan(traj, /*P=*/40, scores);
  auto seg_dims = seg_->output_dims("bev_semantic_logits");  // [1,7,256,256]
  out.seg_h = seg_dims[2];
  out.seg_w = seg_dims[3];
  out.seg = decode_segmentation(seg_logits, seg_dims[1], out.seg_h, out.seg_w);
  return out;
}

}  // namespace gtrs
