// SPDX-License-Identifier: MIT
// GTRS-BEVFusion end-to-end deploy pipeline.
//
// Reuses the fork's CUDA front-end (Depth/Backbone/BEVPool/Geometry/VTransform)
// and the custom-spconv SCN, parameterized to NAVSIM geometry (+/-32 m, voxel
// 0.08, F_env 100x100x512). The camera raw->normalized step uses roiconvert and
// the LiDAR range-crop uses our 5-ch kernel. The fuser + 3 heads run through the
// name-agnostic TrtModule (their ONNX binding names differ from the fork's).
#ifndef __GTRS_PIPELINE_HPP__
#define __GTRS_PIPELINE_HPP__

#include <cuda_fp16.h>

#include <memory>
#include <string>
#include <vector>

#include "bevfusion/camera-backbone.hpp"
#include "bevfusion/camera-bevpool.hpp"
#include "bevfusion/camera-depth.hpp"
#include "bevfusion/camera-geometry.hpp"
#include "bevfusion/camera-vtransform.hpp"
#include "bevfusion/lidar-scn.hpp"
#include "gtrs/camera_frontend.hpp"
#include "gtrs/gtrs_heads.hpp"
#include "gtrs/lidar_frontend.hpp"
#include "gtrs/trt_module.hpp"

namespace gtrs {

struct PipelineParam {
  std::string model_dir;            // e.g. "model/gtrs_bevfusion"
  bool fp16_only = true;            // SCN precision toggle
  bool use_camera_frontend = true;  // roiconvert raw->normalized
  bool use_range_crop = true;       // 5-ch passthrough before SCN
  std::string dump_dir;             // if set, write per-stage tensors here for parity
  std::string cam_bev_override;     // if set, load this .tensor as cam_bev and skip the
                                    // camera LSS branch (isolates SCN+fuser+heads parity)
  bool enable_timer = false;        // per-stage GPU timing (lidar/camera/fuser/heads)
  CameraFrontendParam camera_frontend;
};

struct PipelineOutput {
  std::vector<DetBox> detections;
  std::vector<uint8_t> seg;  // [256*256] argmax class ids
  int seg_h = 256, seg_w = 256;
  PlanResult plan;
};

class Pipeline {
 public:
  explicit Pipeline(const PipelineParam& param);

  // Host 6x4x4 calibration matrices (row-major), as in the fork's *.tensor files.
  void update(const float* camera2lidar, const float* camera_intrinsics, const float* lidar2image,
              const float* img_aug_matrix, void* stream = nullptr);

  // camera_planes[i]: device pointer to camera i raw frame (roiconv InputFormat).
  // lidar_points: device fp16 [num_points, 5]; status: host float[24] ego status.
  PipelineOutput forward(const std::vector<const void*>& camera_planes, const half* lidar_points, int num_points,
                         const float status[24], void* stream);

  // Print averaged per-stage GPU latency (only meaningful with enable_timer=true).
  void report_timing() const;
  void reset_timing();  // clear accumulators (call after warmup)

  ~Pipeline();

 private:
  PipelineParam param_;
  // fork modules
  std::shared_ptr<bevfusion::lidar::SCN> scn_;
  std::shared_ptr<bevfusion::camera::Depth> depth_;
  std::shared_ptr<bevfusion::camera::Backbone> backbone_;
  std::shared_ptr<bevfusion::camera::BEVPool> bevpool_;
  std::shared_ptr<bevfusion::camera::Geometry> geometry_;
  std::shared_ptr<bevfusion::camera::VTransform> vtransform_;
  // our modules
  std::unique_ptr<CameraFrontend> camera_frontend_;
  std::unique_ptr<LidarFrontend> lidar_frontend_;
  std::unique_ptr<TrtModule> fuser_;
  std::unique_ptr<TrtModule> planning_;
  std::unique_ptr<TrtModule> det_;
  std::unique_ptr<TrtModule> seg_;
  half* status_dev_ = nullptr;        // [24] fp16
  half* cam_bev_override_ = nullptr;  // [1,80,100,100] fp16, loaded from cam_bev_override

  // per-stage GPU timing: 5 timeline events -> 4 stages [lidar, camera, fuser, heads]
  static constexpr int kNumStages = 4;
  std::vector<cudaEvent_t> tl_;            // size kNumStages+1
  double acc_ms_[kNumStages] = {0, 0, 0, 0};
  int timer_iters_ = 0;
};

}  // namespace gtrs

#endif  // __GTRS_PIPELINE_HPP__
