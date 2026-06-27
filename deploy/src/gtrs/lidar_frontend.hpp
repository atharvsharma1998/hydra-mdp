// SPDX-License-Identifier: MIT
// LiDAR front-end for GTRS-BEVFusion.
//
// SCN voxelization already range-crops, but dropping out-of-range points up
// front shrinks the H2D copy and voxelization work. We do that with a 5-channel
// aware passthrough kernel (cuPCL's cudaFilter assumes float4 / XYZI and would
// drop our 5th channel `t`). cuPCL's VOXELGRID downsample is exposed as an
// optional float4 fast-path (see voxel_downsample()), useful when point counts
// are very high and the intensity/time channels can be approximated.
#ifndef __GTRS_LIDAR_FRONTEND_HPP__
#define __GTRS_LIDAR_FRONTEND_HPP__

#include <cuda_fp16.h>

namespace gtrs {

struct LidarFrontendParam {
  int num_feature = 5;  // x, y, z, intensity, t
  float min_x = -32.0f, max_x = 32.0f;
  float min_y = -32.0f, max_y = 32.0f;
  float min_z = -3.0f, max_z = 5.0f;
  int max_points = 300000;
};

// Range-crops a [N, num_feature] fp16 point cloud (device) in place into a
// compacted device buffer. Returns the device pointer + writes the kept count.
class LidarFrontend {
 public:
  explicit LidarFrontend(const LidarFrontendParam& param);
  ~LidarFrontend();

  // points_in: device fp16 [num_points, num_feature]. Returns compacted device
  // buffer [out_count, num_feature]; out_count written to *kept.
  const half* range_crop(const half* points_in, int num_points, unsigned int* kept, void* stream);

 private:
  LidarFrontendParam param_;
  half* out_ = nullptr;
  unsigned int* d_count_ = nullptr;
};

}  // namespace gtrs

#endif  // __GTRS_LIDAR_FRONTEND_HPP__
