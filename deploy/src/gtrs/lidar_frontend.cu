// SPDX-License-Identifier: MIT
#include "gtrs/lidar_frontend.hpp"

#include "common/check.hpp"

namespace gtrs {

__global__ void range_crop_kernel(const half* __restrict__ in, int num_points, int F, float min_x, float max_x,
                                  float min_y, float max_y, float min_z, float max_z, half* __restrict__ out,
                                  unsigned int* __restrict__ count, int max_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_points) return;
  const half* p = in + static_cast<size_t>(i) * F;
  float x = __half2float(p[0]);
  float y = __half2float(p[1]);
  float z = __half2float(p[2]);
  if (x < min_x || x >= max_x || y < min_y || y >= max_y || z < min_z || z >= max_z) return;
  unsigned int slot = atomicAdd(count, 1u);
  if (static_cast<int>(slot) >= max_out) return;
  half* o = out + static_cast<size_t>(slot) * F;
  for (int c = 0; c < F; ++c) o[c] = p[c];
}

LidarFrontend::LidarFrontend(const LidarFrontendParam& param) : param_(param) {
  checkRuntime(cudaMalloc(&out_, static_cast<size_t>(param_.max_points) * param_.num_feature * sizeof(half)));
  checkRuntime(cudaMalloc(&d_count_, sizeof(unsigned int)));
}

LidarFrontend::~LidarFrontend() {
  if (out_) cudaFree(out_);
  if (d_count_) cudaFree(d_count_);
}

const half* LidarFrontend::range_crop(const half* points_in, int num_points, unsigned int* kept, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  checkRuntime(cudaMemsetAsync(d_count_, 0, sizeof(unsigned int), s));
  int threads = 256;
  int blocks = (num_points + threads - 1) / threads;
  checkKernel(range_crop_kernel<<<blocks, threads, 0, s>>>(
      points_in, num_points, param_.num_feature, param_.min_x, param_.max_x, param_.min_y, param_.max_y,
      param_.min_z, param_.max_z, out_, d_count_, param_.max_points));
  unsigned int h = 0;
  checkRuntime(cudaMemcpyAsync(&h, d_count_, sizeof(unsigned int), cudaMemcpyDeviceToHost, s));
  checkRuntime(cudaStreamSynchronize(s));
  if (h > static_cast<unsigned int>(param_.max_points)) h = param_.max_points;
  if (kept) *kept = h;
  return out_;
}

}  // namespace gtrs
