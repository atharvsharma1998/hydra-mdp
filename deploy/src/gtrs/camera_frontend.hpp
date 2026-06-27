// SPDX-License-Identifier: MIT
// Camera front-end for GTRS-BEVFusion using the roiconvert library.
//
// One batched roiconv call turns N raw camera frames into the normalized,
// planar fp16 [N,3,Hout,Wout] tensor consumed by camera.backbone. roiconv fuses
// crop + affine resize + per-channel (x*alpha+beta) normalization + RGB->CHW, so
// it replaces the fork's bespoke camera-normalization.cu and also accepts true
// camera formats (NV12 / YUV422 / RGBA) -- pick the InputFormat accordingly
// (this makes a separate YUVToRGB pass unnecessary).
#ifndef __GTRS_CAMERA_FRONTEND_HPP__
#define __GTRS_CAMERA_FRONTEND_HPP__

#include <cuda_fp16.h>

#include <memory>
#include <vector>

#include "roi_conversion/roi_conversion.hpp"

namespace gtrs {

struct CameraFrontendParam {
  int num_camera = 6;
  int input_width = 1920;   // raw frame size (per source)
  int input_height = 1080;
  int output_width = 704;   // model input (matches config.image_size = (256,704))
  int output_height = 256;
  // ImageNet normalization with 1/255 scaling (matches GTRSBevfusionConfig).
  float mean[3] = {0.485f, 0.456f, 0.406f};
  float std[3] = {0.229f, 0.224f, 0.225f};
  roiconv::InputFormat input_format = roiconv::InputFormat::RGB;  // decoded JPEG = RGB
  roiconv::Interpolation interpolation = roiconv::Interpolation::Bilinear;
};

// Produces a device fp16 buffer [N,3,Hout,Wout] (planar RGB, batch-contiguous).
class CameraFrontend {
 public:
  explicit CameraFrontend(const CameraFrontendParam& param);
  ~CameraFrontend();

  // image_planes[i] = device pointer to camera i's raw frame (format = param.input_format).
  // For RGB: stride = input_width*3. Returns the normalized device fp16 tensor.
  const half* forward(const std::vector<const void*>& image_planes, void* stream);

  const half* output() const { return output_; }
  int output_numel() const { return param_.num_camera * 3 * param_.output_height * param_.output_width; }

 private:
  CameraFrontendParam param_;
  std::shared_ptr<roiconv::ROIConversion> roi_;
  half* output_ = nullptr;
};

}  // namespace gtrs

#endif  // __GTRS_CAMERA_FRONTEND_HPP__
