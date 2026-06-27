// SPDX-License-Identifier: MIT
#include "gtrs/camera_frontend.hpp"

#include "common/check.hpp"

namespace gtrs {

CameraFrontend::CameraFrontend(const CameraFrontendParam& param) : param_(param) {
  roi_ = roiconv::create();
  checkRuntime(cudaMalloc(&output_, output_numel() * sizeof(half)));
}

CameraFrontend::~CameraFrontend() {
  if (output_) cudaFree(output_);
}

const half* CameraFrontend::forward(const std::vector<const void*>& image_planes, void* stream) {
  Assertf(static_cast<int>(image_planes.size()) == param_.num_camera,
          "expected %d camera planes, got %zu", param_.num_camera, image_planes.size());

  const int stride_per_cam = 3 * param_.output_height * param_.output_width;  // CHW fp16 elems
  // roiconv applies out = in * alpha + beta per channel. To get (x/255 - mean)/std:
  //   alpha = 1/(255*std),  beta = -mean/std.
  for (int c = 0; c < param_.num_camera; ++c) {
    roiconv::Task task;
    task.x0 = 0;
    task.y0 = 0;
    task.x1 = param_.input_width;
    task.y1 = param_.input_height;
    task.input_planes[0] = image_planes[c];
    task.input_planes[1] = nullptr;
    task.input_planes[2] = nullptr;
    task.input_width = param_.input_width;
    task.input_height = param_.input_height;
    task.input_stride = param_.input_width * 3;  // RGB packed
    task.output = output_ + c * stride_per_cam;
    task.output_width = param_.output_width;
    task.output_height = param_.output_height;
    for (int k = 0; k < 3; ++k) {
      task.alpha[k] = 1.0f / (255.0f * param_.std[k]);
      task.beta[k] = -param_.mean[k] / param_.std[k];
      task.fillcolor[k] = 0;
    }
    // Stretch-resize the full frame to the model input. If the training feature
    // builder used aspect-preserving resize+crop, swap to center_resize_affine()
    // and mirror the same crop here (one-time calibration vs a real sample).
    task.resize_affine();
    roi_->add(task);
  }

  bool ok = roi_->run(param_.input_format, roiconv::OutputDType::Float16, roiconv::OutputFormat::CHW_RGB,
                      param_.interpolation, stream, /*sync=*/false, /*clear=*/true);
  Asserts(ok, "roiconv run failed");
  return output_;
}

}  // namespace gtrs
