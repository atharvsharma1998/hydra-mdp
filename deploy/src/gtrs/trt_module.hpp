// SPDX-License-Identifier: MIT
// Generic TensorRT module wrapper for the GTRS-BEVFusion deploy pipeline.
//
// The fork's camera Backbone/VTransform wrappers bind by hard-coded tensor names
// (img/depth/camera_feature/..., feat_in/feat_out) which match our exported
// graphs, so those are reused as-is. Our fuser + 3 heads use different binding
// names (camera_bev/lidar_bev/fenv, scores/trajectory, agent_states/..., etc.),
// so they run through this small name-agnostic wrapper: it allocates one device
// buffer per output from the engine's static dims and runs enqueueV3 by name.
#ifndef __GTRS_TRT_MODULE_HPP__
#define __GTRS_TRT_MODULE_HPP__

#include <cuda_fp16.h>

#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/check.hpp"
#include "common/tensorrt.hpp"

namespace gtrs {

class TrtModule {
 public:
  explicit TrtModule(const std::string& plan_path) {
    engine_ = TensorRT::load(plan_path);
    Assertf(engine_ != nullptr, "failed to load engine: %s", plan_path.c_str());
    for (int i = 0; i < engine_->num_bindings(); ++i) {
      if (engine_->is_input(i)) continue;
      auto dims = engine_->static_dims(i);
      int n = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
      // binding name lookup by reverse map: TensorRT::Engine has no name(i), so
      // we record outputs by walking known output names provided via outputs().
      out_dims_by_index_[i] = dims;
      out_numel_by_index_[i] = n;
    }
  }

  // Register the output tensor names (in any order); allocates fp16 device buffers.
  void declare_outputs(const std::vector<std::string>& names) {
    for (const auto& name : names) {
      int idx = engine_->index(name);
      Assertf(idx >= 0 && !engine_->is_input(name.c_str()), "not an output binding: %s", name.c_str());
      auto dims = engine_->static_dims(name);
      int n = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
      void* buf = nullptr;
      checkRuntime(cudaMalloc(&buf, n * sizeof(half)));
      outputs_[name] = {buf, n, dims};
    }
  }

  // inputs: name -> device fp16 pointer. Runs the engine on stream.
  void forward(const std::unordered_map<std::string, const void*>& inputs, void* stream) {
    std::unordered_map<std::string, const void*> bindings = inputs;
    for (auto& kv : outputs_) bindings[kv.first] = kv.second.ptr;
    bool ok = engine_->forward(bindings, stream);
    Asserts(ok, "TensorRT forward failed");
  }

  const half* output(const std::string& name) const {
    auto it = outputs_.find(name);
    Assertf(it != outputs_.end(), "output not declared: %s", name.c_str());
    return reinterpret_cast<const half*>(it->second.ptr);
  }
  int output_numel(const std::string& name) const { return outputs_.at(name).numel; }
  const std::vector<int>& output_dims(const std::string& name) const { return outputs_.at(name).dims; }

  // Copy an output to a host float vector (fp16 -> fp32 on host).
  void copy_output_to_host(const std::string& name, std::vector<float>& host, void* stream) {
    int n = output_numel(name);
    std::vector<half> tmp(n);
    checkRuntime(cudaMemcpyAsync(tmp.data(), outputs_.at(name).ptr, n * sizeof(half),
                                 cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)));
    checkRuntime(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    host.resize(n);
    for (int i = 0; i < n; ++i) host[i] = __half2float(tmp[i]);
  }

  ~TrtModule() {
    for (auto& kv : outputs_) cudaFree(kv.second.ptr);
  }

 private:
  struct OutBuf {
    void* ptr = nullptr;
    int numel = 0;
    std::vector<int> dims;
  };
  std::shared_ptr<TensorRT::Engine> engine_;
  std::unordered_map<std::string, OutBuf> outputs_;
  std::unordered_map<int, std::vector<int>> out_dims_by_index_;
  std::unordered_map<int, int> out_numel_by_index_;
};

}  // namespace gtrs

#endif  // __GTRS_TRT_MODULE_HPP__
