// SPDX-License-Identifier: MIT
// Host-side decoders for the GTRS-BEVFusion heads (DETR detection, BEV seg,
// planner). The heads are tiny, so decoding on the host after a fp16->fp32
// copy is simpler and cheaper than custom kernels.
#ifndef __GTRS_HEADS_HPP__
#define __GTRS_HEADS_HPP__

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gtrs {

// Class order matches GTRSBevfusionConfig.detection_class_names; index K = background.
static const std::vector<std::string> kDetClassNames = {"vehicle", "pedestrian", "bicycle", "traffic_cone",
                                                         "barrier"};

struct DetBox {
  float x = 0, y = 0;        // metric ego-frame meters (already tanh*32 in the head)
  float heading = 0;         // radians (already tanh*pi)
  float length = 0, width = 0;
  float score = 0;
  int cls = -1;
};

// states: [Q,5] = (x, y, heading, length, width); logits: [Q,K+1] (last = bg).
// Softmax over K+1; foreground score = prob of best non-bg class. Greedy
// center-distance NMS (meters) like the python viz.
std::vector<DetBox> decode_detections(const std::vector<float>& states, const std::vector<float>& logits, int num_queries,
                                      int num_classes, float score_thresh = 0.2f, float nms_dist = 2.0f);

// logits: [C,H,W] row-major. Returns argmax class id per pixel [H*W].
std::vector<uint8_t> decode_segmentation(const std::vector<float>& logits, int num_classes, int height, int width);

struct PlanResult {
  std::vector<std::array<float, 3>> poses;  // (x, y, heading) per pose
  float best_score = 0;
};

// trajectory: [P,3]; scores: [V] vocab logits (best score reported for confidence).
PlanResult decode_plan(const std::vector<float>& trajectory, int num_poses, const std::vector<float>& scores);

}  // namespace gtrs

#endif  // __GTRS_HEADS_HPP__
