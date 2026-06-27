// SPDX-License-Identifier: MIT
#include "gtrs/gtrs_heads.hpp"

#include <algorithm>
#include <cmath>

namespace gtrs {

std::vector<DetBox> decode_detections(const std::vector<float>& states, const std::vector<float>& logits, int Q,
                                      int K, float score_thresh, float nms_dist) {
  const int S = 5;          // x,y,heading,length,width
  const int C = K + 1;      // +background
  std::vector<DetBox> boxes;
  boxes.reserve(Q);
  for (int q = 0; q < Q; ++q) {
    const float* lg = logits.data() + q * C;
    // softmax
    float mx = lg[0];
    for (int c = 1; c < C; ++c) mx = std::max(mx, lg[c]);
    float sum = 0.f;
    std::vector<float> prob(C);
    for (int c = 0; c < C; ++c) {
      prob[c] = std::exp(lg[c] - mx);
      sum += prob[c];
    }
    for (int c = 0; c < C; ++c) prob[c] /= sum;

    int best = 0;
    for (int c = 1; c < K; ++c)
      if (prob[c] > prob[best]) best = c;
    float score = prob[best];
    if (score < score_thresh) continue;

    const float* st = states.data() + q * S;
    DetBox b;
    b.x = st[0];
    b.y = st[1];
    b.heading = st[2];
    b.length = st[3];
    b.width = st[4];
    b.score = score;
    b.cls = best;
    boxes.push_back(b);
  }

  // greedy center-distance NMS
  std::sort(boxes.begin(), boxes.end(), [](const DetBox& a, const DetBox& b) { return a.score > b.score; });
  std::vector<DetBox> kept;
  std::vector<char> removed(boxes.size(), 0);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (removed[i]) continue;
    kept.push_back(boxes[i]);
    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (removed[j]) continue;
      float dx = boxes[i].x - boxes[j].x;
      float dy = boxes[i].y - boxes[j].y;
      if (std::sqrt(dx * dx + dy * dy) < nms_dist) removed[j] = 1;
    }
  }
  return kept;
}

std::vector<uint8_t> decode_segmentation(const std::vector<float>& logits, int C, int H, int W) {
  std::vector<uint8_t> out(static_cast<size_t>(H) * W, 0);
  const int HW = H * W;
  for (int p = 0; p < HW; ++p) {
    int best = 0;
    float bestv = logits[p];  // class 0 plane
    for (int c = 1; c < C; ++c) {
      float v = logits[static_cast<size_t>(c) * HW + p];
      if (v > bestv) {
        bestv = v;
        best = c;
      }
    }
    out[p] = static_cast<uint8_t>(best);
  }
  return out;
}

PlanResult decode_plan(const std::vector<float>& trajectory, int P, const std::vector<float>& scores) {
  PlanResult r;
  r.poses.reserve(P);
  for (int i = 0; i < P; ++i) {
    r.poses.push_back({trajectory[i * 3 + 0], trajectory[i * 3 + 1], trajectory[i * 3 + 2]});
  }
  r.best_score = scores.empty() ? 0.f : *std::max_element(scores.begin(), scores.end());
  return r;
}

}  // namespace gtrs
