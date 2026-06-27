// SPDX-License-Identifier: MIT
// Entry point for the GTRS-BEVFusion C++ deploy pipeline.
//
//   ./gtrs_bevfusion <data-dir> <model-tag> [fp16|int8]
//
// data-dir must contain (same format as the fork's example-data):
//   camera2lidar.tensor camera_intrinsics.tensor lidar2image.tensor
//   img_aug_matrix.tensor points.tensor  0-FRONT.jpg .. 5-BACK_RIGHT.jpg
//   (optional) status.tensor  -> ego status [24] float; zeros if absent.
#include <cuda_runtime.h>
#include <NvInferPlugin.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/visualize.hpp"
#include "cuosd.h"
#include "gtrs/gtrs_pipeline.hpp"

#define VIZ_FONT "tool/simhei.ttf"

static std::vector<unsigned char*> load_images(const std::string& root, int* w, int* h) {
  const char* names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                         "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};
  std::vector<unsigned char*> imgs;
  for (int i = 0; i < 6; ++i) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", root.c_str(), names[i]);
    int c;
    imgs.push_back(stbi_load(path, w, h, &c, 3));  // force RGB
  }
  return imgs;
}

// Detection class names + colors (must match gtrs::kDetClassNames ordering) so the
// cuOSD artists print our labels instead of the default nuScenes set.
static std::vector<nv::NameAndColor> gtrs_det_classes() {
  return {{"vehicle", 255, 158, 0}, {"pedestrian", 0, 0, 230}, {"bicycle", 220, 20, 60},
          {"traffic_cone", 47, 200, 200}, {"barrier", 112, 128, 144}};
}

// BEV semantic palette (7 NAVSIM classes); index 0 = background (kept dark).
static const unsigned char kSegPalette[8][3] = {
    {20, 20, 20},    {90, 90, 90},    {0, 150, 220}, {60, 200, 60},
    {230, 200, 40},  {220, 90, 40},   {200, 40, 200}, {40, 220, 200}};

// ---- host-side trajectory overlay (matches the python viz: red polyline+dots) ----
static float half_to_float(uint16_t b) {
  uint32_t sign = (b & 0x8000u) << 16, exp = (b >> 10) & 0x1Fu, man = b & 0x3FFu, f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; --exp; }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  memcpy(&out, &f, 4);
  return out;
}

// 5th-percentile of lidar z (ego frame) -> ground plane, like the python viz.
static float compute_ground_z(const nv::Tensor& pts_host) {
  int P = static_cast<int>(pts_host.size(0)), F = static_cast<int>(pts_host.size(1));
  const uint16_t* p = reinterpret_cast<const uint16_t*>(pts_host.ptr<unsigned short>());
  std::vector<float> z(P);
  for (int i = 0; i < P; ++i) z[i] = half_to_float(p[i * F + 2]);
  if (z.empty()) return -1.8f;
  size_t k = static_cast<size_t>(0.05 * z.size());
  std::nth_element(z.begin(), z.begin() + k, z.end());
  return z[k];
}

static inline void put_px(std::vector<unsigned char>& im, int w, int h, int x, int y, const unsigned char c[3]) {
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  size_t o = (static_cast<size_t>(y) * w + x) * 3;
  im[o] = c[0]; im[o + 1] = c[1]; im[o + 2] = c[2];
}
static void plot_disk(std::vector<unsigned char>& im, int w, int h, int cx, int cy, int r, const unsigned char c[3]) {
  for (int dy = -r; dy <= r; ++dy)
    for (int dx = -r; dx <= r; ++dx)
      if (dx * dx + dy * dy <= r * r) put_px(im, w, h, cx + dx, cy + dy, c);
}
static void plot_line(std::vector<unsigned char>& im, int w, int h, float x0, float y0, float x1, float y1, int r,
                      const unsigned char c[3], bool dashed = false) {
  float dx = x1 - x0, dy = y1 - y0;
  int n = std::max(1, static_cast<int>(std::sqrt(dx * dx + dy * dy)));
  // dotted style: short "on" run then a gap wide enough to read past the disk radius.
  const int on = std::max(3, r * 2), off = std::max(8, r * 5), period = on + off;
  for (int i = 0; i <= n; ++i) {
    if (dashed && (i % period) >= on) continue;
    float t = static_cast<float>(i) / n;
    plot_disk(im, w, h, static_cast<int>(std::lround(x0 + dx * t)), static_cast<int>(std::lround(y0 + dy * t)), r, c);
  }
}
// pts: (px, py, valid) in image space; connect consecutive valid samples.
static void draw_polyline(std::vector<unsigned char>& im, int w, int h, const std::vector<std::array<float, 3>>& pts,
                          const unsigned char c[3], int lw, int dotr, bool dashed = false) {
  for (size_t i = 1; i < pts.size(); ++i)
    if (pts[i - 1][2] > 0 && pts[i][2] > 0)
      plot_line(im, w, h, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], lw, c, dashed);
  for (const auto& p : pts)
    if (p[2] > 0) plot_disk(im, w, h, static_cast<int>(std::lround(p[0])), static_cast<int>(std::lround(p[1])), dotr, c);
}

// ---- box drawing (GT solid, predictions dashed) ----
static const unsigned char* det_cls_rgb(int cls) {
  static const unsigned char t[5][3] = {{255, 158, 0}, {0, 0, 230}, {220, 20, 60}, {47, 200, 200}, {112, 128, 144}};
  return t[(cls < 0 || cls >= 5) ? 0 : cls];
}

// 8 box corners in ego frame: base rectangle (rotated by heading) at z0 and z0+h.
static void box_corners3d(const gtrs::DetBox& b, float z0, float zh, std::array<std::array<float, 3>, 8>& out) {
  float hl = b.length * 0.5f, hw = b.width * 0.5f, ch = std::cos(b.heading), sh = std::sin(b.heading);
  const float lc[4][2] = {{hl, hw}, {hl, -hw}, {-hl, -hw}, {-hl, hw}};
  for (int i = 0; i < 4; ++i) {
    float wx = b.x + lc[i][0] * ch - lc[i][1] * sh, wy = b.y + lc[i][0] * sh + lc[i][1] * ch;
    out[i] = {wx, wy, z0};
    out[i + 4] = {wx, wy, zh};
  }
}

// draw the given edges of projected corners; skip the box if any used corner is invalid.
static void draw_edges(std::vector<unsigned char>& im, int w, int h, const std::vector<std::array<float, 3>>& cor,
                       const int (*edges)[2], int ne, const unsigned char c[3], int lw, bool dashed) {
  for (int e = 0; e < ne; ++e)
    if (cor[edges[e][0]][2] <= 0 || cor[edges[e][1]][2] <= 0) return;
  for (int e = 0; e < ne; ++e)
    plot_line(im, w, h, cor[edges[e][0]][0], cor[edges[e][0]][1], cor[edges[e][1]][0], cor[edges[e][1]][1], lw, c, dashed);
}

static const int kEdges3D[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                                    {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
static const int kEdges2D[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

// ego (x,y) -> BEVArtist pixel (lidar2image_bev * rot_z(10deg)), matching render_bev.
// Signature matches seg_px so both can be passed to overlay_topdown (square: w==h==side).
static std::array<float, 3> bev_px(float x, float y, int side, int /*unused*/) {
  const float th = 10.0f * 3.14159265f / 180.0f, ct = std::cos(th), st = std::sin(th);
  const float s = (side * 0.5f) * (0.075f / 0.08f) / 50.0f, c0 = side * 0.5f;
  return {s * (x * ct - y * st) + c0, -s * (x * st + y * ct) + c0, 1.0f};
}
// ego (x,y) -> seg pixel (row=x/ps+H/2, col=y/ps+W/2).
static std::array<float, 3> seg_px(float x, float y, int w, int h) {
  const float ps = 64.0f / w;
  return {y / ps + w * 0.5f, x / ps + h * 0.5f, 1.0f};
}

// project an ego-frame point through a row-major 4x4 (lidar2image) onto a camera.
static std::array<float, 3> proj_cam(const float* m, float x, float y, float z) {
  float u = m[0] * x + m[1] * y + m[2] * z + m[3];
  float v = m[4] * x + m[5] * y + m[6] * z + m[7];
  float wt = m[8] * x + m[9] * y + m[10] * z + m[11];
  if (wt <= 0.5f) return {0, 0, 0};  // behind the camera
  return {u / wt, v / wt, 1.0f};
}

static std::vector<unsigned char> colorize_seg(const std::vector<uint8_t>& seg, int h, int w) {
  std::vector<unsigned char> rgb(static_cast<size_t>(h) * w * 3);
  for (int i = 0; i < h * w; ++i) {
    int c = seg[i] & 7;
    rgb[i * 3 + 0] = kSegPalette[c][0];
    rgb[i * 3 + 1] = kSegPalette[c][1];
    rgb[i * 3 + 2] = kSegPalette[c][2];
  }
  return rgb;
}

// Resize a host RGB image into the destination rect [x0,y0,x1,y1] of the scene canvas.
static void blit(const std::shared_ptr<nv::SceneArtist>& scene, const unsigned char* host_rgb, int sw, int sh,
                 int x0, int y0, int x1, int y1, cudaStream_t stream) {
  nv::Tensor t(std::vector<int>{sh, sw, 3}, nv::DataType::UInt8);
  t.copy_from_host(host_rgb, stream);
  scene->resize_to(t.ptr<unsigned char>(), x0, y0, x1, y1, sw, sw * 3, sh, 1.0f, stream);
  checkRuntime(cudaStreamSynchronize(stream));
}

// Render the top-down BEV lidar points + ego marker to a fresh square host RGB image.
// Boxes + trajectories are overlaid afterwards on the host (overlay_bev) so we can
// style GT (solid) vs predictions (dashed), which cuOSD lines can't express.
static std::vector<unsigned char> render_bev(const nvtype::half* pts_dev, int num_points, int side,
                                             cudaStream_t stream) {
  nv::Tensor canvas(std::vector<int>{side, side, 3}, nv::DataType::UInt8);
  canvas.memset(0x00, stream);

  nv::BEVArtistParameter bp;
  bp.image_width = side;
  bp.image_height = side;
  bp.image_stride = side * 3;
  bp.classes = gtrs_det_classes();
  bp.rotate_x = 0.0f;
  bp.cx = side * 0.5f;
  bp.cy = side * 0.5f;
  bp.norm_size = (side * 0.5f) * (0.075f / 0.08f);  // pixels-per-meter scale (calibrated to BEV grid)
  auto bev = nv::create_bev_artist(bp);
  bev->draw_lidar_points(pts_dev, num_points);
  bev->draw_ego();
  bev->apply(canvas.ptr<unsigned char>(), stream);
  checkRuntime(cudaStreamSynchronize(stream));
  auto host = canvas.to_host(stream);
  std::vector<unsigned char> out(static_cast<size_t>(side) * side * 3);
  memcpy(out.data(), host.ptr<unsigned char>(), out.size());
  return out;
}

// Overlay boxes (GT solid / pred dashed, color=class) + trajectories (GT white solid /
// pred red dashed) on a host image, using a caller-supplied ego->pixel projector.
static void overlay_topdown(std::vector<unsigned char>& im, int w, int h,
                            const std::vector<gtrs::DetBox>& gt, const std::vector<gtrs::DetBox>& pred,
                            const std::vector<std::array<float, 3>>& gt_traj,
                            const std::vector<std::array<float, 3>>& pred_traj,
                            std::array<float, 3> (*proj)(float, float, int, int), int side_w, int side_h, int lw,
                            int dotr) {
  static const unsigned char gt_green[3] = {0, 255, 0};  // GT drawn green/solid so it stays
  auto boxes = [&](const std::vector<gtrs::DetBox>& bs, bool dashed, const unsigned char* fixed) {
    for (const auto& b : bs) {
      std::array<std::array<float, 3>, 8> c3;
      box_corners3d(b, 0, 0, c3);
      std::vector<std::array<float, 3>> fp(4);
      for (int i = 0; i < 4; ++i) fp[i] = proj(c3[i][0], c3[i][1], side_w, side_h);
      draw_edges(im, w, h, fp, kEdges2D, 4, fixed ? fixed : det_cls_rgb(b.cls), lw, dashed);
    }
  };
  boxes(gt, false, gt_green);  // GT: green solid; distinguishable even when overlapping a pred
  boxes(pred, true, nullptr);  // predictions: class-colored, dashed
  auto pl = [&](const std::vector<std::array<float, 3>>& t, const unsigned char c[3], bool dashed) {
    std::vector<std::array<float, 3>> o(t.size());
    for (size_t i = 0; i < t.size(); ++i) o[i] = proj(t[i][0], t[i][1], side_w, side_h);
    draw_polyline(im, w, h, o, c, lw, dotr, dashed);
  };
  const unsigned char green[3] = {0, 255, 0}, red[3] = {255, 40, 40};
  pl(gt_traj, green, false);   // GT trajectory: solid green
  pl(pred_traj, red, true);    // predicted trajectory: dashed red
}

// Overlay 3D boxes + trajectories on one camera image (project ego->pixel via lidar2image).
static void overlay_cam(std::vector<unsigned char>& im, int w, int h, const float* l2i,
                        const std::vector<gtrs::DetBox>& gt, const std::vector<gtrs::DetBox>& pred,
                        const std::vector<std::array<float, 3>>& gt_traj,
                        const std::vector<std::array<float, 3>>& pred_traj, float z0, float zh) {
  static const unsigned char gt_green[3] = {0, 255, 0};  // GT drawn green/solid so it stays
  auto boxes = [&](const std::vector<gtrs::DetBox>& bs, bool dashed, const unsigned char* fixed) {
    for (const auto& b : bs) {
      std::array<std::array<float, 3>, 8> c3;
      box_corners3d(b, z0, zh, c3);
      std::vector<std::array<float, 3>> pc(8);
      for (int i = 0; i < 8; ++i) pc[i] = proj_cam(l2i, c3[i][0], c3[i][1], c3[i][2]);
      draw_edges(im, w, h, pc, kEdges3D, 12, fixed ? fixed : det_cls_rgb(b.cls), 2, dashed);
    }
  };
  boxes(gt, false, gt_green);  // GT: green solid
  boxes(pred, true, nullptr);  // predictions: class-colored, dashed
  auto pl = [&](const std::vector<std::array<float, 3>>& t, const unsigned char c[3], bool dashed) {
    std::vector<std::array<float, 3>> o(t.size());
    for (size_t i = 0; i < t.size(); ++i) o[i] = proj_cam(l2i, t[i][0], t[i][1], z0);
    draw_polyline(im, w, h, o, c, 2, 3, dashed);
  };
  const unsigned char green[3] = {0, 255, 0}, red[3] = {255, 40, 40};
  pl(gt_traj, green, false);   // GT trajectory: solid green
  pl(pred_traj, red, true);    // predicted trajectory: dashed red
}

// Load "cls x y heading length width [score]" rows (GT has no score column).
static std::vector<gtrs::DetBox> load_boxes(const std::string& path) {
  std::vector<gtrs::DetBox> out;
  std::ifstream f(path);
  if (!f.good()) return out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    gtrs::DetBox b;
    if (!(ss >> b.cls >> b.x >> b.y >> b.heading >> b.length >> b.width)) continue;
    ss >> b.score;  // optional
    out.push_back(b);
  }
  return out;
}

static std::vector<std::array<float, 3>> load_traj(const std::string& path) {
  std::vector<std::array<float, 3>> out;
  std::ifstream f(path);
  if (!f.good()) return out;
  float x, y, h;
  while (f >> x >> y >> h) out.push_back({x, y, h});
  return out;
}

int main(int argc, char** argv) {
  const char* data = (argc > 1) ? argv[1] : "example-data";
  const char* tag = (argc > 2) ? argv[2] : "gtrs_bevfusion";
  const char* precision = (argc > 3) ? argv[3] : "fp16";
  const char* dump_dir = (argc > 4) ? argv[4] : nullptr;  // per-stage parity dumps
  const char* cam_bev_file = (argc > 5) ? argv[5] : nullptr;  // inject ref cam_bev, skip LSS

  checkRuntime(cudaSetDevice(0));
  // Register the standard TensorRT plugins (seg head's GroupNorm lowers to the
  // InstanceNormalization_TRT plugin).
  initLibNvInferPlugins(nullptr, "");
  cudaStream_t stream;
  checkRuntime(cudaStreamCreate(&stream));

  gtrs::PipelineParam pp;
  pp.model_dir = std::string("model/") + tag;
  pp.fp16_only = std::string(precision) != "int8";
  if (dump_dir) pp.dump_dir = dump_dir;
  if (cam_bev_file) pp.cam_bev_override = cam_bev_file;
  pp.enable_timer = true;  // per-stage GPU latency breakdown

  // calibration matrices (host float, 6x4x4)
  auto camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", data), false);
  auto camera_intrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", data), false);
  auto lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", data), false);
  auto img_aug_matrix = nv::Tensor::load(nv::format("%s/img_aug_matrix.tensor", data), false);
  if (camera2lidar.empty() || camera_intrinsics.empty() || lidar2image.empty() || img_aug_matrix.empty()) {
    printf("ERROR: missing calibration tensors in %s/\n", data);
    return -1;
  }

  pp.camera_frontend.num_camera = 6;

  // Camera input. Prefer a PRE-NORMALIZED camera.tensor ([6,3,256,704] fp16, the
  // exact PyTorch backbone input from --dump-cpp-inputs): this bypasses roiconvert
  // so the LSS/backbone/fuser/head numerics can be diffed against PyTorch without
  // any resize/normalize mismatch. Otherwise fall back to raw JPEGs + roiconvert.
  std::vector<unsigned char*> images;       // host jpeg buffers (raw path)
  std::vector<void*> dev_imgs;              // device jpeg buffers (raw path)
  std::vector<const void*> planes;
  nv::Tensor cam_pre = nv::Tensor::load(nv::format("%s/camera.tensor", data), false);
  nv::Tensor cam_pre_dev;                   // keep device buffer alive for forward
  if (!cam_pre.empty()) {
    printf("camera: pre-normalized camera.tensor [%d,%d,%d,%d] -> bypassing roiconvert\n",
           static_cast<int>(cam_pre.size(0)), static_cast<int>(cam_pre.size(1)),
           static_cast<int>(cam_pre.size(2)), static_cast<int>(cam_pre.size(3)));
    pp.use_camera_frontend = false;
    cam_pre_dev = cam_pre.to_device();
    planes = {cam_pre_dev.ptr<half>()};     // forward() reads camera_planes[0] as normed
  } else {
    int iw = 0, ih = 0;
    images = load_images(data, &iw, &ih);
    for (int i = 0; i < 6; ++i)
      if (!images[i]) {
        printf("ERROR: failed to load camera image %d from %s/\n", i, data);
        return -1;
      }
    pp.camera_frontend.input_width = iw;
    pp.camera_frontend.input_height = ih;
    pp.camera_frontend.input_format = roiconv::InputFormat::RGB;
    dev_imgs.assign(6, nullptr);
    planes.assign(6, nullptr);
    size_t img_bytes = static_cast<size_t>(iw) * ih * 3;
    for (int i = 0; i < 6; ++i) {
      checkRuntime(cudaMalloc(&dev_imgs[i], img_bytes));
      checkRuntime(cudaMemcpyAsync(dev_imgs[i], images[i], img_bytes, cudaMemcpyHostToDevice, stream));
      planes[i] = dev_imgs[i];
    }
  }

  // lidar points -> device fp16
  auto lidar_points = nv::Tensor::load(nv::format("%s/points.tensor", data), false);
  if (lidar_points.empty()) {
    printf("ERROR: missing %s/points.tensor\n", data);
    return -1;
  }
  auto pts_dev = lidar_points.to_device();
  int num_points = lidar_points.size(0);

  // ego status [24]
  float status[24] = {0};
  {
    auto st = nv::Tensor::load(nv::format("%s/status.tensor", data), false);
    if (!st.empty()) {
      auto sh = st.to_host(stream);
      int n = std::min<int>(24, static_cast<int>(sh.numel));
      for (int i = 0; i < n; ++i) status[i] = sh.ptr<float>()[i];
    } else {
      printf("WARN: no status.tensor, using zero ego status.\n");
    }
  }

  gtrs::Pipeline pipe(pp);
  pipe.update(camera2lidar.ptr<float>(), camera_intrinsics.ptr<float>(), lidar2image.ptr<float>(),
              img_aug_matrix.ptr<float>(), stream);

  // ---- warmup + latency benchmark (mirrors the fork's main.cpp timed loop) ----
  int bench_iters = 20;
  if (const char* e = std::getenv("GTRS_BENCH_ITERS")) bench_iters = std::max(1, atoi(e));

  gtrs::PipelineOutput out;
  for (int i = 0; i < 3; ++i)  // warmup (engine capture, allocator, clocks)
    out = pipe.forward(planes, pts_dev.ptr<half>(), num_points, status, stream);
  checkRuntime(cudaStreamSynchronize(stream));
  pipe.reset_timing();  // discard warmup from the GPU per-stage averages

  std::vector<double> wall_ms;
  wall_ms.reserve(bench_iters);
  for (int i = 0; i < bench_iters; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    out = pipe.forward(planes, pts_dev.ptr<half>(), num_points, status, stream);
    checkRuntime(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  pipe.report_timing();  // GPU per-stage (lidar/camera/fuser/heads)
  {  // end-to-end wall clock (includes host-side decode + device->host copies)
    std::vector<double> s = wall_ms;
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double v : s) sum += v;
    double mean = sum / s.size();
    double med = s[s.size() / 2];
    double p95 = s[std::min(s.size() - 1, (size_t)std::lround(0.95 * (s.size() - 1)))];
    printf("\n========== End-to-end latency (wall, %d iters) ==========\n", bench_iters);
    printf("  mean=%.3f ms  median=%.3f ms  p95=%.3f ms  min=%.3f ms  max=%.3f ms\n", mean, med, p95, s.front(),
           s.back());
    printf("  throughput: %.1f FPS (mean)\n", mean > 0 ? 1000.0 / mean : 0.0);
    printf("========================================================\n");
  }

  // ---- report ----
  printf("\n=== detections (%zu) ===\n", out.detections.size());
  for (const auto& b : out.detections) {
    printf("  %-12s score=%.3f  pos=(%.2f, %.2f)  hdg=%.2f  size=(L%.2f x W%.2f)\n",
           gtrs::kDetClassNames[b.cls].c_str(), b.score, b.x, b.y, b.heading, b.length, b.width);
  }
  printf("\n=== plan (score=%.3f) ===\n", out.plan.best_score);
  for (size_t i = 0; i < out.plan.poses.size(); i += 5)
    printf("  t=%2zu  (%.2f, %.2f, hdg=%.2f)\n", i, out.plan.poses[i][0], out.plan.poses[i][1], out.plan.poses[i][2]);

  // ---- dump seg + trajectory for the python visualizer ----
  {
    std::ofstream f("build/gtrs_seg_256x256_u8.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.seg.data()), out.seg.size());
  }
  {
    std::ofstream f("build/gtrs_trajectory.txt");
    for (auto& p : out.plan.poses) f << p[0] << " " << p[1] << " " << p[2] << "\n";
  }
  {  // machine-readable detections for the python multi-cam visualizer
    std::ofstream f("build/gtrs_detections.txt");
    f << "# cls x y heading length width score\n";
    for (const auto& b : out.detections)
      f << b.cls << " " << b.x << " " << b.y << " " << b.heading << " " << b.length << " " << b.width
        << " " << b.score << "\n";
  }

  // ---- ground truth (solid) vs predictions (dashed) ----
  // ground_z = 5th-percentile of lidar z (matches the python viz) so boxes/trajectory
  // sit on the road plane in the cameras instead of floating.
  float ground_z = compute_ground_z(lidar_points), box_h = 1.6f;
  if (const char* e = std::getenv("GTRS_GROUND_Z")) ground_z = atof(e);
  if (const char* e = std::getenv("GTRS_BOX_H")) box_h = atof(e);
  printf("viz: ground_z=%.2f m (lidar 5th pct), box_h=%.2f m\n", ground_z, box_h);

  const std::vector<gtrs::DetBox>& pred_boxes = out.detections;
  const std::vector<std::array<float, 3>>& pred_traj = out.plan.poses;  // (x,y,heading) per pose
  std::vector<gtrs::DetBox> gt_boxes = load_boxes(std::string(data) + "/gt_detections.txt");
  std::vector<std::array<float, 3>> gt_traj = load_traj(std::string(data) + "/gt_trajectory.txt");
  printf("viz: GT %zu boxes / %zu traj poses (solid); pred %zu boxes / %zu poses (dashed)\n", gt_boxes.size(),
         gt_traj.size(), pred_boxes.size(), pred_traj.size());

  // ---- standalone BEV render (kept for quick inspection) ----
  {
    auto bev = render_bev(pts_dev.ptr<nvtype::half>(), num_points, 1024, stream);
    overlay_topdown(bev, 1024, 1024, gt_boxes, pred_boxes, gt_traj, pred_traj, bev_px, 1024, 1024, 4, 6);
    stbi_write_jpg("build/gtrs-bev.jpg", 1024, 1024, 3, bev.data(), 100);
  }

  // ---- display images: de-normalized resized RGB (camera_rgb.tensor) or raw JPEGs ----
  // ordering matches config.camera_names = (cam_l0, cam_f0, cam_r0, cam_l1, cam_r1, cam_b0),
  // i.e. the same order as lidar2image / camera.tensor.
  std::vector<std::vector<unsigned char>> disp(6);
  int disp_w = 0, disp_h = 0;
  bool have_disp = false;
  {
    nv::Tensor rgb = nv::Tensor::load(nv::format("%s/camera_rgb.tensor", data), false);
    if (!rgb.empty()) {  // [6,H,W,3] uint8
      disp_h = static_cast<int>(rgb.size(1));
      disp_w = static_cast<int>(rgb.size(2));
      const unsigned char* p = rgb.ptr<unsigned char>();
      size_t plane = static_cast<size_t>(disp_h) * disp_w * 3;
      for (int i = 0; i < 6; ++i) disp[i].assign(p + i * plane, p + (i + 1) * plane);
      have_disp = true;
      printf("camera viz: using camera_rgb.tensor (%dx%d)\n", disp_w, disp_h);
    } else if (!images.empty()) {  // raw JPEG path
      disp_w = pp.camera_frontend.input_width;
      disp_h = pp.camera_frontend.input_height;
      size_t plane = static_cast<size_t>(disp_h) * disp_w * 3;
      for (int i = 0; i < 6; ++i) disp[i].assign(images[i], images[i] + plane);
      have_disp = true;
      printf("camera viz: using raw JPEGs (%dx%d)\n", disp_w, disp_h);
    } else {
      printf("camera viz: no camera_rgb.tensor and no JPEGs -> skipping camera panels\n");
    }
  }

  // ---- composite: 2x3 cameras (with projected boxes) over BEV-det + BEV-seg ----
  if (have_disp) {
    const int TW = 540;                                  // camera tile width
    const int TH = std::max(1, TW * disp_h / disp_w);    // keep aspect
    const int Sbev = 600;                                // bottom panel side
    const int canvas_w = std::max(3 * TW, 2 * Sbev);
    const int cam_area_h = 2 * TH;
    const int canvas_h = cam_area_h + Sbev;

    nv::SceneArtistParameter csp;
    csp.width = canvas_w;
    csp.height = canvas_h;
    csp.stride = canvas_w * 3;
    nv::Tensor comp(std::vector<int>{canvas_h, canvas_w, 3}, nv::DataType::UInt8);
    comp.memset(0x10, stream);  // dark gray background
    csp.image_device = comp.ptr<unsigned char>();
    auto scene = nv::create_scene_artist(csp);

    // cuOSD context for prediction labels (boxes/trajectory are CPU-drawn for solid/dashed)
    cuOSDContext_t osd = cuosd_context_create();

    // display grid -> tensor index: row0 [l0,f0,r0], row1 [l1,b0,r1]
    const int disp_to_tensor[6] = {0, 1, 2, 3, 5, 4};
    const int x_pad = (canvas_w - 3 * TW) / 2;
    for (int d = 0; d < 6; ++d) {
      int ti = disp_to_tensor[d];
      const float* l2i = lidar2image.ptr<float>() + ti * 16;
      // GT boxes/traj solid, predictions dashed (CPU draw on host, under the labels)
      overlay_cam(disp[ti], disp_w, disp_h, l2i, gt_boxes, pred_boxes, gt_traj, pred_traj, ground_z,
                  ground_z + box_h);

      nv::Tensor di(std::vector<int>{disp_h, disp_w, 3}, nv::DataType::UInt8);
      di.copy_from_host(disp[ti].data(), stream);

      // prediction labels (class + score) via cuOSD at each visible box's top corner
      for (const auto& b : pred_boxes) {
        std::array<std::array<float, 3>, 8> c3;
        box_corners3d(b, ground_z, ground_z + box_h, c3);
        float minx = disp_w, miny = disp_h;
        bool vis = true;
        for (int i = 0; i < 8; ++i) {
          auto p = proj_cam(l2i, c3[i][0], c3[i][1], c3[i][2]);
          if (p[2] <= 0) { vis = false; break; }
          minx = std::min(minx, p[0]);
          miny = std::min(miny, p[1]);
        }
        if (!vis) continue;
        const unsigned char* col = det_cls_rgb(b.cls);
        char title[64];
        snprintf(title, sizeof(title), "%s %.2f", gtrs::kDetClassNames[b.cls].c_str(), b.score);
        cuosd_draw_text(osd, title, 14, VIZ_FONT, static_cast<int>(minx), static_cast<int>(miny),
                        {col[0], col[1], col[2], 255}, {0, 0, 0, 180});
      }
      cuosd_apply(osd, di.ptr<unsigned char>(), nullptr, disp_w, disp_w * 3, disp_h, cuOSDImageFormat::RGB, stream);
      checkRuntime(cudaStreamSynchronize(stream));

      int r = d / 3, c = d % 3;
      int x0 = x_pad + c * TW, y0 = r * TH;
      scene->resize_to(di.ptr<unsigned char>(), x0, y0, x0 + TW, y0 + TH, disp_w, disp_w * 3, disp_h, 1.0f, stream);
      checkRuntime(cudaStreamSynchronize(stream));
    }
    cuosd_context_destroy(osd);

    // bottom-left: BEV lidar + boxes (GT solid/pred dashed) + trajectories
    int bx_pad = (canvas_w - 2 * Sbev) / 2;
    auto bevimg = render_bev(pts_dev.ptr<nvtype::half>(), num_points, Sbev, stream);
    overlay_topdown(bevimg, Sbev, Sbev, gt_boxes, pred_boxes, gt_traj, pred_traj, bev_px, Sbev, Sbev, 2, 4);
    blit(scene, bevimg.data(), Sbev, Sbev, bx_pad, cam_area_h, bx_pad + Sbev, cam_area_h + Sbev, stream);

    // bottom-right: BEV semantic + boxes + trajectories (seg pixel frame, +/-32 m -> 256)
    auto segrgb = colorize_seg(out.seg, out.seg_h, out.seg_w);
    overlay_topdown(segrgb, out.seg_w, out.seg_h, gt_boxes, pred_boxes, gt_traj, pred_traj, seg_px, out.seg_w,
                    out.seg_h, 2, 2);
    blit(scene, segrgb.data(), out.seg_w, out.seg_h, bx_pad + Sbev, cam_area_h, bx_pad + 2 * Sbev,
         cam_area_h + Sbev, stream);

    stbi_write_jpg("build/gtrs_cpp_viz.jpg", canvas_w, canvas_h, 3, comp.to_host(stream).ptr(), 95);
    printf("Saved build/gtrs_cpp_viz.jpg (%dx%d: GT solid / pred dashed boxes+traj on 6 cams, BEV, BEV-seg)\n",
           canvas_w, canvas_h);
  }

  printf("\nSaved build/gtrs-bev.jpg, build/gtrs_seg_256x256_u8.bin, build/gtrs_trajectory.txt\n");

  for (size_t i = 0; i < dev_imgs.size(); ++i) cudaFree(dev_imgs[i]);
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);
  return 0;
}
