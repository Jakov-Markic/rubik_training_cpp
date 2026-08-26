#pragma once

#include <vector>

#include <torch/torch.h>

#include "rubik/modules.hpp"

// A YOLOv8n-pose-shaped single-stage detector: one class ("cube"), 8 keypoints,
// backbone/neck/head channel widths matching yolov8n-pose.yaml's n-scale (width
// multiple 0.25, depth multiple 0.33 -> 16/32/64/128/256 channels). See the plan doc
// for the full architecture write-up. Not a byte-exact port of ultralytics' model --
// architecturally faithful (same module graph shape), independently reimplemented.

namespace rubik {

constexpr int kNumClasses = 1;
constexpr int kNumKeypoints = 8;
constexpr int kRegMax = 16;

// Raw per-FPN-level outputs, before DFL/sigmoid decode -- what the loss consumes.
struct PoseRawOutput {
    std::vector<torch::Tensor> box;  // per level: [B, 4*reg_max, H, W]
    std::vector<torch::Tensor> cls;  // per level: [B, num_classes, H, W]
    std::vector<torch::Tensor> kpt;  // per level: [B, num_keypoints*3, H, W]
    std::vector<int64_t> strides;    // {8, 16, 32}
};

struct YoloPose8ptImpl : torch::nn::Module {
    YoloPose8ptImpl();
    PoseRawOutput forward(const torch::Tensor& x);

    // Backbone
    Conv stem{nullptr}, down1{nullptr}, down2{nullptr}, down3{nullptr}, down4{nullptr};
    C2f c2f_p2{nullptr}, c2f_p3{nullptr}, c2f_p4{nullptr}, c2f_p5{nullptr};
    SPPF sppf{nullptr};

    // Neck (PAN-FPN)
    C2f neck_c2f1{nullptr}, neck_c2f2{nullptr}, neck_c2f3{nullptr}, neck_c2f4{nullptr};
    Conv neck_down1{nullptr}, neck_down2{nullptr};

    // Head: independent weights per level (stride 8/16/32), each a 3-layer branch.
    DFL dfl{nullptr};
    torch::nn::Sequential box_head0{nullptr}, box_head1{nullptr}, box_head2{nullptr};
    torch::nn::Sequential cls_head0{nullptr}, cls_head1{nullptr}, cls_head2{nullptr};
    torch::nn::Sequential kpt_head0{nullptr}, kpt_head1{nullptr}, kpt_head2{nullptr};
};
TORCH_MODULE(YoloPose8pt);

// Anchor points (grid-cell units, offset 0.5) and matching per-anchor stride, flattened
// and concatenated across all FPN levels. Order matches PoseRawOutput's flattened order
// (level 0 first, row-major H then W).
struct Anchors {
    torch::Tensor points;   // [A, 2] (x, y) in grid-cell units
    torch::Tensor strides;  // [A, 1]
};
Anchors make_anchors(const std::vector<torch::Tensor>& feature_maps, const std::vector<int64_t>& strides,
                      double offset = 0.5);

// Decodes DFL-expectation ltrb distances (grid-cell units) + anchor points into an
// xyxy box, still in grid-cell units (multiply by stride for input-pixel units).
torch::Tensor dist2bbox(const torch::Tensor& ltrb, const torch::Tensor& anchor_points);

// Flattens a per-level [B, C, H, W] list into [B, A_total, C] (concat over levels).
torch::Tensor flatten_levels(const std::vector<torch::Tensor>& per_level);

}  // namespace rubik
