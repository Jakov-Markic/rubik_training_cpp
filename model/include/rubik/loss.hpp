#pragma once

#include <torch/torch.h>

#include "rubik/yolo_pose_8pt.hpp"

// Loss for the single-object-per-image cube pose dataset. Every training image has
// exactly one GT cube, which removes the need for ultralytics' full task-aligned
// assigner (whose complexity mostly comes from resolving *competition between multiple
// GT boxes* for the same anchor -- see the plan doc's "Loss & assigner" section):
//
//   1. Candidate anchors = grid points whose center falls inside the GT box.
//   2. Alignment metric = pred_cls_sigmoid^0.5 * pred_iou^6 per candidate.
//   3. Top-k candidates by that metric become positives (no cross-GT conflict to
//      resolve, since there is only ever one GT per image).
//
// cls uses hard binary targets (positive anchors = 1, everything else = 0) rather than
// ultralytics' soft alignment-metric-scaled targets -- a deliberate simplification
// (documented, not an oversight): it avoids a class of soft-target-normalization bugs
// for a fraction of the training-quality upside, given this isn't claiming ultralytics
// parity in the first place.

namespace rubik {

struct LossWeights {
    double box = 7.5;
    double cls = 0.5;
    double dfl = 1.5;
    double kpt_pos = 12.0;
    double kpt_vis = 1.0;
};

struct LossOutput {
    torch::Tensor total;
    torch::Tensor box, cls, dfl, kpt;  // scalar components, for logging
};

// targets: [B, 5 + 8*3] = cx,cy,w,h,(unused), then 8*(x,y,v), all normalized [0,1] to
// imgsz. imgsz is the square input resolution the model/dataset were built at.
LossOutput compute_loss(const PoseRawOutput& raw, const torch::Tensor& targets, int imgsz,
                         const LossWeights& weights = {});

}  // namespace rubik
