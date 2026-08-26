#include "rubik/loss.hpp"

#include <cmath>
#include <vector>

#include "rubik/modules.hpp"

namespace rubik {

namespace {

// [N,4] xyxy pairwise (elementwise, N vs N) IoU.
torch::Tensor bbox_iou_xyxy(const torch::Tensor& b1, const torch::Tensor& b2) {
    auto x1 = torch::max(b1.select(1, 0), b2.select(1, 0));
    auto y1 = torch::max(b1.select(1, 1), b2.select(1, 1));
    auto x2 = torch::min(b1.select(1, 2), b2.select(1, 2));
    auto y2 = torch::min(b1.select(1, 3), b2.select(1, 3));
    auto inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0);
    auto area1 = (b1.select(1, 2) - b1.select(1, 0)).clamp_min(0) * (b1.select(1, 3) - b1.select(1, 1)).clamp_min(0);
    auto area2 = (b2.select(1, 2) - b2.select(1, 0)).clamp_min(0) * (b2.select(1, 3) - b2.select(1, 1)).clamp_min(0);
    return inter / (area1 + area2 - inter + 1e-7);
}

torch::Tensor ciou_xyxy(const torch::Tensor& pred, const torch::Tensor& target) {
    auto iou = bbox_iou_xyxy(pred, target);
    auto px1 = pred.select(1, 0), py1 = pred.select(1, 1), px2 = pred.select(1, 2), py2 = pred.select(1, 3);
    auto tx1 = target.select(1, 0), ty1 = target.select(1, 1), tx2 = target.select(1, 2), ty2 = target.select(1, 3);
    auto pcx = (px1 + px2) / 2, pcy = (py1 + py2) / 2;
    auto tcx = (tx1 + tx2) / 2, tcy = (ty1 + ty2) / 2;
    auto pw = (px2 - px1).clamp_min(1e-7), ph = (py2 - py1).clamp_min(1e-7);
    auto tw = (tx2 - tx1).clamp_min(1e-7), th = (ty2 - ty1).clamp_min(1e-7);
    auto cw = torch::max(px2, tx2) - torch::min(px1, tx1);
    auto ch = torch::max(py2, ty2) - torch::min(py1, ty1);
    auto c2 = cw.pow(2) + ch.pow(2) + 1e-7;
    auto rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2);
    auto v = (4.0 / (M_PI * M_PI)) * torch::pow(torch::atan(tw / th) - torch::atan(pw / ph), 2);
    torch::Tensor alpha;
    {
        torch::NoGradGuard no_grad;
        alpha = v / (v - iou + (1.0 + 1e-7));
    }
    return iou - (rho2 / c2 + v * alpha);
}

// pred: [P,4,reg_max] logits, target: [P,4] continuous ltrb in grid-cell units.
torch::Tensor dfl_ce_loss(const torch::Tensor& pred, const torch::Tensor& target, int reg_max) {
    auto tl = target.floor();
    auto tr = tl + 1;
    auto wl = tr - target;
    auto wr = target - tl;
    auto tl_idx = tl.clamp(0, reg_max - 1).to(torch::kLong).reshape({-1});
    auto tr_idx = tr.clamp(0, reg_max - 1).to(torch::kLong).reshape({-1});

    auto pred_flat = pred.reshape({-1, reg_max});
    namespace F = torch::nn::functional;
    auto ce_l = F::cross_entropy(pred_flat, tl_idx, F::CrossEntropyFuncOptions().reduction(torch::kNone));
    auto ce_r = F::cross_entropy(pred_flat, tr_idx, F::CrossEntropyFuncOptions().reduction(torch::kNone));
    auto loss = ce_l * wl.reshape({-1}) + ce_r * wr.reshape({-1});
    return loss.mean();
}

// Softmax-weighted expectation over reg_max bins (stateless equivalent of DFLImpl).
// box_bca: [B, 4*reg_max, A] -> [B, 4, A] ltrb, grid-cell units.
torch::Tensor dfl_decode(const torch::Tensor& box_bca, int reg_max) {
    auto sizes = box_bca.sizes();
    int64_t B = sizes[0], A = sizes[2];
    auto y = box_bca.view({B, 4, reg_max, A}).transpose(1, 2);  // [B, reg_max, 4, A]
    y = torch::softmax(y, 1);
    auto weight = torch::arange(reg_max, y.options()).view({1, reg_max, 1, 1});
    auto expectation = (y * weight).sum(1);  // [B, 4, A]
    return expectation;
}

}  // namespace

LossOutput compute_loss(const PoseRawOutput& raw, const torch::Tensor& targets_in, int imgsz,
                         const LossWeights& w) {
    auto device = raw.box[0].device();
    // Read every GT scalar from one CPU-side copy of targets, not `.item<double>()` calls
    // on the GPU tensor: each `.item<>()` on a CUDA tensor forces a blocking device sync,
    // and the per-image loop below used to do ~28 of them (4 for the box, 3*8 for the
    // keypoints) -- at batch=8 that's ~224 blocking syncs per training step, which was the
    // actual bottleneck (a WDDM CUDA kernel-launch/sync round-trip costs far more than the
    // tiny compute this model does per anchor). A single .to(kCPU) copy up front removes
    // essentially all of it; the one sync that's inherent to the algorithm itself is
    // torch::nonzero() below (CUDA has to know the match count to size its output, so that
    // one blocking sync per image can't be avoided without a fundamentally different -
    // padded/masked - assignment scheme).
    torch::Tensor targets_cpu = targets_in.to(torch::kCPU).contiguous();
    auto targets_acc = targets_cpu.accessor<float, 2>();
    int64_t B = raw.box[0].size(0);

    auto box_raw = flatten_levels(raw.box);              // [B, A, 4*reg_max]
    auto cls_raw = flatten_levels(raw.cls).squeeze(-1);   // [B, A]  (nc == 1)
    auto kpt_raw = flatten_levels(raw.kpt);               // [B, A, 24]

    Anchors anchors = make_anchors(raw.box, raw.strides);
    auto anchor_points = anchors.points.to(device);   // [A, 2] grid units
    auto stride_tensor = anchors.strides.to(device);  // [A, 1]
    int64_t A = anchor_points.size(0);

    auto box_for_dfl = box_raw.transpose(1, 2).contiguous();  // [B, 4*reg_max, A]
    auto ltrb_grid = dfl_decode(box_for_dfl, kRegMax).transpose(1, 2);  // [B, A, 4]
    auto pred_bbox_grid = dist2bbox(ltrb_grid, anchor_points);          // [B, A, 4] xyxy, grid units
    auto pred_bbox_px = pred_bbox_grid * stride_tensor;                 // [B, A, 4] xyxy, pixel units

    auto kpt_view = kpt_raw.view({B, A, kNumKeypoints, 3});
    auto raw_xy = kpt_view.slice(-1, 0, 2);                                       // [B, A, 8, 2]
    auto pred_kpt_vis_logit = kpt_view.select(-1, 2);                             // [B, A, 8]
    auto ap_xy = (anchor_points - 0.5).view({1, A, 1, 2});
    auto stride_4d = stride_tensor.view({1, A, 1, 1});
    auto pred_kpt_px = (raw_xy * 2.0 + ap_xy) * stride_4d;  // [B, A, 8, 2] pixel units

    auto anchor_px = anchor_points * stride_tensor;  // [A, 2]
    auto ax = anchor_px.select(1, 0), ay = anchor_px.select(1, 1);

    torch::Tensor cls_target = torch::zeros({B, A}, cls_raw.options());
    std::vector<torch::Tensor> box_losses, dfl_losses, kpt_losses;

    for (int64_t b = 0; b < B; ++b) {
        double gt_cx = targets_acc[b][0] * imgsz, gt_cy = targets_acc[b][1] * imgsz;
        double gt_w = targets_acc[b][2] * imgsz, gt_h = targets_acc[b][3] * imgsz;
        double gx1 = gt_cx - gt_w / 2, gy1 = gt_cy - gt_h / 2, gx2 = gt_cx + gt_w / 2, gy2 = gt_cy + gt_h / 2;

        auto inside = (ax > gx1).logical_and(ax < gx2).logical_and(ay > gy1).logical_and(ay < gy2);
        auto inside_idx = torch::nonzero(inside).squeeze(-1);
        if (inside_idx.numel() == 0) {
            auto dist2 = (ax - gt_cx).pow(2) + (ay - gt_cy).pow(2);
            inside_idx = torch::argmin(dist2).view({1});
        }

        auto cand_boxes = pred_bbox_px[b].index_select(0, inside_idx);
        auto gt_box_row = torch::tensor({gx1, gy1, gx2, gy2}, cand_boxes.options()).view({1, 4});
        auto ious = bbox_iou_xyxy(cand_boxes, gt_box_row.expand({cand_boxes.size(0), 4})).detach();

        auto cand_cls_score = torch::sigmoid(cls_raw[b].index_select(0, inside_idx)).detach();
        auto metric = torch::pow(cand_cls_score, 0.5) * torch::pow(ious.clamp_min(0), 6.0);

        int64_t topk = std::min<int64_t>(10, inside_idx.numel());
        auto [topk_vals, topk_local] = torch::topk(metric, topk);
        auto pos_idx = inside_idx.index_select(0, topk_local);
        int64_t P = pos_idx.numel();

        cls_target[b].index_fill_(0, pos_idx, 1.0);

        auto pos_pred_boxes = pred_bbox_px[b].index_select(0, pos_idx);
        auto gt_box_rep = gt_box_row.expand({P, 4});
        auto ciou = ciou_xyxy(pos_pred_boxes, gt_box_rep);
        box_losses.push_back((1.0 - ciou).mean());

        auto pos_anchor_pts = anchor_points.index_select(0, pos_idx);  // [P,2]
        auto pos_strides = stride_tensor.index_select(0, pos_idx);     // [P,1]
        auto gt_x1y1 = torch::tensor({gx1, gy1}, pos_anchor_pts.options()).view({1, 2});
        auto gt_x2y2 = torch::tensor({gx2, gy2}, pos_anchor_pts.options()).view({1, 2});
        auto target_lt = pos_anchor_pts - gt_x1y1 / pos_strides;
        auto target_rb = gt_x2y2 / pos_strides - pos_anchor_pts;
        auto target_ltrb = torch::cat({target_lt, target_rb}, 1).clamp(0.0, kRegMax - 1.01);

        auto pos_box_raw = box_raw[b].index_select(0, pos_idx).view({P, 4, kRegMax});
        dfl_losses.push_back(dfl_ce_loss(pos_box_raw, target_ltrb, kRegMax));

        auto pos_kpt_px = pred_kpt_px[b].index_select(0, pos_idx);              // [P,8,2]
        auto pos_kpt_vis_logit = pred_kpt_vis_logit[b].index_select(0, pos_idx);  // [P,8]

        std::vector<float> gt_kxy(16), gt_v(8);
        for (int k = 0; k < 8; ++k) {
            gt_kxy[2 * k] = targets_acc[b][5 + 3 * k] * imgsz;
            gt_kxy[2 * k + 1] = targets_acc[b][5 + 3 * k + 1] * imgsz;
            gt_v[k] = targets_acc[b][5 + 3 * k + 2];
        }
        auto gt_kxy_t = torch::from_blob(gt_kxy.data(), {8, 2}, torch::kFloat32).clone().to(device).view({1, 8, 2});
        auto gt_v_t = torch::from_blob(gt_v.data(), {8}, torch::kFloat32).clone().to(device).view({1, 8});
        auto vis_mask = (gt_v_t > 0).to(torch::kFloat32).expand({P, 8});

        double scale = std::sqrt(std::max(gt_w * gt_h, 1.0)) + 1e-6;
        constexpr double kSigma = 0.1;
        auto d2 = (pos_kpt_px - gt_kxy_t.expand({P, 8, 2})).pow(2).sum(-1);  // [P,8]
        auto oks = torch::exp(-d2 / (2.0 * scale * scale * kSigma * kSigma));
        auto pos_kpt_loss = ((1.0 - oks) * vis_mask).sum() / vis_mask.sum().clamp_min(1.0);

        auto vis_target = (gt_v_t > 0).to(torch::kFloat32).expand({P, 8});
        auto vis_loss = torch::binary_cross_entropy_with_logits(pos_kpt_vis_logit, vis_target);

        kpt_losses.push_back(pos_kpt_loss * w.kpt_pos + vis_loss * w.kpt_vis);
    }

    auto cls_loss = torch::binary_cross_entropy_with_logits(cls_raw, cls_target);
    auto box_loss = torch::stack(box_losses).mean();
    auto dfl_loss = torch::stack(dfl_losses).mean();
    auto kpt_loss = torch::stack(kpt_losses).mean();

    LossOutput out;
    out.box = box_loss;
    out.cls = cls_loss;
    out.dfl = dfl_loss;
    out.kpt = kpt_loss;
    out.total = box_loss * w.box + cls_loss * w.cls + dfl_loss * w.dfl + kpt_loss;
    return out;
}

}  // namespace rubik
