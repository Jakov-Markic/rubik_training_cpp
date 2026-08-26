#include "rubik/evaluate.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include <opencv2/imgproc.hpp>

namespace rubik {

namespace {

// Same softmax-weighted-expectation DFL decode as loss.cpp's private dfl_decode
// (duplicated rather than shared across a translation-unit boundary for this small a
// function -- not worth a header just for this).
torch::Tensor dfl_decode(const torch::Tensor& box_bca, int reg_max) {
    auto sizes = box_bca.sizes();
    int64_t B = sizes[0], A = sizes[2];
    auto y = box_bca.view({B, 4, reg_max, A}).transpose(1, 2);
    y = torch::softmax(y, 1);
    auto weight = torch::arange(reg_max, y.options()).view({1, reg_max, 1, 1});
    return (y * weight).sum(1);
}

double iou_xyxy(const std::array<double, 4>& a, const std::array<double, 4>& b) {
    double x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    double x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    double area_a = std::max(0.0, a[2] - a[0]) * std::max(0.0, a[3] - a[1]);
    double area_b = std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]);
    return inter / (area_a + area_b - inter + 1e-9);
}

cv::Mat tensor_to_bgr(const torch::Tensor& chw_rgb01) {
    auto t = chw_rgb01.detach().to(torch::kCPU).contiguous();
    auto hwc = t.permute({1, 2, 0}).contiguous();
    int H = static_cast<int>(hwc.size(0)), W = static_cast<int>(hwc.size(1));
    cv::Mat rgb(H, W, CV_32FC3);
    std::memcpy(rgb.data, hwc.data_ptr<float>(), sizeof(float) * H * W * 3);
    cv::Mat rgb8, bgr;
    rgb.convertTo(rgb8, CV_8UC3, 255.0);
    cv::cvtColor(rgb8, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

const std::vector<std::pair<int, int>> kSkeleton = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                                                      {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

}  // namespace

std::vector<SampleEval> evaluate_dataset(YoloPose8pt& model, const PoseDataset& dataset, int imgsz, int batch_size,
                                          torch::Device device, int max_visualize_samples) {
    model->eval();
    torch::NoGradGuard no_grad;

    AugmentConfig cfg;
    cfg.imgsz = imgsz;
    std::mt19937 rng(0);  // unused (is_train=false disables all randomness) but required by the API

    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<SampleEval> results;
    results.reserve(dataset.size());

    for (size_t start = 0; start < indices.size(); start += batch_size) {
        size_t end = std::min(indices.size(), start + static_cast<size_t>(batch_size));
        std::vector<size_t> batch_idx(indices.begin() + start, indices.begin() + end);
        PoseBatch batch = dataset.get_batch(batch_idx, cfg, /*is_train=*/false, rng);

        auto images = batch.images.to(device);
        auto targets = batch.targets.to(device);
        auto raw = model->forward(images);

        auto box_raw = flatten_levels(raw.box);
        auto cls_raw = flatten_levels(raw.cls).squeeze(-1);
        auto kpt_raw = flatten_levels(raw.kpt);

        Anchors anchors = make_anchors(raw.box, raw.strides);
        auto anchor_points = anchors.points.to(device);
        auto stride_tensor = anchors.strides.to(device);
        int64_t A = anchor_points.size(0);

        auto box_for_dfl = box_raw.transpose(1, 2).contiguous();
        auto ltrb_grid = dfl_decode(box_for_dfl, kRegMax).transpose(1, 2);
        auto pred_bbox_grid = dist2bbox(ltrb_grid, anchor_points);
        auto pred_bbox_px = pred_bbox_grid * stride_tensor;

        auto kpt_view = kpt_raw.view({box_raw.size(0), A, kNumKeypoints, 3});
        auto raw_xy = kpt_view.slice(-1, 0, 2);
        auto vis_logit = kpt_view.select(-1, 2);
        auto ap_xy = (anchor_points - 0.5).view({1, A, 1, 2});
        auto stride_4d = stride_tensor.view({1, A, 1, 1});
        auto pred_kpt_px = (raw_xy * 2.0 + ap_xy) * stride_4d;
        auto pred_kpt_vis = torch::sigmoid(vis_logit);

        auto best_idx = torch::argmax(cls_raw, /*dim=*/1);  // [B]
        auto best_conf = torch::sigmoid(torch::gather(cls_raw, 1, best_idx.unsqueeze(1)).squeeze(1));

        int64_t B = images.size(0);
        for (int64_t b = 0; b < B; ++b) {
            int64_t ai = best_idx[b].item<int64_t>();
            SampleEval s;
            s.confidence = best_conf[b].item<double>();

            std::array<double, 4> pred_box, gt_box;
            for (int k = 0; k < 4; ++k) pred_box[k] = pred_bbox_px[b][ai][k].item<double>();

            auto t = targets[b];
            double gt_cx = t[0].item<double>() * imgsz, gt_cy = t[1].item<double>() * imgsz;
            double gt_w = t[2].item<double>() * imgsz, gt_h = t[3].item<double>() * imgsz;
            gt_box = {gt_cx - gt_w / 2, gt_cy - gt_h / 2, gt_cx + gt_w / 2, gt_cy + gt_h / 2};

            s.iou = iou_xyxy(pred_box, gt_box);
            s.correct = s.iou >= 0.5;

            for (int k = 0; k < 8; ++k) {
                double px = pred_kpt_px[b][ai][k][0].item<double>();
                double py = pred_kpt_px[b][ai][k][1].item<double>();
                double gx = t[5 + 3 * k].item<double>() * imgsz;
                double gy = t[5 + 3 * k + 1].item<double>() * imgsz;
                double gv = t[5 + 3 * k + 2].item<double>();
                s.kpt_gt_visible[k] = gv > 0;
                s.kpt_error_px[k] = s.kpt_gt_visible[k] ? std::hypot(px - gx, py - gy) : -1.0;
            }

            if (static_cast<int>(results.size()) < max_visualize_samples) {
                s.image = tensor_to_bgr(images[b]);
                s.pred_box_norm = {pred_box[0] / imgsz, pred_box[1] / imgsz, pred_box[2] / imgsz, pred_box[3] / imgsz};
                s.gt_box_norm = {gt_box[0] / imgsz, gt_box[1] / imgsz, gt_box[2] / imgsz, gt_box[3] / imgsz};
                for (int k = 0; k < 8; ++k) {
                    s.pred_kpts_norm[k] = {pred_kpt_px[b][ai][k][0].item<double>() / imgsz,
                                           pred_kpt_px[b][ai][k][1].item<double>() / imgsz,
                                           pred_kpt_vis[b][ai][k].item<double>()};
                    s.gt_kpts_norm[k] = {t[5 + 3 * k].item<double>(), t[5 + 3 * k + 1].item<double>(),
                                         t[5 + 3 * k + 2].item<double>()};
                }
            }

            results.push_back(std::move(s));
        }
    }

    return results;
}

std::vector<PRPoint> compute_pr_curve(const std::vector<SampleEval>& samples, double step) {
    std::vector<PRPoint> out;
    if (samples.empty()) return out;
    size_t total = samples.size();

    for (double t = 0.0; t <= 1.0 + 1e-9; t += step) {
        size_t predicted_positive = 0, tp = 0;
        for (const auto& s : samples) {
            if (s.confidence >= t) {
                predicted_positive++;
                if (s.correct) tp++;
            }
        }
        double precision = predicted_positive > 0 ? static_cast<double>(tp) / predicted_positive : 1.0;
        double recall = static_cast<double>(tp) / total;
        double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
        out.push_back({t, precision, recall, f1});
    }
    return out;
}

cv::Mat render_prediction_contact_sheet(const std::vector<SampleEval>& samples, int max_images, int thumb_size) {
    std::vector<cv::Mat> thumbs;
    int n = std::min(static_cast<int>(samples.size()), max_images);

    for (int i = 0; i < n; ++i) {
        const SampleEval& s = samples[i];
        if (s.image.empty()) continue;
        cv::Mat img = s.image.clone();
        int size = img.cols;

        auto denorm = [&](double n) { return static_cast<int>(n * size); };

        cv::rectangle(img, {denorm(s.gt_box_norm[0]), denorm(s.gt_box_norm[1])},
                      {denorm(s.gt_box_norm[2]), denorm(s.gt_box_norm[3])}, cv::Scalar(0, 200, 0), 2);
        cv::rectangle(img, {denorm(s.pred_box_norm[0]), denorm(s.pred_box_norm[1])},
                      {denorm(s.pred_box_norm[2]), denorm(s.pred_box_norm[3])}, cv::Scalar(0, 220, 255), 2);

        for (const auto& [a, b] : kSkeleton) {
            if (s.gt_kpts_norm[a][2] > 0 && s.gt_kpts_norm[b][2] > 0) {
                cv::line(img, {denorm(s.gt_kpts_norm[a][0]), denorm(s.gt_kpts_norm[a][1])},
                         {denorm(s.gt_kpts_norm[b][0]), denorm(s.gt_kpts_norm[b][1])}, cv::Scalar(0, 200, 0), 1);
            }
            cv::line(img, {denorm(s.pred_kpts_norm[a][0]), denorm(s.pred_kpts_norm[a][1])},
                     {denorm(s.pred_kpts_norm[b][0]), denorm(s.pred_kpts_norm[b][1])}, cv::Scalar(0, 220, 255), 1);
        }
        for (int k = 0; k < 8; ++k) {
            if (s.gt_kpts_norm[k][2] > 0) {
                cv::circle(img, {denorm(s.gt_kpts_norm[k][0]), denorm(s.gt_kpts_norm[k][1])}, 4, cv::Scalar(0, 200, 0), -1);
            }
            cv::circle(img, {denorm(s.pred_kpts_norm[k][0]), denorm(s.pred_kpts_norm[k][1])}, 3, cv::Scalar(0, 220, 255), -1);
        }

        char conf_text[32];
        std::snprintf(conf_text, sizeof(conf_text), "conf %.2f iou %.2f", s.confidence, s.iou);
        cv::putText(img, conf_text, {6, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    s.correct ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        cv::Mat thumb;
        cv::resize(img, thumb, cv::Size(thumb_size, thumb_size));
        thumbs.push_back(thumb);
    }

    if (thumbs.empty()) return cv::Mat();

    int cols = std::min(4, static_cast<int>(thumbs.size()));
    int rows = static_cast<int>(std::ceil(static_cast<double>(thumbs.size()) / cols));
    cv::Mat sheet(rows * thumb_size, cols * thumb_size, CV_8UC3, cv::Scalar(20, 20, 20));
    for (size_t i = 0; i < thumbs.size(); ++i) {
        int r = static_cast<int>(i) / cols, c = static_cast<int>(i) % cols;
        thumbs[i].copyTo(sheet(cv::Rect(c * thumb_size, r * thumb_size, thumb_size, thumb_size)));
    }
    return sheet;
}

}  // namespace rubik
