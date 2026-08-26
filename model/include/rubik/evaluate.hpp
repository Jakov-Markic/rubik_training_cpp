#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include "rubik/dataset.hpp"
#include "rubik/yolo_pose_8pt.hpp"

// Runs the trained model over a dataset (no augmentation, no gradient) and decodes each
// image's single highest-confidence anchor -- the same "one detection per image"
// convention pseudo_label_real_cube_images and the Flutter app's decode both use -- so
// the numbers here reflect what the model would actually report at inference time.

namespace rubik {

struct SampleEval {
    double confidence = 0.0;  // sigmoid(cls) of the chosen anchor
    double iou = 0.0;         // predicted vs GT box IoU, pixel space
    bool correct = false;     // iou >= 0.5
    std::array<double, 8> kpt_error_px{};   // pixel distance from GT, per keypoint
    std::array<bool, 8> kpt_gt_visible{};   // whether that GT keypoint had v > 0 (error only meaningful if true)

    // Only populated for the first few samples (for the prediction contact sheet).
    cv::Mat image;                                  // BGR, imgsz x imgsz
    std::array<double, 4> pred_box_norm{};           // x1,y1,x2,y2 normalized to imgsz
    std::array<double, 4> gt_box_norm{};
    std::array<std::array<double, 3>, 8> pred_kpts_norm{};  // x,y normalized, v = confidence
    std::array<std::array<double, 3>, 8> gt_kpts_norm{};
};

std::vector<SampleEval> evaluate_dataset(YoloPose8pt& model, const PoseDataset& dataset, int imgsz, int batch_size,
                                          torch::Device device, int max_visualize_samples = 12);

struct PRPoint {
    double threshold, precision, recall, f1;
};
// Precision/recall/F1 as a function of confidence threshold. "Positive" here means "the
// model's top detection for this image is both above the confidence threshold and
// actually correct (IoU >= 0.5)" -- since every image has exactly one real cube, recall's
// denominator is just the sample count.
std::vector<PRPoint> compute_pr_curve(const std::vector<SampleEval>& samples, double step = 0.05);

// Contact sheet: GT (green) vs predicted (yellow) box + keypoints, one panel per sample.
cv::Mat render_prediction_contact_sheet(const std::vector<SampleEval>& samples, int max_images = 12,
                                         int thumb_size = 220);

}  // namespace rubik
