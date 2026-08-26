#pragma once

#include <array>
#include <random>

#include <opencv2/core.hpp>

// Augmentation pipeline for the single-object (one cube per image) pose dataset.
// All coordinates on PoseSample are normalized [0,1] relative to the *current* image,
// which is always square (imgsz x imgsz) once load_and_letterbox has run.

namespace rubik {

struct PoseSample {
    cv::Mat image;  // BGR uint8, imgsz x imgsz
    double cx = 0, cy = 0, w = 0, h = 0;
    std::array<std::array<double, 3>, 8> kpts{};  // x, y (normalized), v (0/1/2)
};

struct AugmentConfig {
    int imgsz = 640;
    double degrees = 18.0, translate = 0.08, scale = 0.25, shear = 2.0;
    double perspective = 0.0005;  // accepted, currently a no-op -- see README_CPP.md
    double mosaic_prob = 0.15;
    double mixup_prob = 0.0;  // accepted, unimplemented -- current training config never sets this nonzero
    double hsv_h = 0.015, hsv_s = 0.5, hsv_v = 0.35;
    double erasing_prob = 0.1;
};

// Deterministic scale-to-fit + centered gray-pad resize to imgsz x imgsz, remapping the
// raw (image-relative-normalized) box/keypoints into the new square canvas' normalized
// coordinates. This is the non-random resize every sample goes through (train and val).
PoseSample load_and_letterbox(const cv::Mat& image_bgr, double cx, double cy, double w, double h,
                               const std::array<std::array<double, 3>, 8>& kpts_norm, int imgsz);

// Random rotate/translate/scale/shear around the canvas center, output size unchanged
// (imgsz x imgsz), gray border fill. Keypoints that land outside the canvas after the
// transform are marked invisible (v=0) rather than extrapolated.
PoseSample apply_random_affine(const PoseSample& in, const AugmentConfig& cfg, std::mt19937& rng);

// In-place HSV jitter (image only, doesn't touch labels).
void apply_hsv_jitter(cv::Mat& image, const AugmentConfig& cfg, std::mt19937& rng);

// In-place random erasing: with probability erasing_prob, paints one random gray
// rectangle over the image (image only, doesn't touch labels).
void apply_random_erasing(cv::Mat& image, const AugmentConfig& cfg, std::mt19937& rng);

// Combines 4 already-letterboxed imgsz x imgsz samples into one 2x2 grid, then downsizes
// back to imgsz x imgsz, and keeps ONE of the 4 sub-images' object as this sample's
// target (not all 4) -- deliberate, to stay consistent with the single-GT-per-image
// assigner in loss.hpp: mosaic here is a background/context augmentation, not a
// multi-instance composor.
PoseSample apply_mosaic(const std::array<PoseSample, 4>& samples, int imgsz, std::mt19937& rng);

}  // namespace rubik
