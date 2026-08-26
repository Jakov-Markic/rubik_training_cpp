#pragma once

#include <opencv2/core.hpp>

namespace rubik {

// Variance of the Laplacian, on a grayscale conversion of the frame. Higher = sharper.
double blur_score(const cv::Mat& frame_bgr);

// Downscales so the long side is at most long_side pixels (no-op if long_side <= 0 or
// already small enough). Mirrors extract_video_frames.py's _resize_long_side.
cv::Mat resize_long_side(const cv::Mat& frame, int long_side);

}  // namespace rubik
