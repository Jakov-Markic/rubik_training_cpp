#include "rubik/image_io.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace rubik {

double blur_score(const cv::Mat& frame_bgr) {
    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];  // variance = stddev^2
}

cv::Mat resize_long_side(const cv::Mat& frame, int long_side) {
    int h = frame.rows, w = frame.cols;
    int current_long = std::max(h, w);
    if (long_side <= 0 || current_long <= long_side) return frame;
    double scale = static_cast<double>(long_side) / current_long;
    int new_w = std::max(1, static_cast<int>(std::lround(w * scale)));
    int new_h = std::max(1, static_cast<int>(std::lround(h * scale)));
    cv::Mat out;
    cv::resize(frame, out, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    return out;
}

}  // namespace rubik
