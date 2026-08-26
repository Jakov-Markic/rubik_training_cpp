#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Minimal OpenCV-drawn plotting (no matplotlib in C++) -- line plots, histograms, and bar
// charts, styled close enough to ultralytics' results.png/PR-curve/etc. to serve the same
// purpose: a quick visual read on how training went.

namespace rubik {

struct PlotSeries {
    std::string label;
    std::vector<double> y;  // same length as the x vector passed to plot_lines
    cv::Scalar color;       // BGR
};

// Multi-series line plot (e.g. train/val loss curves over epochs).
cv::Mat plot_lines(const std::string& title, const std::string& xlabel, const std::string& ylabel,
                    const std::vector<double>& x, const std::vector<PlotSeries>& series, int width = 900,
                    int height = 600);

// Histogram of a single value distribution (e.g. prediction confidence).
cv::Mat plot_histogram(const std::string& title, const std::string& xlabel, const std::vector<double>& values,
                        int num_bins = 20, cv::Scalar color = {180, 119, 31}, int width = 900, int height = 600);

// Categorical vertical bar chart (e.g. mean pixel error per keypoint).
cv::Mat plot_bar(const std::string& title, const std::string& ylabel, const std::vector<std::string>& labels,
                  const std::vector<double>& values, cv::Scalar color = {60, 160, 60}, int width = 900,
                  int height = 600);

}  // namespace rubik
