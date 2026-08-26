#include "rubik/plotting.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace rubik {

namespace {

constexpr int kMarginLeft = 75;
constexpr int kMarginRight = 190;  // room for a legend
constexpr int kMarginTop = 55;
constexpr int kMarginBottom = 65;

std::string fmt_num(double v) {
    std::ostringstream ss;
    if (std::abs(v) >= 1000 || (std::abs(v) < 0.001 && v != 0.0)) {
        ss.precision(2);
        ss << std::scientific << v;
    } else {
        ss.precision(std::abs(v) < 10 ? 3 : 1);
        ss << std::fixed << v;
    }
    return ss.str();
}

// Maps data-space (x,y) to pixel-space within the plot's inner rectangle, and draws the
// shared chrome (background, frame, grid, ticks, title, axis labels).
struct AxisMapper {
    double x_min, x_max, y_min, y_max;
    cv::Rect inner;

    cv::Point to_px(double x, double y) const {
        double fx = (x_max > x_min) ? (x - x_min) / (x_max - x_min) : 0.5;
        double fy = (y_max > y_min) ? (y - y_min) / (y_max - y_min) : 0.5;
        int px = inner.x + static_cast<int>(std::lround(fx * inner.width));
        int py = inner.y + inner.height - static_cast<int>(std::lround(fy * inner.height));
        return {px, py};
    }
};

AxisMapper draw_chrome(cv::Mat& canvas, const std::string& title, const std::string& xlabel,
                        const std::string& ylabel, double x_min, double x_max, double y_min, double y_max) {
    canvas.setTo(cv::Scalar(255, 255, 255));
    int width = canvas.cols, height = canvas.rows;
    cv::Rect inner(kMarginLeft, kMarginTop, width - kMarginLeft - kMarginRight, height - kMarginTop - kMarginBottom);

    // Pad the y-range slightly so lines/bars don't touch the frame edges.
    double pad = (y_max - y_min) * 0.08;
    if (pad <= 0) pad = std::max(1.0, std::abs(y_max) * 0.1);
    y_min -= pad;
    y_max += pad;

    AxisMapper m{x_min, x_max, y_min, y_max, inner};

    cv::rectangle(canvas, inner, cv::Scalar(60, 60, 60), 1);

    constexpr int kGridLines = 5;
    for (int i = 0; i <= kGridLines; ++i) {
        double t = static_cast<double>(i) / kGridLines;
        int gy = inner.y + inner.height - static_cast<int>(std::lround(t * inner.height));
        cv::line(canvas, {inner.x, gy}, {inner.x + inner.width, gy}, cv::Scalar(225, 225, 225), 1);
        double val = y_min + t * (y_max - y_min);
        cv::putText(canvas, fmt_num(val), {8, gy + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(40, 40, 40), 1,
                    cv::LINE_AA);

        int gx = inner.x + static_cast<int>(std::lround(t * inner.width));
        cv::line(canvas, {gx, inner.y}, {gx, inner.y + inner.height}, cv::Scalar(235, 235, 235), 1);
        double xval = x_min + t * (x_max - x_min);
        cv::putText(canvas, fmt_num(xval), {gx - 15, inner.y + inner.height + 20}, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    }

    cv::putText(canvas, title, {kMarginLeft, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(20, 20, 20), 2,
                cv::LINE_AA);
    cv::putText(canvas, xlabel, {inner.x + inner.width / 2 - 30, height - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
    cv::putText(canvas, ylabel, {8, kMarginTop - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(20, 20, 20), 1,
                cv::LINE_AA);

    return m;
}

}  // namespace

cv::Mat plot_lines(const std::string& title, const std::string& xlabel, const std::string& ylabel,
                    const std::vector<double>& x, const std::vector<PlotSeries>& series, int width, int height) {
    cv::Mat canvas(height, width, CV_8UC3);
    if (x.empty() || series.empty()) {
        canvas.setTo(cv::Scalar(255, 255, 255));
        return canvas;
    }

    double x_min = x.front(), x_max = x.back();
    double y_min = std::numeric_limits<double>::infinity(), y_max = -std::numeric_limits<double>::infinity();
    for (const auto& s : series) {
        for (double v : s.y) {
            y_min = std::min(y_min, v);
            y_max = std::max(y_max, v);
        }
    }
    if (!std::isfinite(y_min) || !std::isfinite(y_max)) {
        y_min = 0;
        y_max = 1;
    }

    AxisMapper m = draw_chrome(canvas, title, xlabel, ylabel, x_min, x_max, y_min, y_max);

    int legend_y = m.inner.y + 10;
    for (const auto& s : series) {
        std::vector<cv::Point> pts;
        pts.reserve(s.y.size());
        for (size_t i = 0; i < s.y.size() && i < x.size(); ++i) pts.push_back(m.to_px(x[i], s.y[i]));
        cv::polylines(canvas, pts, false, s.color, 2, cv::LINE_AA);

        cv::rectangle(canvas, {m.inner.x + m.inner.width + 15, legend_y}, {m.inner.x + m.inner.width + 30, legend_y + 12},
                      s.color, -1);
        cv::putText(canvas, s.label, {m.inner.x + m.inner.width + 36, legend_y + 11}, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        legend_y += 22;
    }
    return canvas;
}

cv::Mat plot_histogram(const std::string& title, const std::string& xlabel, const std::vector<double>& values,
                        int num_bins, cv::Scalar color, int width, int height) {
    cv::Mat canvas(height, width, CV_8UC3);
    if (values.empty()) {
        canvas.setTo(cv::Scalar(255, 255, 255));
        return canvas;
    }

    double v_min = *std::min_element(values.begin(), values.end());
    double v_max = *std::max_element(values.begin(), values.end());
    if (v_max <= v_min) v_max = v_min + 1.0;

    std::vector<int> counts(num_bins, 0);
    for (double v : values) {
        int bin = static_cast<int>((v - v_min) / (v_max - v_min) * num_bins);
        bin = std::clamp(bin, 0, num_bins - 1);
        counts[bin]++;
    }
    int max_count = *std::max_element(counts.begin(), counts.end());

    AxisMapper m = draw_chrome(canvas, title, xlabel, "count", v_min, v_max, 0, max_count);

    double bin_width_data = (v_max - v_min) / num_bins;
    for (int i = 0; i < num_bins; ++i) {
        double bx0 = v_min + i * bin_width_data;
        double bx1 = bx0 + bin_width_data;
        cv::Point p0 = m.to_px(bx0, 0);
        cv::Point p1 = m.to_px(bx1, counts[i]);
        cv::rectangle(canvas, {p0.x + 1, p1.y}, {p1.x - 1, p0.y}, color, -1);
    }
    return canvas;
}

cv::Mat plot_bar(const std::string& title, const std::string& ylabel, const std::vector<std::string>& labels,
                  const std::vector<double>& values, cv::Scalar color, int width, int height) {
    cv::Mat canvas(height, width, CV_8UC3);
    if (values.empty()) {
        canvas.setTo(cv::Scalar(255, 255, 255));
        return canvas;
    }

    double v_max = *std::max_element(values.begin(), values.end());
    v_max = std::max(v_max, 1e-6);

    AxisMapper m = draw_chrome(canvas, title, "", ylabel, 0, static_cast<double>(values.size()), 0, v_max);

    double slot = static_cast<double>(m.inner.width) / values.size();
    for (size_t i = 0; i < values.size(); ++i) {
        int x0 = m.inner.x + static_cast<int>(std::lround(i * slot)) + 6;
        int x1 = m.inner.x + static_cast<int>(std::lround((i + 1) * slot)) - 6;
        cv::Point top = m.to_px(0, values[i]);
        cv::rectangle(canvas, {x0, top.y}, {x1, m.inner.y + m.inner.height}, color, -1);
        cv::putText(canvas, fmt_num(values[i]), {x0, top.y - 6}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        if (i < labels.size()) {
            cv::putText(canvas, labels[i], {x0, m.inner.y + m.inner.height + 20}, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                        cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        }
    }
    return canvas;
}

}  // namespace rubik
