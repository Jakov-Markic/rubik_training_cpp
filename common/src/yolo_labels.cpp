#include "rubik/yolo_labels.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>

namespace rubik {

std::optional<YoloPoseLabel> read_yolo_pose_label(const std::filesystem::path& path, int num_keypoints) {
    std::ifstream in(path);
    if (!in) return std::nullopt;

    std::string line;
    if (!std::getline(in, line)) return std::nullopt;
    // Trim trailing CR (files may have been written/read across CRLF boundaries).
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    if (line.empty()) return std::nullopt;

    std::istringstream iss(line);
    std::vector<double> vals;
    double v;
    while (iss >> v) vals.push_back(v);

    const size_t expected = 5 + static_cast<size_t>(num_keypoints) * 3;
    if (vals.size() < expected) return std::nullopt;

    YoloPoseLabel label;
    label.cx = vals[1];
    label.cy = vals[2];
    label.w = vals[3];
    label.h = vals[4];
    label.keypoints.reserve(num_keypoints);
    for (int i = 0; i < num_keypoints; ++i) {
        label.keypoints.push_back({vals[5 + 3 * i], vals[5 + 3 * i + 1], vals[5 + 3 * i + 2]});
    }
    return label;
}

void write_yolo_pose_label(const std::filesystem::path& path, const YoloPoseLabel& label) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "0 " << label.cx << ' ' << label.cy << ' ' << label.w << ' ' << label.h;
    for (const auto& kp : label.keypoints) {
        out << ' ' << kp[0] << ' ' << kp[1] << ' ' << kp[2];
    }
    out << '\n';
}

std::array<double, 4> bbox_from_keypoints(const std::vector<double>& flat_keypoints_abs, int width, int height) {
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    bool any = false;

    for (size_t i = 0; i + 2 < flat_keypoints_abs.size(); i += 3) {
        double x = flat_keypoints_abs[i];
        double y = flat_keypoints_abs[i + 1];
        int v = static_cast<int>(flat_keypoints_abs[i + 2]);
        if (v <= 0) continue;
        any = true;
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    if (!any) return {0.5, 0.5, 0.2, 0.2};

    double cx = ((min_x + max_x) / 2.0) / width;
    double cy = ((min_y + max_y) / 2.0) / height;
    double bw = std::max(1.0, max_x - min_x) / width;
    double bh = std::max(1.0, max_y - min_y) / height;
    return {cx, cy, bw, bh};
}

}  // namespace rubik
