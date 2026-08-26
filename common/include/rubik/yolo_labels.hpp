#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <vector>

namespace rubik {

// One YOLO-pose label line: class 0, normalized box (cx,cy,w,h), then
// num_keypoints * (x_norm, y_norm, visibility) triples.
struct YoloPoseLabel {
    double cx = 0, cy = 0, w = 0, h = 0;
    std::vector<std::array<double, 3>> keypoints;  // normalized x, y, then raw v (0/1/2 or a confidence)
};

// Reads the first non-empty line of a YOLO-pose label file. Returns nullopt if the
// file is missing/empty or doesn't have enough values for num_keypoints.
std::optional<YoloPoseLabel> read_yolo_pose_label(const std::filesystem::path& path, int num_keypoints);

void write_yolo_pose_label(const std::filesystem::path& path, const YoloPoseLabel& label);

// Mirrors coco_keypoints_to_yolo_pose.py's _bbox_from_keypoints: tight box around
// keypoints with visibility > 0, normalized to [0,1]. Falls back to a centered
// default box if nothing is visible.
std::array<double, 4> bbox_from_keypoints(const std::vector<double>& flat_keypoints_abs, int width, int height);

}  // namespace rubik
