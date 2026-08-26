#pragma once

#include <filesystem>
#include <random>
#include <vector>

#include <torch/torch.h>

#include "rubik/augment.hpp"

namespace rubik {

struct PoseBatch {
    torch::Tensor images;   // [N, 3, imgsz, imgsz] float32, 0..1
    torch::Tensor targets;  // [N, 5 + 8*3]: cx,cy,w,h,area_placeholder(unused=0), then 8*(x,y,v) -- see loss.hpp
};

// Loads a YOLO-pose dataset directory (images/<split>, labels/<split>) built by
// coco_keypoints_to_yolo_pose. Every sample has exactly one object (one cube).
class PoseDataset {
public:
    PoseDataset(const std::filesystem::path& dataset_root, const std::string& split, int num_keypoints = 8);

    size_t size() const { return entries_.size(); }

    // is_train=true applies the full random augmentation pipeline (mosaic/affine/hsv/
    // erasing per cfg); false just letterboxes deterministically (validation).
    PoseBatch get_batch(const std::vector<size_t>& indices, const AugmentConfig& cfg, bool is_train,
                         std::mt19937& rng) const;

private:
    struct Entry {
        std::filesystem::path image_path;
        std::filesystem::path label_path;
    };
    std::vector<Entry> entries_;
    int num_keypoints_;

    PoseSample load_raw(size_t idx, int imgsz) const;
};

}  // namespace rubik
