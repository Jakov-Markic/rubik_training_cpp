#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rubik {

struct CocoCategory {
    int id = 0;
    std::string name;
    std::string supercategory = "object";
    std::vector<std::string> keypoints;          // e.g. "kp0".."kp7"
    std::vector<std::array<int, 2>> skeleton;     // 1-indexed pairs, as COCO stores them
};

struct CocoImage {
    int id = 0;
    std::string file_name;
    int width = 0;
    int height = 0;
};

// keypoints is a flat [x0,y0,v0, x1,y1,v1, ...] list, absolute pixel coords, empty if unset.
struct CocoAnnotation {
    int id = 0;
    int image_id = 0;
    int category_id = 0;
    std::array<double, 4> bbox{0, 0, 0, 0};  // x, y, w, h
    double area = 0.0;
    int iscrowd = 0;
    int num_keypoints = -1;  // -1 = field omitted
    std::vector<double> keypoints;
};

struct CocoDataset {
    std::string description = "COCO dataset";
    std::vector<CocoCategory> categories;
    std::vector<CocoImage> images;
    std::vector<CocoAnnotation> annotations;

    std::unordered_map<int, const CocoImage*> images_by_id() const;
    std::unordered_map<int, std::string> category_name_by_id() const;
    std::optional<int> category_id_by_name(const std::string& name) const;
};

CocoDataset load_coco(const std::filesystem::path& path);

// minimal_images: yolo_pose_to_coco_keypoints.py writes bare {id,file_name,width,height}
// image entries and an empty "licenses" list, unlike the other tools' fuller schema.
// Same information either way -- readers (CVAT included) ignore the extra fields -- but
// matching each script's exact output keeps the port faithful.
void save_coco(const CocoDataset& coco, const std::filesystem::path& path, bool minimal_images = false);

// Mirrors merge_coco_keypoints_datasets.py's _resolve_image: try images_root/file_name,
// then images_root/basename(file_name), then a unique recursive match on the basename.
std::optional<std::filesystem::path> resolve_image(const std::filesystem::path& images_root,
                                                     const std::string& file_name);

}  // namespace rubik
