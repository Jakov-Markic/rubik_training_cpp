// Convert a COCO keypoints dataset to YOLO pose format.
// C++ port of coco_keypoints_to_yolo_pose.py.
//
// NOTE: the train/val shuffle uses std::mt19937 seeded the same way as the CLI
// --seed, but this will NOT reproduce the exact same split as Python's
// random.shuffle() (different PRNG algorithm) -- deterministic, not
// cross-language-identical. See README_CPP.md.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <CLI11.hpp>

#include "rubik/coco.hpp"
#include "rubik/yolo_labels.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    CLI::App app{"Convert COCO keypoints dataset to YOLO pose format."};

    fs::path coco_json, images_root, output_root;
    std::string keypoint_label = "cube_corners";
    int num_keypoints = 8;
    double train_ratio = 0.9;
    unsigned int seed = 42;

    app.add_option("--coco-json", coco_json)->required();
    app.add_option("--images-root", images_root)->required();
    app.add_option("--output-root", output_root)->required();
    app.add_option("--keypoint-label", keypoint_label);
    app.add_option("--num-keypoints", num_keypoints);
    app.add_option("--train-ratio", train_ratio);
    app.add_option("--seed", seed);

    CLI11_PARSE(app, argc, argv);

    rubik::CocoDataset coco = rubik::load_coco(coco_json);
    auto images_by_id = coco.images_by_id();

    std::optional<int> keypoint_cat_id = coco.category_id_by_name(keypoint_label);
    if (!keypoint_cat_id) {
        std::cerr << "Category '" << keypoint_label << "' not found in COCO categories" << std::endl;
        return 1;
    }

    struct Sample {
        const rubik::CocoImage* image;
        const rubik::CocoAnnotation* ann;
    };
    std::vector<Sample> samples;
    for (const auto& ann : coco.annotations) {
        if (ann.category_id != *keypoint_cat_id) continue;
        auto it = images_by_id.find(ann.image_id);
        if (it == images_by_id.end()) continue;
        if (ann.keypoints.empty()) continue;
        samples.push_back({it->second, &ann});
    }

    std::mt19937 rng(seed);
    std::shuffle(samples.begin(), samples.end(), rng);

    size_t split_idx = std::max<size_t>(1, static_cast<size_t>(samples.size() * train_ratio));
    std::vector<Sample> train_samples(samples.begin(), samples.begin() + std::min(split_idx, samples.size()));
    std::vector<Sample> val_samples(samples.begin() + std::min(split_idx, samples.size()), samples.end());

    // Clear first: earlier runs' images/labels never get removed otherwise, so a dataset
    // regenerated multiple times (e.g. after dropping a source) silently accumulates every
    // prior run's files underneath the current ones, and training (which scans these
    // directories directly, not a manifest) picks up all of it. Mirrors the same fix in
    // coco_keypoints_to_yolo_pose.py.
    for (const std::string& split : {"train", "val"}) {
        fs::remove_all(output_root / "images" / split);
        fs::remove_all(output_root / "labels" / split);
        fs::create_directories(output_root / "images" / split);
        fs::create_directories(output_root / "labels" / split);
    }

    auto write_split = [&](const std::string& split_name, const std::vector<Sample>& split_samples) -> int {
        int written = 0;
        for (const auto& sample : split_samples) {
            const std::string& img_name = sample.image->file_name;
            fs::path src_img = images_root / img_name;
            if (!fs::exists(src_img)) continue;

            fs::path dst_img = output_root / "images" / split_name / img_name;
            if (!fs::exists(dst_img)) fs::copy_file(src_img, dst_img);

            int w = std::max(1, sample.image->width);
            int h = std::max(1, sample.image->height);

            double cx, cy, bw_n, bh_n;
            const auto& bbox = sample.ann->bbox;
            bool has_bbox = bbox[2] != 0.0 || bbox[3] != 0.0;
            if (has_bbox) {
                cx = (bbox[0] + bbox[2] / 2.0) / w;
                cy = (bbox[1] + bbox[3] / 2.0) / h;
                bw_n = std::max(1e-6, bbox[2] / w);
                bh_n = std::max(1e-6, bbox[3] / h);
            } else {
                auto box = rubik::bbox_from_keypoints(sample.ann->keypoints, w, h);
                cx = box[0];
                cy = box[1];
                bw_n = box[2];
                bh_n = box[3];
            }

            const auto& keypoints = sample.ann->keypoints;
            if (static_cast<int>(keypoints.size()) < num_keypoints * 3) continue;

            rubik::YoloPoseLabel label;
            label.cx = cx;
            label.cy = cy;
            label.w = bw_n;
            label.h = bh_n;
            for (int i = 0; i < num_keypoints; ++i) {
                double x = keypoints[3 * i] / w;
                double y = keypoints[3 * i + 1] / h;
                int v = static_cast<int>(keypoints[3 * i + 2]);
                label.keypoints.push_back({x, y, static_cast<double>(v)});
            }

            fs::path label_path = output_root / "labels" / split_name / (fs::path(img_name).stem().string() + ".txt");
            rubik::write_yolo_pose_label(label_path, label);
            written++;
        }
        return written;
    };

    int train_written = write_split("train", train_samples);
    int val_written = write_split("val", val_samples);

    fs::path data_yaml = output_root / "data.yaml";
    std::ofstream yaml_out(data_yaml);
    yaml_out << "path: " << fs::absolute(output_root).string() << "\n";
    yaml_out << "train: images/train\n";
    yaml_out << "val: images/val\n";
    yaml_out << "nc: 1\n";
    yaml_out << "names:\n";
    yaml_out << "  0: cube\n";
    yaml_out << "kpt_shape: [" << num_keypoints << ", 3]\n";

    std::cout << "Wrote YOLO pose dataset to " << output_root << std::endl;
    std::cout << "train samples: " << train_written << std::endl;
    std::cout << "val samples: " << val_written << std::endl;
    return 0;
}
