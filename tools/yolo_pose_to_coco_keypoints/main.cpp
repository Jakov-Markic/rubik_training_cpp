// Convert YOLO pose txt labels to a CVAT-importable COCO keypoints JSON.
// C++ port of yolo_pose_to_coco_keypoints.py.

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>

#include <CLI11.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rubik/coco.hpp"
#include "rubik/yolo_labels.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    CLI::App app{"Convert YOLO pose txt labels to a CVAT-importable COCO keypoints JSON."};

    fs::path images_dir, labels_dir, output_json;
    int num_keypoints = 8;
    std::string category_name = "cube";

    app.add_option("--images-dir", images_dir, "Directory with images (e.g. pseudo_labels/images/train)")->required();
    app.add_option("--labels-dir", labels_dir, "Directory with YOLO pose labels (e.g. pseudo_labels/labels/train)")
        ->required();
    app.add_option("--output-json", output_json, "Destination COCO keypoints JSON file")->required();
    app.add_option("--num-keypoints", num_keypoints, "Number of keypoints per object");
    app.add_option("--category-name", category_name, "Category name to write into COCO categories");

    CLI11_PARSE(app, argc, argv);

    images_dir = fs::absolute(images_dir);
    labels_dir = fs::absolute(labels_dir);
    output_json = fs::absolute(output_json);

    if (!fs::exists(images_dir)) {
        std::cerr << "Images directory not found: " << images_dir << std::endl;
        return 1;
    }
    if (!fs::exists(labels_dir)) {
        std::cerr << "Labels directory not found: " << labels_dir << std::endl;
        return 1;
    }

    static const std::set<std::string> kImageExts = {".jpg", ".jpeg", ".png", ".webp"};
    std::vector<fs::path> image_files;
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (kImageExts.count(ext)) image_files.push_back(entry.path());
    }
    std::sort(image_files.begin(), image_files.end());

    rubik::CocoDataset coco;
    coco.description = "Pseudo labels converted from YOLO pose to COCO keypoints for CVAT review";
    coco.categories.push_back({1, category_name, "object", {}, {}});
    for (int i = 0; i < num_keypoints; ++i) coco.categories[0].keypoints.push_back("kp" + std::to_string(i));

    int image_id = 1;
    int ann_id = 1;

    for (const auto& image_path : image_files) {
        fs::path label_path = labels_dir / (image_path.stem().string() + ".txt");
        if (!fs::exists(label_path)) continue;

        cv::Mat img = cv::imread(image_path.string());
        if (img.empty()) throw std::runtime_error("Could not read image: " + image_path.string());
        int width = img.cols, height = img.rows;

        rubik::CocoImage cim;
        cim.id = image_id;
        cim.file_name = image_path.filename().string();
        cim.width = width;
        cim.height = height;
        coco.images.push_back(cim);

        auto label = rubik::read_yolo_pose_label(label_path, num_keypoints);
        if (!label) {
            image_id++;
            continue;
        }

        double abs_bw = label->w * width;
        double abs_bh = label->h * height;
        double abs_x = (label->cx * width) - abs_bw / 2.0;
        double abs_y = (label->cy * height) - abs_bh / 2.0;

        std::vector<double> coco_kpts;
        int visible_count = 0;
        for (int i = 0; i < num_keypoints; ++i) {
            double x = label->keypoints[i][0] * width;
            double y = label->keypoints[i][1] * height;
            int v = static_cast<int>(label->keypoints[i][2]);
            coco_kpts.insert(coco_kpts.end(), {x, y, static_cast<double>(v)});
            if (v > 0) visible_count++;
        }

        rubik::CocoAnnotation ann;
        ann.id = ann_id;
        ann.image_id = image_id;
        ann.category_id = 1;
        ann.bbox = {abs_x, abs_y, abs_bw, abs_bh};
        ann.area = abs_bw * abs_bh;
        ann.iscrowd = 0;
        ann.num_keypoints = visible_count;
        ann.keypoints = coco_kpts;
        coco.annotations.push_back(ann);

        image_id++;
        ann_id++;
    }

    rubik::save_coco(coco, output_json, /*minimal_images=*/true);

    std::cout << "Wrote COCO keypoints JSON: " << output_json << std::endl;
    std::cout << "Images: " << coco.images.size() << std::endl;
    std::cout << "Annotations: " << coco.annotations.size() << std::endl;
    return 0;
}
