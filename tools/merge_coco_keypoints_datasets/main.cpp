// Merge multiple CVAT COCO keypoints exports into one dataset with copied images.
// C++ port of merge_coco_keypoints_datasets.py.

#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <CLI11.hpp>

#include "rubik/coco.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    CLI::App app{"Merge multiple CVAT COCO keypoints exports into one dataset with copied images."};

    std::vector<std::string> input_jsons, images_roots, dataset_tags;
    fs::path output_root;
    std::string cube_label = "cube";
    std::string keypoint_label = "cube_corners";

    app.add_option("--input-json", input_jsons, "Path to a COCO keypoints JSON (repeat for each dataset).")
        ->required();
    app.add_option("--images-root", images_roots, "Path to image root corresponding to each --input-json.")
        ->required();
    app.add_option("--dataset-tag", dataset_tags, "Optional tag for each dataset (same order).");
    app.add_option("--output-root", output_root)->required();
    app.add_option("--cube-label", cube_label);
    app.add_option("--keypoint-label", keypoint_label);

    CLI11_PARSE(app, argc, argv);

    if (input_jsons.size() != images_roots.size()) {
        std::cerr << "--input-json and --images-root counts must match" << std::endl;
        return 1;
    }
    std::vector<std::string> tags;
    if (dataset_tags.empty()) {
        for (size_t i = 0; i < input_jsons.size(); ++i) tags.push_back("ds" + std::to_string(i + 1));
    } else {
        if (dataset_tags.size() != input_jsons.size()) {
            std::cerr << "--dataset-tag count must match --input-json count when provided" << std::endl;
            return 1;
        }
        tags = dataset_tags;
    }

    fs::path out_root = fs::absolute(output_root);
    fs::path out_images = out_root / "images" / "all";
    fs::path out_ann = out_root / "annotations";
    // Clear first: merged_keypoints.json always fully reflects the current inputs, but
    // images/all/ would otherwise keep every earlier run's copied files too. Mirrors the
    // same fix in merge_coco_keypoints_datasets.py.
    fs::remove_all(out_images);
    fs::create_directories(out_images);
    fs::create_directories(out_ann);

    rubik::CocoDataset merged;
    merged.description = "Merged CVAT COCO keypoints dataset";
    merged.categories.push_back({1, cube_label, "object", {}, {}});
    rubik::CocoCategory kp_cat;
    kp_cat.id = 2;
    kp_cat.name = keypoint_label;
    kp_cat.supercategory = "object";
    for (int i = 0; i < 8; ++i) kp_cat.keypoints.push_back("kp" + std::to_string(i));
    kp_cat.skeleton = {{1, 2}, {2, 3}, {3, 4}, {4, 1}, {5, 6}, {6, 7}, {7, 8}, {8, 5}, {1, 5}, {2, 6}, {3, 7}, {4, 8}};
    merged.categories.push_back(kp_cat);

    int next_image_id = 1;
    int next_ann_id = 1;

    for (size_t idx = 0; idx < input_jsons.size(); ++idx) {
        fs::path json_path = fs::absolute(input_jsons[idx]);
        fs::path img_root = fs::absolute(images_roots[idx]);
        const std::string& tag = tags[idx];

        rubik::CocoDataset coco = rubik::load_coco(json_path);
        auto cat_by_id = coco.category_name_by_id();

        std::set<int> kept_category_ids;
        for (const auto& [cid, name] : cat_by_id) {
            if (name == cube_label || name == keypoint_label) kept_category_ids.insert(cid);
        }

        std::map<int, std::vector<const rubik::CocoAnnotation*>> anns_by_image;
        for (const auto& ann : coco.annotations) {
            if (!kept_category_ids.count(ann.category_id)) continue;
            anns_by_image[ann.image_id].push_back(&ann);
        }

        // Iterate coco.images (JSON insertion order) rather than an unordered_map, so
        // the merged output's image/annotation ids are deterministic like Python's dict.
        for (const auto& image_info : coco.images) {
            auto it = anns_by_image.find(image_info.id);
            if (it == anns_by_image.end()) continue;

            const std::string& src_name = image_info.file_name;
            auto src_path = rubik::resolve_image(img_root, src_name);
            if (!src_path) throw std::runtime_error("Could not resolve image " + src_name + " under " + img_root.string());

            std::string dst_name = tag + "__" + fs::path(src_name).filename().string();
            fs::path dst_path = out_images / dst_name;
            if (!fs::exists(dst_path)) fs::copy_file(*src_path, dst_path);

            rubik::CocoImage new_image;
            new_image.id = next_image_id;
            new_image.file_name = dst_name;
            new_image.width = image_info.width;
            new_image.height = image_info.height;
            merged.images.push_back(new_image);

            for (const auto* ann : it->second) {
                std::string ann_name = cat_by_id[ann->category_id];
                int new_cat_id = (ann_name == cube_label) ? 1 : 2;

                rubik::CocoAnnotation new_ann;
                new_ann.id = next_ann_id;
                new_ann.image_id = next_image_id;
                new_ann.category_id = new_cat_id;
                new_ann.bbox = ann->bbox;
                new_ann.area = ann->area;
                new_ann.iscrowd = ann->iscrowd;
                if (!ann->keypoints.empty()) new_ann.keypoints = ann->keypoints;
                if (ann->num_keypoints >= 0) new_ann.num_keypoints = ann->num_keypoints;
                merged.annotations.push_back(new_ann);
                next_ann_id++;
            }
            next_image_id++;
        }

        std::cout << "[" << (idx + 1) << "/" << input_jsons.size() << "] merged " << json_path.filename().string()
                  << " with tag '" << tag << "'" << std::endl;
    }

    fs::path out_json = out_ann / "merged_keypoints.json";
    rubik::save_coco(merged, out_json);

    std::cout << "Wrote merged COCO: " << out_json << std::endl;
    std::cout << "Images: " << merged.images.size() << std::endl;
    std::cout << "Annotations: " << merged.annotations.size() << std::endl;
    return 0;
}
