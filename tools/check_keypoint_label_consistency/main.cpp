// Automated sanity check for CVAT keypoint label ordering. A cube face's 4 corners,
// labeled in the correct rotational order, can never project to a self-intersecting
// (bowtie) quad under any camera angle -- that's a projective-geometry fact. If a
// labeled quad IS self-intersecting, two of its corner labels were swapped.
//
// C++ port of check_keypoint_label_consistency.py.

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <CLI11.hpp>

#include "rubik/coco.hpp"
#include "rubik/geometry.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    CLI::App app{"Automated sanity check for CVAT keypoint label ordering."};

    // Flat [json0, root0, json1, root1, ...] -- each --source occurrence contributes
    // exactly 2 tokens (type_size(2)), and the option is repeatable.
    std::vector<std::string> raw_sources;
    std::string keypoint_label = "cube_corners";

    app.add_option("--source", raw_sources, "Pair of (coco_json, images_root). Repeatable.")
        ->required()
        ->type_size(2);
    app.add_option("--keypoint-label", keypoint_label);

    CLI11_PARSE(app, argc, argv);

    if (raw_sources.size() % 2 != 0) {
        std::cerr << "--source expects two values (coco_json, images_root) per occurrence" << std::endl;
        return 1;
    }
    std::vector<std::pair<std::string, std::string>> sources;
    for (size_t i = 0; i < raw_sources.size(); i += 2) sources.emplace_back(raw_sources[i], raw_sources[i + 1]);

    int total = 0, front_bad = 0, back_bad = 0, front_cw = 0, front_ccw = 0;
    std::vector<std::string> bad_files;

    for (const auto& [json_str, images_root_str] : sources) {
        (void)images_root_str;  // unused, mirrors check_keypoint_label_consistency.py
        rubik::CocoDataset coco = rubik::load_coco(json_str);
        auto cat_by_id = coco.category_name_by_id();
        auto images_by_id = coco.images_by_id();

        for (const auto& ann : coco.annotations) {
            auto cat_it = cat_by_id.find(ann.category_id);
            if (cat_it == cat_by_id.end() || cat_it->second != keypoint_label) continue;
            if (ann.keypoints.empty()) continue;

            std::array<rubik::Point2d, 8> pts;
            std::array<int, 8> vis;
            for (int i = 0; i < 8; ++i) {
                pts[i] = {ann.keypoints[3 * i], ann.keypoints[3 * i + 1]};
                vis[i] = static_cast<int>(ann.keypoints[3 * i + 2]);
            }

            std::string fname;
            auto img_it = images_by_id.find(ann.image_id);
            fname = img_it != images_by_id.end() ? img_it->second->file_name
                                                  : ("image_id=" + std::to_string(ann.image_id));
            total++;

            std::array<rubik::Point2d, 4> front = {pts[0], pts[1], pts[2], pts[3]};
            std::array<rubik::Point2d, 4> back = {pts[4], pts[5], pts[6], pts[7]};
            bool front_vis = vis[0] > 0 && vis[1] > 0 && vis[2] > 0 && vis[3] > 0;
            bool back_vis = vis[4] > 0 && vis[5] > 0 && vis[6] > 0 && vis[7] > 0;

            bool is_bad = false;
            if (front_vis) {
                if (rubik::quad_self_intersects(front)) {
                    front_bad++;
                    is_bad = true;
                } else {
                    double area = rubik::signed_area(front);
                    if (area > 0) front_ccw++;
                    else front_cw++;
                }
            }
            if (back_vis && rubik::quad_self_intersects(back)) {
                back_bad++;
                is_bad = true;
            }

            if (is_bad) bad_files.push_back(fname);
        }
    }

    std::cout << "Checked " << total << " labeled cubes." << std::endl;
    std::cout << "Front-face self-intersecting (BAD - corner order wrong): " << front_bad << std::endl;
    std::cout << "Back-face self-intersecting (BAD - corner order wrong): " << back_bad << std::endl;
    std::cout << "Front-face winding direction: " << front_cw << " clockwise, " << front_ccw
              << " counter-clockwise (a real mix is normal - depends on which side of the cube faces the camera)"
              << std::endl;
    if (!bad_files.empty()) {
        std::cout << "\nFiles with a self-intersecting quad (check/relabel these in CVAT):" << std::endl;
        for (const auto& f : bad_files) std::cout << "  " << f << std::endl;
    } else {
        std::cout << "\nNo self-intersecting quads found - corner ordering looks structurally consistent."
                  << std::endl;
    }
    return 0;
}
