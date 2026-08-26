// Render numbered keypoints + skeleton over labeled images, as a contact-sheet grid,
// for a quick visual QA of label ordering consistency.
// C++ port of visualize_keypoint_labels.py.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <CLI11.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rubik/coco.hpp"

namespace fs = std::filesystem;

namespace {

// Same order/skeleton convention as detector_service.dart's _cubeObjectPoints:
// 0-3 front face (TL,TR,BR,BL), 4-7 back face (TL,TR,BR,BL).
const std::vector<std::pair<int, int>> kSkeleton = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                                                      {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
const std::vector<cv::Scalar> kKpColorsBgr = {
    {0, 0, 255}, {0, 128, 255}, {0, 255, 255}, {0, 255, 0}, {255, 255, 0}, {255, 128, 0}, {255, 0, 0}, {255, 0, 255},
};

cv::Mat render_one(const fs::path& img_path, const std::vector<double>& keypoints, int thumb_size, bool crop_to_cube) {
    cv::Mat img = cv::imread(img_path.string());
    if (img.empty()) return cv::Mat();
    int h = img.rows, w = img.cols;

    std::vector<cv::Point2d> pts(8);
    std::vector<int> vis(8);
    for (int i = 0; i < 8; ++i) {
        pts[i] = {keypoints[3 * i], keypoints[3 * i + 1]};
        vis[i] = static_cast<int>(keypoints[3 * i + 2]);
    }

    if (crop_to_cube) {
        std::vector<cv::Point2d> visible_pts;
        for (int i = 0; i < 8; ++i)
            if (vis[i] > 0) visible_pts.push_back(pts[i]);
        if (!visible_pts.empty()) {
            double min_x = visible_pts[0].x, max_x = visible_pts[0].x;
            double min_y = visible_pts[0].y, max_y = visible_pts[0].y;
            for (const auto& p : visible_pts) {
                min_x = std::min(min_x, p.x);
                max_x = std::max(max_x, p.x);
                min_y = std::min(min_y, p.y);
                max_y = std::max(max_y, p.y);
            }
            double pad = 0.35 * std::max({max_x - min_x, max_y - min_y, 1.0});
            int x0 = std::max(0, static_cast<int>(min_x - pad));
            int y0 = std::max(0, static_cast<int>(min_y - pad));
            int x1 = std::min(w, static_cast<int>(max_x + pad));
            int y1 = std::min(h, static_cast<int>(max_y + pad));
            if (x1 > x0 && y1 > y0) {
                img = img(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
                for (auto& p : pts) {
                    p.x -= x0;
                    p.y -= y0;
                }
                h = img.rows;
                w = img.cols;
            }
        }
    }

    for (const auto& [a, b] : kSkeleton) {
        if (vis[a] > 0 && vis[b] > 0) {
            cv::Scalar color = (a < 4 && b < 4) ? cv::Scalar(0, 255, 0)
                                                 : ((a >= 4 && b >= 4) ? cv::Scalar(255, 200, 0) : cv::Scalar(200, 200, 200));
            cv::line(img, cv::Point(static_cast<int>(pts[a].x), static_cast<int>(pts[a].y)),
                     cv::Point(static_cast<int>(pts[b].x), static_cast<int>(pts[b].y)), color, 4);
        }
    }

    for (int i = 0; i < 8; ++i) {
        if (vis[i] <= 0) continue;
        cv::Point p(static_cast<int>(pts[i].x), static_cast<int>(pts[i].y));
        cv::circle(img, p, 10, kKpColorsBgr[i], -1);
        cv::circle(img, p, 10, cv::Scalar(0, 0, 0), 2);
        cv::putText(img, std::to_string(i), p + cv::Point(12, -8), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 4);
        cv::putText(img, std::to_string(i), p + cv::Point(12, -8), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(255, 255, 255), 2);
    }

    double scale = static_cast<double>(thumb_size) / std::max(w, h);
    cv::Mat thumb;
    cv::resize(img, thumb, cv::Size(std::max(1, static_cast<int>(w * scale)), std::max(1, static_cast<int>(h * scale))));
    cv::Mat canvas(thumb_size, thumb_size, CV_8UC3, cv::Scalar(40, 40, 40));
    int y0 = (thumb_size - thumb.rows) / 2;
    int x0 = (thumb_size - thumb.cols) / 2;
    thumb.copyTo(canvas(cv::Rect(x0, y0, thumb.cols, thumb.rows)));
    return canvas;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Render numbered keypoints + skeleton over labeled images as a contact-sheet grid."};

    std::vector<std::string> raw_sources;  // flat [json0, root0, json1, root1, ...]
    std::string keypoint_label = "cube_corners";
    fs::path output = fs::current_path() / "label_qa" / "contact_sheet.jpg";
    int thumb_size = 260;
    int cols = 8;
    int max_images = 200;
    int skip = 0;
    bool crop_to_cube = false;

    app.add_option("--source", raw_sources, "Pair of (coco_json, images_root). Repeatable.")
        ->required()
        ->type_size(2);
    app.add_option("--keypoint-label", keypoint_label);
    app.add_option("--output", output);
    app.add_option("--thumb-size", thumb_size);
    app.add_option("--cols", cols);
    app.add_option("--max-images", max_images);
    app.add_option("--skip", skip, "Skip the first N matched annotations (for paging through in batches).");
    app.add_flag("--crop-to-cube", crop_to_cube,
                 "Crop tightly around each labeled cube instead of showing the full photo.");

    CLI11_PARSE(app, argc, argv);

    if (raw_sources.size() % 2 != 0) {
        std::cerr << "--source expects two values (coco_json, images_root) per occurrence" << std::endl;
        return 1;
    }

    std::vector<std::pair<std::string, cv::Mat>> thumbs;

    for (size_t si = 0; si < raw_sources.size(); si += 2) {
        fs::path json_path = raw_sources[si];
        fs::path images_root = raw_sources[si + 1];

        rubik::CocoDataset coco = rubik::load_coco(json_path);
        auto cat_by_id = coco.category_name_by_id();
        auto images_by_id = coco.images_by_id();

        for (const auto& ann : coco.annotations) {
            auto cat_it = cat_by_id.find(ann.category_id);
            if (cat_it == cat_by_id.end() || cat_it->second != keypoint_label) continue;
            if (ann.keypoints.empty()) continue;
            auto img_it = images_by_id.find(ann.image_id);
            if (img_it == images_by_id.end()) continue;

            auto img_path = rubik::resolve_image(images_root, img_it->second->file_name);
            if (!img_path) {
                std::cerr << "WARN: could not resolve " << img_it->second->file_name << " under " << images_root
                          << std::endl;
                continue;
            }
            cv::Mat thumb = render_one(*img_path, ann.keypoints, thumb_size, crop_to_cube);
            if (!thumb.empty()) thumbs.emplace_back(img_path->filename().string(), thumb);
        }
    }

    if (thumbs.empty()) {
        std::cout << "No labeled images found." << std::endl;
        return 0;
    }

    if (skip > 0) {
        if (static_cast<size_t>(skip) >= thumbs.size()) thumbs.clear();
        else thumbs.erase(thumbs.begin(), thumbs.begin() + skip);
    }
    if (static_cast<int>(thumbs.size()) > max_images) thumbs.resize(max_images);

    int rows = static_cast<int>(std::ceil(static_cast<double>(thumbs.size()) / cols));
    cv::Mat sheet(rows * thumb_size, cols * thumb_size, CV_8UC3, cv::Scalar(20, 20, 20));
    for (size_t idx = 0; idx < thumbs.size(); ++idx) {
        int r = static_cast<int>(idx) / cols;
        int c = static_cast<int>(idx) % cols;
        thumbs[idx].second.copyTo(sheet(cv::Rect(c * thumb_size, r * thumb_size, thumb_size, thumb_size)));
    }

    fs::create_directories(output.parent_path());
    cv::imwrite(output.string(), sheet, {cv::IMWRITE_JPEG_QUALITY, 90});
    std::cout << "Wrote contact sheet with " << thumbs.size() << " images to " << output << std::endl;
    std::cout << "Legend: kp0-3 = front face (green edges), kp4-7 = back face (cyan edges), "
                 "gray = front-to-back connectors."
              << std::endl;
    std::cout << "Look for: front-face color/number pattern (0=red,1=orange,2=yellow,3=green corner) landing on "
                 "the same relative corner every time, and no self-crossing edges."
              << std::endl;
    return 0;
}
