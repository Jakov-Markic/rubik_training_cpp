// Thin out near-duplicate frames from extract_video_frames output. Frames are grouped
// by source-video prefix (the "clip1" in clip1_000042.jpg) and processed in
// frame-index order; each candidate is compared to the last KEPT frame in its group
// (not just the previous raw frame), so slow motion still accumulates into new keeps
// once enough has actually changed.
//
// C++ port of dedupe_similar_frames.py.

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include <CLI11.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace {

const std::regex kFrameIndexRe(R"(^(.*)_(\d+)$)");

std::pair<std::string, long long> group_key(const fs::path& path) {
    std::string stem = path.stem().string();
    std::smatch m;
    if (std::regex_match(stem, m, kFrameIndexRe)) {
        return {m[1].str(), std::stoll(m[2].str())};
    }
    return {stem, 0};
}

cv::Mat compare_key(const fs::path& path) {
    cv::Mat img = cv::imread(path.string());
    cv::Mat small;
    cv::resize(img, small, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
    cv::Mat gray;
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::Mat out;
    gray.convertTo(out, CV_32F);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{
        "Thin out near-duplicate frames from extract_video_frames output. Frames are "
        "grouped by source-video prefix and compared to the last KEPT frame in each group."};

    // Assumes invocation from rubik_training_cpp/build/ (where CMake places every tool).
    fs::path folder = fs::current_path() / ".." / ".." / "rubik_training" / "video_frames";
    double min_diff = 12.0;
    fs::path move_to;
    bool do_delete = false;
    bool dry_run = false;

    app.add_option("--folder", folder, "Folder of extracted frames.");
    app.add_option("--min-diff", min_diff,
                    "Minimum mean-absolute-difference (0-255) vs. the last kept frame to keep a new frame.");
    app.add_option("--move-to", move_to, "Move near-duplicates here instead of deleting (default: <folder>/near_duplicates).");
    app.add_flag("--delete", do_delete, "Delete near-duplicates instead of moving them.");
    app.add_flag("--dry-run", dry_run, "Only report what would happen; don't move/delete anything.");

    CLI11_PARSE(app, argc, argv);

    std::vector<fs::path> files;
    if (fs::exists(folder)) {
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") files.push_back(entry.path());
        }
    }
    if (files.empty()) {
        std::cout << "No .jpg files found in " << folder << std::endl;
        return 0;
    }

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return group_key(a) < group_key(b);
    });

    std::map<std::string, std::vector<fs::path>> groups;
    std::vector<std::string> group_order;
    for (const auto& f : files) {
        std::string prefix = group_key(f).first;
        if (groups.find(prefix) == groups.end()) group_order.push_back(prefix);
        groups[prefix].push_back(f);
    }

    if (move_to.empty()) move_to = folder / "near_duplicates";

    int total_kept = 0, total_removed = 0;

    for (const auto& prefix : group_order) {
        const auto& group_files = groups[prefix];
        bool have_kept = false;
        cv::Mat kept_key;
        int kept = 0, removed = 0;

        for (const auto& f : group_files) {
            cv::Mat key = compare_key(f);
            if (!have_kept) {
                kept_key = key;
                have_kept = true;
                kept++;
                continue;
            }
            cv::Mat diff_mat;
            cv::absdiff(key, kept_key, diff_mat);
            double diff = cv::mean(diff_mat)[0];
            if (diff < min_diff) {
                removed++;
                if (!dry_run) {
                    if (do_delete) {
                        fs::remove(f);
                    } else {
                        fs::create_directories(move_to);
                        fs::rename(f, move_to / f.filename());
                    }
                }
            } else {
                kept_key = key;
                kept++;
            }
        }
        std::cout << prefix << ": " << group_files.size() << " frames -> keep " << kept << ", remove " << removed
                  << std::endl;
        total_kept += kept;
        total_removed += removed;
    }

    std::string action = dry_run ? "would remove" : (do_delete ? "deleted" : ("moved to " + move_to.string()));
    std::cout << "\nTotal: keep " << total_kept << ", " << action << " " << total_removed << std::endl;
    return 0;
}
