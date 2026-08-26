// Extract a labeling-ready set of frames from one or more short videos of the cube
// (record while slowly rotating/moving it through many angles, backgrounds, and
// distances). Samples at a target rate and drops blurry frames, so you don't waste
// CVAT labeling effort on near-duplicate or motion-blurred frames.
//
// C++ port of extract_video_frames.py.

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <CLI11.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "rubik/image_io.hpp"

namespace fs = std::filesystem;

namespace {

struct ExtractStats {
    int scanned = 0;
    int saved = 0;
    int skipped_blurry = 0;
};

ExtractStats extract_from_video(const fs::path& video_path, const fs::path& output_dir, double target_fps,
                                 double blur_threshold, int max_frames, int jpeg_quality, int resize_long_side) {
    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened()) throw std::runtime_error("Could not open video: " + video_path.string());

    double source_fps = cap.get(cv::CAP_PROP_FPS);
    if (source_fps <= 0) source_fps = 30.0;
    int step = std::max(1, static_cast<int>(std::lround(source_fps / target_fps)));

    std::string stem = video_path.stem().string();
    fs::create_directories(output_dir);

    ExtractStats stats;
    std::vector<std::pair<int, cv::Mat>> candidates;

    int frame_idx = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame_idx % step == 0) {
            stats.scanned++;
            double score = rubik::blur_score(frame);
            if (score < blur_threshold) {
                stats.skipped_blurry++;
            } else {
                candidates.emplace_back(frame_idx, frame.clone());
            }
        }
        frame_idx++;
    }
    cap.release();

    if (max_frames > 0 && static_cast<int>(candidates.size()) > max_frames) {
        // Evenly subsample across the kept (sharp) candidates rather than just taking
        // the first N, to preserve angle/pose diversity.
        std::vector<size_t> idx_set;
        for (int i = 0; i < max_frames; ++i) {
            double t = static_cast<double>(i) * (candidates.size() - 1) / (max_frames - 1);
            idx_set.push_back(static_cast<size_t>(std::lround(t)));
        }
        std::sort(idx_set.begin(), idx_set.end());
        idx_set.erase(std::unique(idx_set.begin(), idx_set.end()), idx_set.end());

        std::vector<std::pair<int, cv::Mat>> subsampled;
        subsampled.reserve(idx_set.size());
        for (size_t i : idx_set) subsampled.push_back(candidates[i]);
        candidates = std::move(subsampled);
    }

    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
    for (auto& [idx, cand_frame] : candidates) {
        char name[64];
        std::snprintf(name, sizeof(name), "%s_%06d.jpg", stem.c_str(), idx);
        fs::path out_path = output_dir / name;
        cv::Mat resized = rubik::resize_long_side(cand_frame, resize_long_side);
        cv::imwrite(out_path.string(), resized, encode_params);
        stats.saved++;
    }

    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{
        "Extract a labeling-ready set of frames from one or more short videos of the cube "
        "(record while slowly rotating/moving it through many angles, backgrounds, and "
        "distances). Samples at a target rate and drops blurry frames, so you don't waste "
        "CVAT labeling effort on near-duplicate or motion-blurred frames."};

    std::vector<std::string> videos;
    // Assumes invocation from rubik_training_cpp/build/ (where CMake places every tool);
    // video_frames/ stays in rubik_training/ (an active, growing workspace, not "dataset"
    // data -- see shared_datasets/ for the COCO-format datasets both pipelines share).
    fs::path output_dir = fs::current_path() / ".." / ".." / "rubik_training" / "video_frames";
    double fps = 2.0;
    double blur_threshold = 60.0;
    int max_frames = 0;  // 0 == unset (Python's None)
    int jpeg_quality = 90;
    int resize_long = 1280;

    app.add_option("--video", videos, "Path to a video file. Repeat --video for multiple clips in one run.")
        ->required();
    app.add_option("--output-dir", output_dir, "Where extracted JPEGs are written.");
    app.add_option("--fps", fps, "Target sampling rate in frames per second (default: 2.0).");
    app.add_option("--blur-threshold", blur_threshold,
                    "Minimum variance-of-Laplacian sharpness to keep a frame (default: 60.0).");
    app.add_option("--max-frames", max_frames, "Optional cap on saved frames per video (0 = no cap).");
    app.add_option("--jpeg-quality", jpeg_quality, "JPEG quality for saved frames (default: 90).");
    app.add_option("--resize-long-side", resize_long,
                    "Downscale saved frames so their long side is at most this many pixels "
                    "(default: 1280; pass 0 to keep original resolution).");

    CLI11_PARSE(app, argc, argv);

    ExtractStats total;
    for (const auto& video_str : videos) {
        fs::path video_path(video_str);
        if (!fs::exists(video_path)) {
            std::cerr << "Video not found: " << video_path << std::endl;
            return 1;
        }
        ExtractStats stats =
            extract_from_video(video_path, output_dir, fps, blur_threshold, max_frames, jpeg_quality, resize_long);
        std::cout << video_path.filename().string() << ": scanned " << stats.scanned << ", saved " << stats.saved
                  << ", skipped (blurry) " << stats.skipped_blurry << std::endl;
        total.scanned += stats.scanned;
        total.saved += stats.saved;
        total.skipped_blurry += stats.skipped_blurry;
    }

    std::cout << "\nTotal: scanned " << total.scanned << ", saved " << total.saved << " frames to " << output_dir
              << ", skipped " << total.skipped_blurry << " blurry" << std::endl;
    std::cout << "Next: import that folder into CVAT (see README_TRAINING.md) and label the 8 corners." << std::endl;
    return 0;
}
