#include "rubik/dataset.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rubik/yolo_labels.hpp"

namespace rubik {

namespace fs = std::filesystem;

PoseDataset::PoseDataset(const fs::path& dataset_root, const std::string& split, int num_keypoints)
    : num_keypoints_(num_keypoints) {
    fs::path images_dir = dataset_root / "images" / split;
    fs::path labels_dir = dataset_root / "labels" / split;
    if (!fs::exists(images_dir)) throw std::runtime_error("Images dir not found: " + images_dir.string());

    static const std::set<std::string> kImageExts = {".jpg", ".jpeg", ".png", ".webp"};
    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (kImageExts.count(ext)) image_paths.push_back(entry.path());
    }
    std::sort(image_paths.begin(), image_paths.end());

    for (const auto& img_path : image_paths) {
        fs::path label_path = labels_dir / (img_path.stem().string() + ".txt");
        if (!fs::exists(label_path)) continue;
        entries_.push_back({img_path, label_path});
    }
    if (entries_.empty()) {
        throw std::runtime_error("No (image,label) pairs found under " + dataset_root.string() + " split=" + split);
    }
}

PoseSample PoseDataset::load_raw(size_t idx, int imgsz) const {
    const Entry& e = entries_[idx];
    cv::Mat img = cv::imread(e.image_path.string());
    if (img.empty()) throw std::runtime_error("Could not read image: " + e.image_path.string());

    auto label = read_yolo_pose_label(e.label_path, num_keypoints_);
    if (!label) throw std::runtime_error("Could not read label: " + e.label_path.string());

    std::array<std::array<double, 3>, 8> kpts{};
    for (int i = 0; i < 8 && i < static_cast<int>(label->keypoints.size()); ++i) kpts[i] = label->keypoints[i];

    return load_and_letterbox(img, label->cx, label->cy, label->w, label->h, kpts, imgsz);
}

namespace {
torch::Tensor sample_to_tensor(const PoseSample& s) {
    cv::Mat rgb;
    cv::cvtColor(s.image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    // HWC -> CHW via from_blob + permute + clone (own the memory before rgb goes out of scope).
    torch::Tensor t = torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, torch::kFloat32).clone();
    return t.permute({2, 0, 1}).contiguous();  // [3, H, W]
}

torch::Tensor sample_to_target(const PoseSample& s) {
    std::vector<float> row;
    row.reserve(5 + 24);
    row.insert(row.end(), {static_cast<float>(s.cx), static_cast<float>(s.cy), static_cast<float>(s.w),
                            static_cast<float>(s.h), 0.0f});
    for (const auto& kp : s.kpts) {
        row.insert(row.end(), {static_cast<float>(kp[0]), static_cast<float>(kp[1]), static_cast<float>(kp[2])});
    }
    return torch::from_blob(row.data(), {static_cast<int64_t>(row.size())}, torch::kFloat32).clone();
}
}  // namespace

PoseBatch PoseDataset::get_batch(const std::vector<size_t>& indices, const AugmentConfig& cfg, bool is_train,
                                  std::mt19937& rng) const {
    std::vector<torch::Tensor> image_tensors, target_tensors;
    image_tensors.reserve(indices.size());
    target_tensors.reserve(indices.size());

    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<size_t> any_idx(0, entries_.size() - 1);

    for (size_t idx : indices) {
        PoseSample sample;
        if (is_train && prob(rng) < cfg.mosaic_prob) {
            std::array<PoseSample, 4> quad;
            quad[0] = load_raw(idx, cfg.imgsz);
            for (int k = 1; k < 4; ++k) quad[k] = load_raw(any_idx(rng), cfg.imgsz);
            sample = apply_mosaic(quad, cfg.imgsz, rng);
        } else {
            sample = load_raw(idx, cfg.imgsz);
        }

        if (is_train) {
            sample = apply_random_affine(sample, cfg, rng);
            apply_hsv_jitter(sample.image, cfg, rng);
            apply_random_erasing(sample.image, cfg, rng);
        }

        image_tensors.push_back(sample_to_tensor(sample));
        target_tensors.push_back(sample_to_target(sample));
    }

    PoseBatch batch;
    batch.images = torch::stack(image_tensors, 0);
    batch.targets = torch::stack(target_tensors, 0);
    return batch;
}

}  // namespace rubik
