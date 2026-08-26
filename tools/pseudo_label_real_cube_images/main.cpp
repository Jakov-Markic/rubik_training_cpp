// Pseudo-label unannotated real_cube images with a trained keypoint model, using
// LibTorch to run a TorchScript export directly (no Python/ultralytics at runtime).
//
// Unlike pseudo_label_real_cube_images.py -- which calls ultralytics' model.predict()
// on the raw checkpoint at a flexible --imgsz, with ultralytics doing letterboxing, NMS,
// and rescaling coordinates back to the original image internally -- this tool loads the
// *wrapped* TorchScript export train_cube_pose_8pt.py's export_for_flutter() already
// produces (PoseSingleBestWrapper): a fixed-input-size model that has already picked the
// single highest-confidence detection and returns a flat
// [conf, left, top, w, h, kp0_x, kp0_y, kp0_vis, ..., kp7_vis] vector, all normalized to
// the model's square input. So this tool has to do the letterbox preprocessing and the
// inverse (model-square -> original-image) coordinate mapping itself -- implemented to
// match solve_my_cube's detector_service.dart exactly (buildModelInputTensor /
// _mapModelPointToFrame), minus the app's camera-buffer-specific 90-degree rotation,
// which doesn't apply to already-upright JPEG photos.

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <CLI11.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>

#include "rubik/yolo_labels.hpp"

namespace fs = std::filesystem;

namespace {

struct Letterbox {
    int content_w, content_h;
    int pad_x, pad_y;
    int input_size;
};

// Mirrors buildModelInputTensor in detector_service.dart: scale-to-fit (never upscale),
// floor-based centered padding, gray(114) fill.
Letterbox compute_letterbox(int w, int h, int input_size) {
    int long_side = std::max(w, h);
    double scale = long_side <= input_size ? 1.0 : static_cast<double>(input_size) / long_side;
    int content_w = std::clamp(static_cast<int>(std::lround(w * scale)), 1, input_size);
    int content_h = std::clamp(static_cast<int>(std::lround(h * scale)), 1, input_size);
    int pad_x = (input_size - content_w) / 2;  // floor for positive ints
    int pad_y = (input_size - content_h) / 2;
    return {content_w, content_h, pad_x, pad_y, input_size};
}

cv::Mat build_letterbox_canvas(const cv::Mat& bgr, const Letterbox& lb) {
    cv::Mat content;
    if (lb.content_w == bgr.cols && lb.content_h == bgr.rows) {
        content = bgr;
    } else {
        cv::resize(bgr, content, cv::Size(lb.content_w, lb.content_h), 0, 0, cv::INTER_LINEAR);
    }
    cv::Mat canvas(lb.input_size, lb.input_size, CV_8UC3, cv::Scalar(114, 114, 114));
    content.copyTo(canvas(cv::Rect(lb.pad_x, lb.pad_y, lb.content_w, lb.content_h)));
    return canvas;
}

// Mirrors _mapModelPointToFrame (minus the rotation undo, which doesn't apply here):
// model-square-normalized (nx,ny) -> original-image-normalized (fx,fy), clamped to [0,1].
std::pair<double, double> map_model_point_to_image(double nx, double ny, const Letterbox& lb) {
    double content_w_frac = static_cast<double>(lb.content_w) / lb.input_size;
    double content_h_frac = static_cast<double>(lb.content_h) / lb.input_size;
    double pad_x_frac = static_cast<double>(lb.pad_x) / lb.input_size;
    double pad_y_frac = static_cast<double>(lb.pad_y) / lb.input_size;

    double content_x = nx - pad_x_frac;
    double content_y = ny - pad_y_frac;

    double fx = content_w_frac > 1e-6 ? content_x / content_w_frac : content_x;
    double fy = content_h_frac > 1e-6 ? content_y / content_h_frac : content_y;

    return {std::clamp(fx, 0.0, 1.0), std::clamp(fy, 0.0, 1.0)};
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Pseudo-label unannotated real_cube images with a trained keypoint model (LibTorch)."};

    // Defaults assume invocation from rubik_training_cpp/build/ (where CMake places every
    // tool); shared_datasets/ is a sibling of rubik_training/ and rubik_training_cpp/.
    fs::path model_path;
    fs::path source = fs::current_path() / ".." / ".." / "shared_datasets" / "detect_dataset_final" / "images";
    fs::path output = fs::current_path() / ".." / ".." / "shared_datasets" / "detect_dataset_final" / "pseudo_labels";
    int imgsz = 320;  // must match the imgsz the TorchScript export was traced at (see --mobile-imgsz)
    double conf_threshold = 0.20;
    std::string split = "train";
    bool flat = false;
    bool dump_raw = false;

    app.add_option("--model", model_path, "TorchScript export (.torchscript) of the trained pose model")
        ->required();
    app.add_option("--source", source, "Folder containing train/val image splits");
    app.add_option("--output", output, "Where to write generated labels");
    app.add_option("--imgsz", imgsz, "Model input size -- must match the export's --mobile-imgsz (default 320)");
    app.add_option("--conf", conf_threshold, "Minimum confidence threshold for detections");
    app.add_option("--split", split, "Which split to pseudo-label")->check(CLI::IsMember({"train", "val"}));
    app.add_flag("--dump-raw", dump_raw,
                 "Debug: print each image's raw (pre-letterbox-inverse) model output + letterbox params to stderr.");
    app.add_flag("--flat", flat,
                 "Treat --source as a flat folder of images directly (e.g. video_frames/), instead of "
                 "expecting a train/val split subfolder.");

    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(model_path)) {
        std::cerr << "Model not found: " << model_path << std::endl;
        return 1;
    }
    fs::path image_dir = flat ? source : source / split;
    if (!fs::exists(image_dir)) {
        std::cerr << "Image split not found: " << image_dir << std::endl;
        return 1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path.string());
    } catch (const c10::Error& e) {
        std::cerr << "Error loading TorchScript model: " << e.what() << std::endl;
        return 1;
    }
    module.eval();

    static const std::set<std::string> kImageExts = {".jpg", ".jpeg", ".png", ".webp"};
    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (kImageExts.count(ext)) image_paths.push_back(entry.path());
    }
    std::sort(image_paths.begin(), image_paths.end());

    fs::path out_labels = output / "labels" / split;
    fs::path out_images = output / "images" / split;
    fs::create_directories(out_labels);
    fs::create_directories(out_images);

    torch::NoGradGuard no_grad;
    int written = 0;

    for (const auto& image_path : image_paths) {
        if (!flat && image_path.filename().string().rfind("real_cube_", 0) != 0) continue;

        cv::Mat bgr = cv::imread(image_path.string());
        if (bgr.empty()) continue;
        int orig_w = bgr.cols, orig_h = bgr.rows;

        Letterbox lb = compute_letterbox(orig_w, orig_h, imgsz);
        cv::Mat canvas = build_letterbox_canvas(bgr, lb);
        cv::Mat rgb;
        cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

        std::vector<float> tensor_data(static_cast<size_t>(3) * imgsz * imgsz);
        int plane = imgsz * imgsz;
        for (int y = 0; y < imgsz; ++y) {
            const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < imgsz; ++x) {
                int idx = y * imgsz + x;
                tensor_data[idx] = row[x][0] / 255.0f;
                tensor_data[plane + idx] = row[x][1] / 255.0f;
                tensor_data[2 * plane + idx] = row[x][2] / 255.0f;
            }
        }
        torch::Tensor input = torch::from_blob(tensor_data.data(), {1, 3, imgsz, imgsz}, torch::kFloat32).clone();

        torch::Tensor output = module.forward({input}).toTensor().to(torch::kCPU).contiguous();
        auto out = output.accessor<float, 2>();  // [1, 29]: conf,left,top,w,h, then 8*(x,y,vis)

        float model_conf = out[0][0];

        if (dump_raw) {
            std::cerr << "image=" << image_path.filename().string() << " conf=" << out[0][0] << " left=" << out[0][1]
                      << " top=" << out[0][2] << " w=" << out[0][3] << " h=" << out[0][4] << std::endl;
            for (int i = 0; i < 8; ++i) {
                std::cerr << "  kp" << i << ": x=" << out[0][5 + 3 * i] << " y=" << out[0][5 + 3 * i + 1]
                          << " v=" << out[0][5 + 3 * i + 2] << std::endl;
            }
            std::cerr << "  letterbox: content_w=" << lb.content_w << " content_h=" << lb.content_h
                      << " pad_x=" << lb.pad_x << " pad_y=" << lb.pad_y << " orig=" << orig_w << "x" << orig_h
                      << std::endl;
        }

        if (model_conf < static_cast<float>(conf_threshold)) continue;

        double left = out[0][1], top = out[0][2], box_w = out[0][3], box_h = out[0][4];

        // Map all four box corners through the inverse letterbox (not just scale each axis
        // independently) -- matches detector_service.dart's box-corner mapping.
        std::array<std::pair<double, double>, 4> corners = {
            map_model_point_to_image(left, top, lb),
            map_model_point_to_image(left + box_w, top, lb),
            map_model_point_to_image(left + box_w, top + box_h, lb),
            map_model_point_to_image(left, top + box_h, lb),
        };
        double min_x = corners[0].first, max_x = corners[0].first;
        double min_y = corners[0].second, max_y = corners[0].second;
        for (const auto& [cx, cy] : corners) {
            min_x = std::min(min_x, cx);
            max_x = std::max(max_x, cx);
            min_y = std::min(min_y, cy);
            max_y = std::max(max_y, cy);
        }

        rubik::YoloPoseLabel label;
        label.cx = (min_x + max_x) / 2.0;
        label.cy = (min_y + max_y) / 2.0;
        label.w = max_x - min_x;
        label.h = max_y - min_y;

        for (int i = 0; i < 8; ++i) {
            double kx = out[0][5 + 3 * i];
            double ky = out[0][5 + 3 * i + 1];
            double kv_raw = out[0][5 + 3 * i + 2];
            auto [fx, fy] = map_model_point_to_image(kx, ky, lb);
            double v = kv_raw > 0.25 ? 2.0 : 0.0;
            label.keypoints.push_back({fx, fy, v});
        }

        fs::path label_path = out_labels / (image_path.stem().string() + ".txt");
        rubik::write_yolo_pose_label(label_path, label);

        fs::path target_image = out_images / image_path.filename();
        if (!fs::exists(target_image)) fs::copy_file(image_path, target_image);
        written++;
    }

    std::cout << "Pseudo labels written to: " << out_labels << " (" << written << " images)" << std::endl;
    return 0;
}
