// Train the single-stage 8-keypoint cube-corner YOLO-pose model from scratch in
// LibTorch. See README_CPP.md / the plan doc for what this does and doesn't promise:
// architecturally faithful to yolov8n-pose, but trains from random init (no pretrained
// checkpoint loading -- ultralytics' .pt files aren't loadable into this custom module
// graph), so convergence speed/quality is not expected to match the Python/ultralytics
// pipeline in train_cube_pose_8pt.py.
//
// After training, evaluates the best checkpoint on the val set and writes a handful of
// PNG/JPG plots into the run directory (alongside weights/) -- see graphs_readme.txt,
// also written there, for what each one shows.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include <CLI11.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include "rubik/dataset.hpp"
#include "rubik/evaluate.hpp"
#include "rubik/loss.hpp"
#include "rubik/plotting.hpp"

namespace fs = std::filesystem;

namespace {

struct EpochMetrics {
    double total = 0, box = 0, cls = 0, dfl = 0, kpt = 0;
};

EpochMetrics run_epoch(rubik::YoloPose8pt& model, const rubik::PoseDataset& dataset, const rubik::AugmentConfig& cfg,
                        int batch_size, torch::Device device, torch::optim::Optimizer* optimizer, std::mt19937& rng) {
    bool is_train = optimizer != nullptr;
    model->train(is_train);

    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (is_train) std::shuffle(indices.begin(), indices.end(), rng);

    EpochMetrics sum;
    int64_t num_batches = 0;

    for (size_t start = 0; start < indices.size(); start += batch_size) {
        size_t end = std::min(indices.size(), start + static_cast<size_t>(batch_size));
        std::vector<size_t> batch_idx(indices.begin() + start, indices.begin() + end);

        rubik::PoseBatch batch = dataset.get_batch(batch_idx, cfg, is_train, rng);
        auto images = batch.images.to(device);
        auto targets = batch.targets.to(device);

        torch::AutoGradMode grad_mode(is_train);
        auto raw = model->forward(images);
        auto loss = rubik::compute_loss(raw, targets, cfg.imgsz);

        if (is_train) {
            optimizer->zero_grad();
            loss.total.backward();
            optimizer->step();
        }

        sum.total += loss.total.item<double>();
        sum.box += loss.box.item<double>();
        sum.cls += loss.cls.item<double>();
        sum.dfl += loss.dfl.item<double>();
        sum.kpt += loss.kpt.item<double>();
        num_batches++;
    }
    if (num_batches > 0) {
        sum.total /= num_batches;
        sum.box /= num_batches;
        sum.cls /= num_batches;
        sum.dfl /= num_batches;
        sum.kpt /= num_batches;
    }
    return sum;
}

void write_graphs(const fs::path& run_dir, const std::vector<double>& epoch_x, const std::vector<EpochMetrics>& train_hist,
                   const std::vector<EpochMetrics>& val_hist, const std::vector<rubik::SampleEval>& val_eval) {
    auto series_of = [](const std::vector<EpochMetrics>& hist, double EpochMetrics::*field) {
        std::vector<double> out;
        out.reserve(hist.size());
        for (const auto& m : hist) out.push_back(m.*field);
        return out;
    };

    // results.png: train vs val total loss.
    cv::Mat results = rubik::plot_lines(
        "Training loss", "epoch", "loss", epoch_x,
        {
            {"train", series_of(train_hist, &EpochMetrics::total), {80, 80, 220}},
            {"val", series_of(val_hist, &EpochMetrics::total), {60, 160, 60}},
        });
    cv::imwrite((run_dir / "results.png").string(), results);

    // results_components.png: the four loss terms that sum to total (train set).
    cv::Mat components = rubik::plot_lines(
        "Training loss components", "epoch", "loss", epoch_x,
        {
            {"box (CIoU)", series_of(train_hist, &EpochMetrics::box), {80, 80, 220}},
            {"cls (BCE)", series_of(train_hist, &EpochMetrics::cls), {60, 160, 60}},
            {"dfl", series_of(train_hist, &EpochMetrics::dfl), {200, 140, 20}},
            {"kpt (OKS+vis)", series_of(train_hist, &EpochMetrics::kpt), {180, 40, 180}},
        });
    cv::imwrite((run_dir / "results_components.png").string(), components);

    if (!val_eval.empty()) {
        std::vector<double> confidences, ious;
        for (const auto& s : val_eval) {
            confidences.push_back(s.confidence);
            ious.push_back(s.iou);
        }
        cv::imwrite((run_dir / "confidence_hist.png").string(),
                    rubik::plot_histogram("Prediction confidence (val set)", "confidence", confidences, 20,
                                          {200, 120, 20}));
        cv::imwrite((run_dir / "iou_hist.png").string(),
                    rubik::plot_histogram("Box IoU vs ground truth (val set)", "IoU", ious, 20, {60, 160, 60}));

        auto pr_curve = rubik::compute_pr_curve(val_eval);
        std::vector<double> thresholds, precisions, recalls, f1s;
        for (const auto& p : pr_curve) {
            thresholds.push_back(p.threshold);
            precisions.push_back(p.precision);
            recalls.push_back(p.recall);
            f1s.push_back(p.f1);
        }
        cv::Mat pr_plot = rubik::plot_lines(
            "Precision / Recall / F1 vs confidence threshold", "confidence threshold", "score", thresholds,
            {
                {"precision", precisions, {80, 80, 220}},
                {"recall", recalls, {60, 160, 60}},
                {"F1", f1s, {20, 140, 220}},
            });
        cv::imwrite((run_dir / "pr_f1_curve.png").string(), pr_plot);

        std::array<double, 8> kpt_err_sum{};
        std::array<int, 8> kpt_err_count{};
        for (const auto& s : val_eval) {
            for (int k = 0; k < 8; ++k) {
                if (s.kpt_gt_visible[k]) {
                    kpt_err_sum[k] += s.kpt_error_px[k];
                    kpt_err_count[k]++;
                }
            }
        }
        std::vector<std::string> kpt_labels;
        std::vector<double> kpt_means;
        for (int k = 0; k < 8; ++k) {
            kpt_labels.push_back("kp" + std::to_string(k));
            kpt_means.push_back(kpt_err_count[k] > 0 ? kpt_err_sum[k] / kpt_err_count[k] : 0.0);
        }
        cv::imwrite((run_dir / "keypoint_error.png").string(),
                    rubik::plot_bar("Mean per-keypoint pixel error (val set, model input resolution)", "pixels",
                                    kpt_labels, kpt_means));

        cv::Mat sheet = rubik::render_prediction_contact_sheet(val_eval, 12);
        if (!sheet.empty()) cv::imwrite((run_dir / "val_predictions.jpg").string(), sheet, {cv::IMWRITE_JPEG_QUALITY, 90});
    }

    std::ofstream readme(run_dir / "graphs_readme.txt");
    readme <<
        "results.png              -- train vs val total loss per epoch. Both should trend down;\n"
        "                             val flattening/rising while train keeps falling = overfitting.\n"
        "results_components.png   -- the four loss terms that sum to the training total: box (CIoU,\n"
        "                             box shape/position), cls (BCE, is-there-a-cube-here confidence),\n"
        "                             dfl (distribution focal loss, box regression precision), kpt\n"
        "                             (OKS-style distance + visibility BCE for the 8 corners).\n"
        "confidence_hist.png      -- distribution of the model's confidence score on val images.\n"
        "                             Skewed toward 1.0 with few low-confidence predictions is what a\n"
        "                             well-trained model looks like.\n"
        "iou_hist.png             -- distribution of predicted-vs-ground-truth box IoU on val images.\n"
        "                             Skewed toward 1.0 is good; a lot of mass near 0 means the model\n"
        "                             often isn't finding the cube at all.\n"
        "pr_f1_curve.png          -- precision/recall/F1 as the confidence threshold used to accept a\n"
        "                             detection is swept from 0 to 1. Shows whether the model's own\n"
        "                             confidence score is trustworthy: ideally, correct detections get\n"
        "                             high confidence and wrong ones get low confidence, so precision\n"
        "                             stays high even as you raise the threshold, while recall only\n"
        "                             drops once you cut into genuinely-correct detections.\n"
        "keypoint_error.png       -- mean pixel error (at the model's input resolution) for each of\n"
        "                             the 8 corners individually, only over instances where that\n"
        "                             corner was labeled visible. Flags whether specific corners\n"
        "                             (e.g. consistently-occluded back-face ones) are weaker than\n"
        "                             others.\n"
        "val_predictions.jpg      -- a contact sheet of a few val images with ground truth (green) vs\n"
        "                             predicted (yellow) box + 8 keypoints drawn together, plus the\n"
        "                             confidence/IoU for that image (green text = correct, i.e. IoU >=\n"
        "                             0.5; red = not) -- the fastest way to eyeball what's actually\n"
        "                             going wrong instead of reading numbers.\n";
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Train the 8-keypoint cube-pose model from the YOLO-pose dataset "
                  "coco_keypoints_to_yolo_pose builds (LibTorch, from-scratch reimplementation)."};

    fs::path data_root;
    int epochs = 240;
    int imgsz = 640;
    int batch = 8;
    std::string device_str = "cpu";
    fs::path project = "runs_cpp/pose";
    std::string name = "cube_corners_8pt_cpp";
    int patience = 60;
    double lr0 = 0.001;
    double degrees = 18.0, translate = 0.08, scale = 0.25, shear = 2.0, mosaic = 0.15, erasing = 0.1;
    unsigned int seed = 42;

    app.add_option("--data", data_root, "YOLO-pose dataset root (images/labels/{train,val}, e.g. pose_dataset_combined_cvat_8pt)")
        ->required();
    app.add_option("--epochs", epochs);
    app.add_option("--imgsz", imgsz, "Training resolution.");
    app.add_option("--batch", batch);
    app.add_option("--device", device_str, "'cpu' or a CUDA device index (e.g. '0')");
    app.add_option("--project", project);
    app.add_option("--name", name);
    app.add_option("--patience", patience);
    app.add_option("--lr0", lr0);
    app.add_option("--degrees", degrees);
    app.add_option("--translate", translate);
    app.add_option("--scale", scale);
    app.add_option("--shear", shear);
    app.add_option("--mosaic", mosaic);
    app.add_option("--erasing", erasing);
    app.add_option("--seed", seed);

    fs::path eval_only_weights;
    app.add_option("--eval-only", eval_only_weights,
                   "Debug: skip training, load this checkpoint, run it on --data's val split, and print "
                   "each sample's raw confidence/box/keypoints. Used to cross-check the ultralytics bridge "
                   "(rubik_training_cpp/tools/bridge_to_ultralytics.py) against this model's own decode.");

    CLI11_PARSE(app, argc, argv);

    torch::Device device = torch::kCPU;
    if (device_str != "cpu") {
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA, std::stoi(device_str));
        } else {
            std::cerr << "CUDA not available in this LibTorch build; falling back to CPU." << std::endl;
        }
    }
    std::cout << "Device: " << device << std::endl;

    rubik::PoseDataset val_ds(data_root, "val");

    if (!eval_only_weights.empty()) {
        rubik::YoloPose8pt eval_model;
        torch::load(eval_model, eval_only_weights.string());
        eval_model->to(device);
        auto samples = rubik::evaluate_dataset(eval_model, val_ds, imgsz, batch, device, 5);
        int n = std::min<int>(5, static_cast<int>(samples.size()));
        for (int i = 0; i < n; ++i) {
            const auto& s = samples[i];
            std::cout << "SAMPLE " << i << " conf=" << s.confidence << " box=" << s.pred_box_norm[0] << ","
                      << s.pred_box_norm[1] << "," << s.pred_box_norm[2] << "," << s.pred_box_norm[3];
            for (int k = 0; k < 8; ++k) {
                std::cout << " kp" << k << "=" << s.pred_kpts_norm[k][0] << "," << s.pred_kpts_norm[k][1] << ","
                          << s.pred_kpts_norm[k][2];
            }
            std::cout << std::endl;
        }
        return 0;
    }

    rubik::PoseDataset train_ds(data_root, "train");
    std::cout << "train samples: " << train_ds.size() << ", val samples: " << val_ds.size() << std::endl;

    rubik::AugmentConfig cfg;
    cfg.imgsz = imgsz;
    cfg.degrees = degrees;
    cfg.translate = translate;
    cfg.scale = scale;
    cfg.shear = shear;
    cfg.mosaic_prob = mosaic;
    cfg.erasing_prob = erasing;

    rubik::AugmentConfig val_cfg = cfg;  // imgsz only matters for val (letterbox); randomness unused (is_train=false)

    rubik::YoloPose8pt model;
    model->to(device);

    torch::optim::AdamWOptions opt_opts(lr0);
    opt_opts.weight_decay(0.0005);
    torch::optim::AdamW optimizer(model->parameters(), opt_opts);

    fs::path run_dir = project / name;
    fs::path out_dir = run_dir / "weights";
    fs::create_directories(out_dir);

    std::ofstream results_csv(run_dir / "results.csv");
    results_csv << "epoch,train_total,train_box,train_cls,train_dfl,train_kpt,val_total,val_box,val_cls,val_dfl,val_kpt,lr\n";

    std::mt19937 rng(seed);
    double best_val_loss = std::numeric_limits<double>::infinity();
    int epochs_no_improve = 0;

    std::vector<double> epoch_x;
    std::vector<EpochMetrics> train_hist, val_hist;

    int last_epoch = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double lr = lr0 * 0.5 * (1.0 + std::cos(M_PI * epoch / std::max(1, epochs)));
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
        }

        EpochMetrics train_m = run_epoch(model, train_ds, cfg, batch, device, &optimizer, rng);
        EpochMetrics val_m = run_epoch(model, val_ds, val_cfg, batch, device, nullptr, rng);

        std::cout << "epoch " << (epoch + 1) << "/" << epochs << "  lr=" << lr << "  train_loss=" << train_m.total
                   << "  val_loss=" << val_m.total << std::endl;

        results_csv << (epoch + 1) << ',' << train_m.total << ',' << train_m.box << ',' << train_m.cls << ','
                    << train_m.dfl << ',' << train_m.kpt << ',' << val_m.total << ',' << val_m.box << ',' << val_m.cls
                    << ',' << val_m.dfl << ',' << val_m.kpt << ',' << lr << '\n';
        results_csv.flush();

        epoch_x.push_back(epoch + 1);
        train_hist.push_back(train_m);
        val_hist.push_back(val_m);
        last_epoch = epoch + 1;

        torch::save(model, (out_dir / "last.pt").string());
        if (val_m.total < best_val_loss) {
            best_val_loss = val_m.total;
            epochs_no_improve = 0;
            torch::save(model, (out_dir / "best.pt").string());
        } else if (++epochs_no_improve >= patience) {
            std::cout << "Early stopping (patience=" << patience << ")." << std::endl;
            break;
        }
    }

    std::cout << "Training finished after " << last_epoch << " epochs. Best val_loss=" << best_val_loss
              << ". Weights: " << out_dir << std::endl;

    std::cout << "Evaluating best checkpoint on val set for graphs..." << std::endl;
    torch::load(model, (out_dir / "best.pt").string());
    auto val_eval = rubik::evaluate_dataset(model, val_ds, imgsz, batch, device);

    write_graphs(run_dir, epoch_x, train_hist, val_hist, val_eval);
    std::cout << "Wrote graphs to " << run_dir << " (see graphs_readme.txt for what each one shows)." << std::endl;

    return 0;
}
