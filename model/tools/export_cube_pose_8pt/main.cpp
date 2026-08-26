// Exports a trained checkpoint. IMPORTANT: this writes a LibTorch-native archive
// (torch::save), loadable only via torch::load() back into a rubik::YoloPose8pt
// instance in this same C++ codebase -- it is NOT a TorchScript file, and cannot be
// consumed by Python/ultralytics or fed into the existing colab_export_tflite.py path.
//
// A LibTorch C++-trained torch::nn::Module doesn't export to a Python/TFLite-loadable
// TorchScript file as directly as a traced Python nn.Module does -- see README_CPP.md's
// "Deployment bridge" section. That bridge (pickle_save'ing named parameters for a
// Python-side script to copy into a matching ultralytics.YOLO instance) is a documented
// follow-up, not implemented here -- shipping a working, honestly-labeled checkpoint
// export beats shipping an unverified TorchScript-tracing attempt.

#include <filesystem>
#include <fstream>
#include <iostream>

#include <CLI11.hpp>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include "rubik/yolo_pose_8pt.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    CLI::App app{"Export a trained cube-pose checkpoint to a LibTorch archive."};

    fs::path weights, output, pickle_out;
    bool dump_params = false;
    app.add_option("--weights", weights, "Trained checkpoint, e.g. runs_cpp/pose/<name>/weights/best.pt");
    app.add_option("--output", output, "Output path for the exported LibTorch archive");
    app.add_option("--pickle-out", pickle_out,
                   "Write the full state (parameters + BatchNorm buffers) as a Python-loadable pickle "
                   "file -- {name: tensor}, own module names -- for the ultralytics bridge "
                   "(see rubik_training_cpp/tools/bridge_to_ultralytics.py). Requires --weights.");
    app.add_flag("--dump-params", dump_params,
                 "Debug: print this model's named_parameters() (name + shape, insertion order) and exit. "
                 "Used to verify the ultralytics bridge mapping -- doesn't need --weights/--output.");

    CLI11_PARSE(app, argc, argv);

    rubik::YoloPose8pt model;

    if (dump_params) {
        int64_t total = 0;
        for (const auto& p : model->named_parameters()) {
            std::cout << p.key() << " (";
            auto sizes = p.value().sizes();
            for (size_t i = 0; i < sizes.size(); ++i) std::cout << sizes[i] << (i + 1 < sizes.size() ? ", " : "");
            std::cout << ")" << std::endl;
            total++;
        }
        std::cout << "TOTAL PARAMS: " << total << std::endl;
        return 0;
    }

    if (weights.empty()) {
        std::cerr << "--weights is required (unless --dump-params)" << std::endl;
        return 1;
    }

    torch::load(model, weights.string());
    model->eval();

    if (!output.empty()) {
        fs::create_directories(output.parent_path());
        torch::save(model, output.string());
        std::cout << "Wrote LibTorch archive: " << output << std::endl;
        std::cout << "NOTE: not a TorchScript file -- loadable only via torch::load() into a "
                     "rubik::YoloPose8pt instance in this codebase, not by Python/TFLite. See "
                     "README_CPP.md's 'Deployment bridge' section."
                  << std::endl;
    }

    if (!pickle_out.empty()) {
        // Full state (learnable params + BatchNorm running_mean/var/num_batches_tracked) --
        // eval-mode inference needs the BN buffers too, not just weight/bias, or the bridged
        // model would use freshly-initialized (garbage) running statistics.
        c10::Dict<std::string, at::Tensor> state;
        for (const auto& p : model->named_parameters()) state.insert(p.key(), p.value());
        for (const auto& b : model->named_buffers()) state.insert(b.key(), b.value());

        torch::IValue ivalue(state);
        std::vector<char> bytes = torch::jit::pickle_save(ivalue);

        fs::create_directories(pickle_out.parent_path());
        std::ofstream out(pickle_out, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

        std::cout << "Wrote pickle state (" << state.size() << " entries): " << pickle_out << std::endl;
        std::cout << "Load in Python with: torch.load('" << pickle_out.filename().string() << "')" << std::endl;
    }

    return 0;
}
