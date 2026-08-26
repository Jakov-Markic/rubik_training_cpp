#pragma once

#include <torch/torch.h>

// Building blocks matching ultralytics' YOLOv8 backbone/neck/head modules
// (nn/modules/{conv,block}.py), reimplemented against LibTorch's C++ nn API.

namespace rubik {

// Conv2d (no bias) + BatchNorm2d + SiLU. autopad: p = k/2 when not given.
struct ConvImpl : torch::nn::Module {
    ConvImpl(int c1, int c2, int k = 1, int s = 1, int p = -1, int g = 1);
    torch::Tensor forward(const torch::Tensor& x);

    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
};
TORCH_MODULE(Conv);

// Two 3x3 Convs with an optional residual add (when c1 == c2 and shortcut is requested).
struct BottleneckImpl : torch::nn::Module {
    BottleneckImpl(int c1, int c2, bool shortcut = true, int g = 1, double e = 0.5);
    torch::Tensor forward(const torch::Tensor& x);

    Conv cv1{nullptr}, cv2{nullptr};
    bool add;
};
TORCH_MODULE(Bottleneck);

// CSP-style block: split into two halves via a 1x1 conv, chain n Bottlenecks off one
// half, concat every intermediate output with both halves, 1x1 conv to fuse.
struct C2fImpl : torch::nn::Module {
    C2fImpl(int c1, int c2, int n = 1, bool shortcut = false, int g = 1, double e = 0.5);
    torch::Tensor forward(const torch::Tensor& x);

    Conv cv1{nullptr}, cv2{nullptr};
    torch::nn::ModuleList m;
    int hidden_channels;
};
TORCH_MODULE(C2f);

// Spatial Pyramid Pooling - Fast: 1x1 reduce, 3x sequential maxpool(k=5,s=1) concat, 1x1 expand.
struct SPPFImpl : torch::nn::Module {
    SPPFImpl(int c1, int c2, int k = 5);
    torch::Tensor forward(const torch::Tensor& x);

    Conv cv1{nullptr}, cv2{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
};
TORCH_MODULE(SPPF);

// Distribution Focal Loss decode head: a non-trainable 1x1 conv computing the
// softmax-weighted expectation over reg_max bins, turning a [B,4*reg_max,A] box
// logit tensor into [B,4,A] ltrb distances (in grid-cell units).
struct DFLImpl : torch::nn::Module {
    explicit DFLImpl(int reg_max = 16);
    torch::Tensor forward(const torch::Tensor& x);

    torch::nn::Conv2d conv{nullptr};
    int reg_max;
};
TORCH_MODULE(DFL);

}  // namespace rubik
