#include "rubik/modules.hpp"

namespace rubik {

ConvImpl::ConvImpl(int c1, int c2, int k, int s, int p, int g) {
    if (p < 0) p = k / 2;
    conv = register_module(
        "conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k).stride(s).padding(p).groups(g).bias(false)));
    bn = register_module("bn", torch::nn::BatchNorm2d(c2));
}

torch::Tensor ConvImpl::forward(const torch::Tensor& x) { return torch::silu(bn(conv(x))); }

BottleneckImpl::BottleneckImpl(int c1, int c2, bool shortcut, int g, double e) {
    int c_hidden = static_cast<int>(c2 * e);
    cv1 = register_module("cv1", Conv(c1, c_hidden, 3, 1));
    cv2 = register_module("cv2", Conv(c_hidden, c2, 3, 1, -1, g));
    add = shortcut && c1 == c2;
}

torch::Tensor BottleneckImpl::forward(const torch::Tensor& x) {
    auto y = cv2(cv1(x));
    return add ? x + y : y;
}

C2fImpl::C2fImpl(int c1, int c2, int n, bool shortcut, int g, double e) {
    hidden_channels = static_cast<int>(c2 * e);
    cv1 = register_module("cv1", Conv(c1, 2 * hidden_channels, 1, 1));
    cv2 = register_module("cv2", Conv((2 + n) * hidden_channels, c2, 1, 1));
    m = register_module("m", torch::nn::ModuleList());
    for (int i = 0; i < n; ++i) m->push_back(Bottleneck(hidden_channels, hidden_channels, shortcut, g, 1.0));
}

torch::Tensor C2fImpl::forward(const torch::Tensor& x) {
    auto split = cv1(x).chunk(2, 1);
    std::vector<torch::Tensor> outs = {split[0], split[1]};
    torch::Tensor last = split[1];
    for (const auto& mod : *m) {
        last = mod->as<Bottleneck>()->forward(last);
        outs.push_back(last);
    }
    return cv2(torch::cat(outs, 1));
}

SPPFImpl::SPPFImpl(int c1, int c2, int k) {
    int c_hidden = c1 / 2;
    cv1 = register_module("cv1", Conv(c1, c_hidden, 1, 1));
    cv2 = register_module("cv2", Conv(c_hidden * 4, c2, 1, 1));
    pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
}

torch::Tensor SPPFImpl::forward(const torch::Tensor& x) {
    auto y0 = cv1(x);
    auto y1 = pool(y0);
    auto y2 = pool(y1);
    auto y3 = pool(y2);
    return cv2(torch::cat({y0, y1, y2, y3}, 1));
}

DFLImpl::DFLImpl(int reg_max_) : reg_max(reg_max_) {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(reg_max, 1, 1).bias(false)));
    torch::NoGradGuard no_grad;
    auto w = torch::arange(reg_max, torch::kFloat32).view({1, reg_max, 1, 1});
    conv->weight.copy_(w);
    conv->weight.set_requires_grad(false);
}

torch::Tensor DFLImpl::forward(const torch::Tensor& x) {
    auto sizes = x.sizes();
    int64_t B = sizes[0], A = sizes[2];
    // [B, 4*reg_max, A] -> [B, 4, reg_max, A] -> [B, reg_max, 4, A]
    auto y = x.view({B, 4, reg_max, A}).transpose(1, 2);
    y = torch::softmax(y, 1);
    y = conv(y);  // [B, 1, 4, A]
    return y.view({B, 4, A});
}

}  // namespace rubik
