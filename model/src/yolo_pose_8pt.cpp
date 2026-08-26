#include "rubik/yolo_pose_8pt.hpp"

#include <string>
#include <unordered_map>

namespace rubik {

namespace {
// Head branch hidden-channel formulas, matching ultralytics' Detect/Pose head
// (computed once from the smallest-stride level's input channels, ch0=64, and reused
// across all 3 levels): box hidden = max(16, ch0/4, reg_max*4); cls hidden = max(ch0,
// min(nc,100)); kpt hidden = max(ch0/4, nkpt*3).
constexpr int kCh0 = 64;
constexpr int kBoxHidden = 64;  // max(16, 16, 64)
constexpr int kClsHidden = 64;  // max(64, 1)
constexpr int kKptHidden = 24;  // max(16, 24)

torch::nn::Sequential make_head_branch(int c_in, int c_hidden, int c_out) {
    torch::nn::Sequential seq(Conv(c_in, c_hidden, 3), Conv(c_hidden, c_hidden, 3),
                               torch::nn::Conv2d(torch::nn::Conv2dOptions(c_hidden, c_out, 1)));
    return seq;
}
}  // namespace

YoloPose8ptImpl::YoloPose8ptImpl() {
    // Backbone
    stem = register_module("stem", Conv(3, 16, 3, 2));
    down1 = register_module("down1", Conv(16, 32, 3, 2));
    c2f_p2 = register_module("c2f_p2", C2f(32, 32, 1, true));
    down2 = register_module("down2", Conv(32, 64, 3, 2));
    c2f_p3 = register_module("c2f_p3", C2f(64, 64, 2, true));
    down3 = register_module("down3", Conv(64, 128, 3, 2));
    c2f_p4 = register_module("c2f_p4", C2f(128, 128, 2, true));
    down4 = register_module("down4", Conv(128, 256, 3, 2));
    c2f_p5 = register_module("c2f_p5", C2f(256, 256, 1, true));
    sppf = register_module("sppf", SPPF(256, 256, 5));

    // Neck (PAN-FPN)
    neck_c2f1 = register_module("neck_c2f1", C2f(256 + 128, 128, 1, false));
    neck_c2f2 = register_module("neck_c2f2", C2f(128 + 64, 64, 1, false));
    neck_down1 = register_module("neck_down1", Conv(64, 64, 3, 2));
    neck_c2f3 = register_module("neck_c2f3", C2f(64 + 128, 128, 1, false));
    neck_down2 = register_module("neck_down2", Conv(128, 128, 3, 2));
    neck_c2f4 = register_module("neck_c2f4", C2f(128 + 256, 256, 1, false));

    // Head
    dfl = register_module("dfl", DFL(kRegMax));
    box_head0 = register_module("box_head0", make_head_branch(64, kBoxHidden, 4 * kRegMax));
    box_head1 = register_module("box_head1", make_head_branch(128, kBoxHidden, 4 * kRegMax));
    box_head2 = register_module("box_head2", make_head_branch(256, kBoxHidden, 4 * kRegMax));
    cls_head0 = register_module("cls_head0", make_head_branch(64, kClsHidden, kNumClasses));
    cls_head1 = register_module("cls_head1", make_head_branch(128, kClsHidden, kNumClasses));
    cls_head2 = register_module("cls_head2", make_head_branch(256, kClsHidden, kNumClasses));
    kpt_head0 = register_module("kpt_head0", make_head_branch(64, kKptHidden, kNumKeypoints * 3));
    kpt_head1 = register_module("kpt_head1", make_head_branch(128, kKptHidden, kNumKeypoints * 3));
    kpt_head2 = register_module("kpt_head2", make_head_branch(256, kKptHidden, kNumKeypoints * 3));
    (void)kCh0;
}

PoseRawOutput YoloPose8ptImpl::forward(const torch::Tensor& x) {
    namespace F = torch::nn::functional;

    auto x0 = stem->forward(x);
    auto x1 = c2f_p2->forward(down1->forward(x0));
    auto p3 = c2f_p3->forward(down2->forward(x1));
    auto p4 = c2f_p4->forward(down3->forward(p3));
    auto p5 = sppf->forward(c2f_p5->forward(down4->forward(p4)));

    auto up_opts = F::InterpolateFuncOptions().scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest);
    auto u1 = F::interpolate(p5, up_opts);
    auto n1 = neck_c2f1->forward(torch::cat({u1, p4}, 1));

    auto u2 = F::interpolate(n1, up_opts);
    auto n2 = neck_c2f2->forward(torch::cat({u2, p3}, 1));  // stride 8 head input

    auto n3 = neck_c2f3->forward(torch::cat({neck_down1->forward(n2), n1}, 1));  // stride 16
    auto n4 = neck_c2f4->forward(torch::cat({neck_down2->forward(n3), p5}, 1));  // stride 32

    PoseRawOutput out;
    out.strides = {8, 16, 32};
    out.box = {box_head0->forward(n2), box_head1->forward(n3), box_head2->forward(n4)};
    out.cls = {cls_head0->forward(n2), cls_head1->forward(n3), cls_head2->forward(n4)};
    out.kpt = {kpt_head0->forward(n2), kpt_head1->forward(n3), kpt_head2->forward(n4)};
    return out;
}

namespace {
// The (h,w) of every FPN level, and therefore every anchor point/stride, never changes
// during a training run (fixed input resolution) -- rebuilding these small tensors from
// scratch every single batch is pure waste (allocation + kernel launches for something
// that's actually a constant). Cache by the shapes/device/offset that determine the
// result; single-threaded training loop, no locking needed.
struct AnchorCacheKey {
    std::vector<int64_t> shapes;  // h0,w0,h1,w1,...
    std::string device_str;
    double offset;
    bool operator==(const AnchorCacheKey& o) const {
        return shapes == o.shapes && device_str == o.device_str && offset == o.offset;
    }
};
struct AnchorCacheKeyHash {
    size_t operator()(const AnchorCacheKey& k) const {
        size_t h = std::hash<std::string>{}(k.device_str) ^ std::hash<double>{}(k.offset);
        for (int64_t s : k.shapes) h ^= std::hash<int64_t>{}(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
}  // namespace

Anchors make_anchors(const std::vector<torch::Tensor>& feature_maps, const std::vector<int64_t>& strides,
                      double offset) {
    static std::unordered_map<AnchorCacheKey, Anchors, AnchorCacheKeyHash> cache;

    AnchorCacheKey key;
    key.offset = offset;
    key.device_str = feature_maps.empty() ? "" : feature_maps[0].device().str();
    for (const auto& fm : feature_maps) {
        key.shapes.push_back(fm.size(2));
        key.shapes.push_back(fm.size(3));
    }
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    std::vector<torch::Tensor> points, stride_list;
    for (size_t i = 0; i < feature_maps.size(); ++i) {
        int64_t h = feature_maps[i].size(2), w = feature_maps[i].size(3);
        auto opts = feature_maps[i].options();
        auto sx = torch::arange(w, opts) + offset;
        auto sy = torch::arange(h, opts) + offset;
        auto grid = torch::meshgrid({sy, sx}, "ij");
        auto gy = grid[0], gx = grid[1];
        points.push_back(torch::stack({gx, gy}, -1).view({-1, 2}));
        stride_list.push_back(torch::full({h * w, 1}, static_cast<double>(strides[i]), opts));
    }
    Anchors out;
    out.points = torch::cat(points, 0);
    out.strides = torch::cat(stride_list, 0);
    cache.emplace(std::move(key), out);
    return out;
}

torch::Tensor dist2bbox(const torch::Tensor& ltrb, const torch::Tensor& anchor_points) {
    auto lt = ltrb.slice(-1, 0, 2);
    auto rb = ltrb.slice(-1, 2, 4);
    auto ap = anchor_points.unsqueeze(0);  // [1, A, 2], broadcasts over batch
    auto x1y1 = ap - lt;
    auto x2y2 = ap + rb;
    return torch::cat({x1y1, x2y2}, -1);  // [B, A, 4] xyxy, grid-cell units
}

torch::Tensor flatten_levels(const std::vector<torch::Tensor>& per_level) {
    std::vector<torch::Tensor> flat;
    flat.reserve(per_level.size());
    for (const auto& t : per_level) {
        auto f = t.flatten(2).transpose(1, 2);  // [B,C,H,W] -> [B, H*W, C]
        flat.push_back(f);
    }
    return torch::cat(flat, 1);
}

}  // namespace rubik
