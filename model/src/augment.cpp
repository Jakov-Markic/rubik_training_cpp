#include "rubik/augment.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/geometry/2d.hpp>  // OpenCV 5: getRotationMatrix2D moved out of imgproc.hpp
#include <opencv2/imgproc.hpp>

namespace rubik {

PoseSample load_and_letterbox(const cv::Mat& image_bgr, double cx, double cy, double w, double h,
                               const std::array<std::array<double, 3>, 8>& kpts_norm, int imgsz) {
    int ow = image_bgr.cols, oh = image_bgr.rows;
    int long_side = std::max(ow, oh);
    double scale = long_side <= imgsz ? 1.0 : static_cast<double>(imgsz) / long_side;
    int content_w = std::clamp(static_cast<int>(std::lround(ow * scale)), 1, imgsz);
    int content_h = std::clamp(static_cast<int>(std::lround(oh * scale)), 1, imgsz);

    cv::Mat content;
    if (content_w == ow && content_h == oh) {
        content = image_bgr;
    } else {
        cv::resize(image_bgr, content, cv::Size(content_w, content_h), 0, 0, cv::INTER_LINEAR);
    }
    int pad_x = (imgsz - content_w) / 2;
    int pad_y = (imgsz - content_h) / 2;

    PoseSample out;
    out.image = cv::Mat(imgsz, imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
    content.copyTo(out.image(cv::Rect(pad_x, pad_y, content_w, content_h)));

    auto map_x = [&](double nx) { return (nx * ow * scale + pad_x) / imgsz; };
    auto map_y = [&](double ny) { return (ny * oh * scale + pad_y) / imgsz; };

    double x1 = map_x(cx - w / 2.0), y1 = map_y(cy - h / 2.0);
    double x2 = map_x(cx + w / 2.0), y2 = map_y(cy + h / 2.0);
    out.cx = (x1 + x2) / 2.0;
    out.cy = (y1 + y2) / 2.0;
    out.w = x2 - x1;
    out.h = y2 - y1;

    for (int i = 0; i < 8; ++i) {
        out.kpts[i] = {map_x(kpts_norm[i][0]), map_y(kpts_norm[i][1]), kpts_norm[i][2]};
    }
    return out;
}

PoseSample apply_random_affine(const PoseSample& in, const AugmentConfig& cfg, std::mt19937& rng) {
    int S = cfg.imgsz;
    std::uniform_real_distribution<double> deg_dist(-cfg.degrees, cfg.degrees);
    std::uniform_real_distribution<double> scale_dist(1.0 - cfg.scale, 1.0 + cfg.scale);
    std::uniform_real_distribution<double> shear_dist(-cfg.shear, cfg.shear);
    std::uniform_real_distribution<double> trans_dist(-cfg.translate, cfg.translate);

    double angle = deg_dist(rng);
    double sc = std::max(0.1, scale_dist(rng));
    double shear_x = shear_dist(rng) * CV_PI / 180.0;
    double shear_y = shear_dist(rng) * CV_PI / 180.0;
    double tx = trans_dist(rng) * S;
    double ty = trans_dist(rng) * S;

    cv::Point2f center(S / 2.0f, S / 2.0f);
    cv::Mat R2x3 = cv::getRotationMatrix2D(center, angle, sc);
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat R2x3d;
    R2x3.convertTo(R2x3d, CV_64F);
    R2x3d.copyTo(R(cv::Rect(0, 0, 3, 2)));

    cv::Mat Sh = cv::Mat::eye(3, 3, CV_64F);
    Sh.at<double>(0, 1) = std::tan(shear_x);
    Sh.at<double>(1, 0) = std::tan(shear_y);

    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    T.at<double>(0, 2) = tx;
    T.at<double>(1, 2) = ty;

    cv::Mat M = T * Sh * R;  // 3x3, applied as: canvas_point = M * source_point

    cv::Mat warped;
    cv::warpAffine(in.image, warped, M(cv::Rect(0, 0, 3, 2)), cv::Size(S, S), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                    cv::Scalar(114, 114, 114));

    auto transform_point = [&](double nx, double ny) -> cv::Point2d {
        cv::Mat pt = (cv::Mat_<double>(3, 1) << nx * S, ny * S, 1.0);
        cv::Mat res = M * pt;
        return {res.at<double>(0, 0) / S, res.at<double>(1, 0) / S};
    };

    PoseSample out;
    out.image = warped;

    double x1 = in.cx - in.w / 2.0, y1 = in.cy - in.h / 2.0;
    double x2 = in.cx + in.w / 2.0, y2 = in.cy + in.h / 2.0;
    std::array<cv::Point2d, 4> corners = {transform_point(x1, y1), transform_point(x2, y1), transform_point(x2, y2),
                                           transform_point(x1, y2)};
    double min_x = corners[0].x, max_x = corners[0].x, min_y = corners[0].y, max_y = corners[0].y;
    for (const auto& p : corners) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    min_x = std::clamp(min_x, 0.0, 1.0);
    max_x = std::clamp(max_x, 0.0, 1.0);
    min_y = std::clamp(min_y, 0.0, 1.0);
    max_y = std::clamp(max_y, 0.0, 1.0);
    out.cx = (min_x + max_x) / 2.0;
    out.cy = (min_y + max_y) / 2.0;
    out.w = std::max(1e-6, max_x - min_x);
    out.h = std::max(1e-6, max_y - min_y);

    for (int i = 0; i < 8; ++i) {
        cv::Point2d p = transform_point(in.kpts[i][0], in.kpts[i][1]);
        double v = in.kpts[i][2];
        if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) v = 0.0;
        out.kpts[i] = {std::clamp(p.x, 0.0, 1.0), std::clamp(p.y, 0.0, 1.0), v};
    }
    return out;
}

void apply_hsv_jitter(cv::Mat& image, const AugmentConfig& cfg, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double rh = 1.0 + u(rng) * cfg.hsv_h;
    double rs = 1.0 + u(rng) * cfg.hsv_s;
    double rv = 1.0 + u(rng) * cfg.hsv_v;

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    ch[0].convertTo(ch[0], -1, rh, 0);
    ch[1].convertTo(ch[1], -1, rs, 0);
    ch[2].convertTo(ch[2], -1, rv, 0);
    cv::merge(ch, hsv);
    cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
}

void apply_random_erasing(cv::Mat& image, const AugmentConfig& cfg, std::mt19937& rng) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    if (prob(rng) > cfg.erasing_prob) return;

    int S = image.cols;
    std::uniform_real_distribution<double> area_frac(0.02, 0.2);
    std::uniform_real_distribution<double> aspect(0.3, 3.3);
    double area = area_frac(rng) * S * S;
    double ar = aspect(rng);
    int w = std::clamp(static_cast<int>(std::sqrt(area * ar)), 1, S);
    int h = std::clamp(static_cast<int>(std::sqrt(area / ar)), 1, S);
    std::uniform_int_distribution<int> xd(0, std::max(0, S - w));
    std::uniform_int_distribution<int> yd(0, std::max(0, S - h));
    int x = xd(rng), y = yd(rng);
    image(cv::Rect(x, y, std::min(w, S - x), std::min(h, S - y))).setTo(cv::Scalar(114, 114, 114));
}

PoseSample apply_mosaic(const std::array<PoseSample, 4>& samples, int imgsz, std::mt19937& rng) {
    int S = imgsz;
    cv::Mat canvas(2 * S, 2 * S, CV_8UC3, cv::Scalar(114, 114, 114));
    std::array<cv::Point, 4> offsets = {{{0, 0}, {S, 0}, {0, S}, {S, S}}};

    std::array<std::array<double, 4>, 4> boxes{};                  // per sub-image: cx,cy,w,h (normalized to 2S canvas)
    std::array<std::array<std::array<double, 3>, 8>, 4> all_kpts;  // per sub-image

    for (int i = 0; i < 4; ++i) {
        samples[i].image.copyTo(canvas(cv::Rect(offsets[i].x, offsets[i].y, S, S)));
        double ox = offsets[i].x, oy = offsets[i].y;
        boxes[i] = {(samples[i].cx * S + ox) / (2 * S), (samples[i].cy * S + oy) / (2 * S), samples[i].w * S / (2 * S),
                    samples[i].h * S / (2 * S)};
        for (int k = 0; k < 8; ++k) {
            double kx = (samples[i].kpts[k][0] * S + ox) / (2 * S);
            double ky = (samples[i].kpts[k][1] * S + oy) / (2 * S);
            all_kpts[i][k] = {kx, ky, samples[i].kpts[k][2]};
        }
    }

    // Uniform 2S->S downscale preserves normalized (fraction-of-canvas) coordinates
    // exactly, so no further coordinate adjustment is needed after resizing.
    PoseSample out;
    cv::resize(canvas, out.image, cv::Size(S, S), 0, 0, cv::INTER_LINEAR);

    std::uniform_int_distribution<int> pick(0, 3);
    int idx = pick(rng);
    out.cx = boxes[idx][0];
    out.cy = boxes[idx][1];
    out.w = boxes[idx][2];
    out.h = boxes[idx][3];
    out.kpts = all_kpts[idx];
    return out;
}

}  // namespace rubik
