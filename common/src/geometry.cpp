#include "rubik/geometry.hpp"

namespace rubik {

namespace {
double cross(const Point2d& o, const Point2d& a, const Point2d& b) {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}
}  // namespace

bool segments_intersect(const Point2d& p1, const Point2d& p2, const Point2d& p3, const Point2d& p4) {
    double d1 = cross(p3, p4, p1);
    double d2 = cross(p3, p4, p2);
    double d3 = cross(p1, p2, p3);
    double d4 = cross(p1, p2, p4);
    if (((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0))) return true;
    return false;
}

bool quad_self_intersects(const std::array<Point2d, 4>& pts) {
    const auto& p0 = pts[0];
    const auto& p1 = pts[1];
    const auto& p2 = pts[2];
    const auto& p3 = pts[3];
    return segments_intersect(p0, p1, p2, p3) || segments_intersect(p1, p2, p3, p0);
}

double signed_area(const std::array<Point2d, 4>& pts) {
    double area = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& a = pts[i];
        const auto& b = pts[(i + 1) % pts.size()];
        area += a[0] * b[1] - b[0] * a[1];
    }
    return area / 2.0;
}

}  // namespace rubik
