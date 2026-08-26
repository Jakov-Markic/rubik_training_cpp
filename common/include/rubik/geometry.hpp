#pragma once

#include <array>

namespace rubik {

using Point2d = std::array<double, 2>;

// True if segment p1-p2 properly crosses segment p3-p4 (mirrors the orientation-sign
// test in check_keypoint_label_consistency.py's segments_intersect).
bool segments_intersect(const Point2d& p1, const Point2d& p2, const Point2d& p3, const Point2d& p4);

// pts = [p0, p1, p2, p3] in intended traversal order (edges 0-1, 1-2, 2-3, 3-0).
// A simple (non-self-intersecting) quad's only possible crossing is between the two
// non-adjacent edge pairs: (0-1 vs 2-3) and (1-2 vs 3-0).
bool quad_self_intersects(const std::array<Point2d, 4>& pts);

double signed_area(const std::array<Point2d, 4>& pts);

}  // namespace rubik
