#include "utility.hpp"

namespace vision {

cv::Rect bounds(const cv::Mat& image)
{
    return cv::Rect(0, 0, image.cols, image.rows);
}

template void VISION_EXPORT inflate<float>(cv::Rect_<float>& rect, float left,
    float top, float right, float bottom);
template void VISION_EXPORT inflate<double>(cv::Rect_<double>& rect,
    double left, double top, double right, double bottom);
template void VISION_EXPORT inflate<long double>(cv::Rect_<long double>& rect,
    long double left, long double top, long double right, long double bottom);

template void VISION_EXPORT inflate<float>(cv::Rect_<float>& rect, float cxy);
template void VISION_EXPORT inflate<double>(cv::Rect_<double>& rect, double cxy);
template void VISION_EXPORT inflate<long double>(cv::Rect_<long double>& rect,
    long double cxy);

} // namespace vision
