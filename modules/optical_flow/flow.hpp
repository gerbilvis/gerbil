/**
 * @file
 * @author Sergiu Dotenco
 * @brief Optical flow visualization.
 */

#ifndef VISION_FLOWCOLOR_HPP
#define VISION_FLOWCOLOR_HPP

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>

#include "colorconversion.hpp"
#include "utility.hpp"
#include "vision_export.hpp"

namespace vision {

//! @cond private_details

namespace detail {

cv::Mat1b VISION_EXPORT colorWheel();

template<class T>
inline cv::Vec3b flowColor(T u, T v)
{
    using std::sqrt;
    using std::atan2;
    using std::floor;
    using cv::saturate_cast;

    static const cv::Mat1b colorwheel = colorWheel();

    T magnitude = sqrt(u * u + v * v);
    T angle = atan2(-v, -u) / boost::math::constants::pi<T>();

    const int ncols = colorwheel.rows;

    T fk = (angle + 1) / 2 * (ncols - 1);
    int k0 = static_cast<int>(floor(fk));
    int k1 = (k0 + 1) % ncols;
    T f = fk - k0;
    //f = 0; // uncomment to see original color wheel

    cv::Vec3b value;

    for (int b = 0; b < 3; ++b) {
        T col0 = colorwheel(k0, b) / T((std::numeric_limits<uchar>::max)());
        T col1 = colorwheel(k1, b) / T((std::numeric_limits<uchar>::max)());
        T col = (1 - f) * col0 + f * col1;

        if (magnitude <= 1)
            col = 1 - magnitude * (1 - col); // increase saturation with radius
        else
            col *= T(0.75); // out of range

        value[b] = saturate_cast<uchar>((std::numeric_limits<uchar>::max)()
            * col);
    }

    return value;
}

template<class T>
inline bool isUnknownFlow(T u, T v, T threshold = T(1e9)) {

    return (std::abs(u) > threshold) || (std::abs(v) > threshold) ||
        isClose(std::numeric_limits<T>::quiet_NaN(), u) ||
        isClose(std::numeric_limits<T>::quiet_NaN(), v);
}

template<class T, class U, class F>
void drawDenseOpticalFlow(cv::Mat_<cv::Vec<T, 3> >& image, const cv::Mat_<U>& u,
    const cv::Mat_<U>& v, const F& functor)
{
    cv::Mat mag;
    cv::Mat angle;

    cv::magnitude(u, v, mag);
    cv::phase(u, v, angle);

    for (int x = 0; x < image.cols; ++x) {
        for (int y = 0; y < image.rows; ++y) {
            U l2 = mag.at<U>(y, x);
            U phi = angle.at<U>(y, x);

            T r, g, b;

            functor(phi, l2, r, g ,b);

            image(y, x)[2] = r;
            image(y, x)[1] = g;
            image(y, x)[0] = b;
        }
    }
}

template<class T, class U>
inline void drawDenseOpticalFlowHSV(U phi, U l2, T& r, T& g, T& b)
{
    T N = (std::numeric_limits<T>::max)();
    T h = cv::saturate_cast<T>(phi / boost::math::constants::two_pi<U>() * 180);
    T s = cv::saturate_cast<T>(l2 * N);

    HSVtoRGB(h, s, N, r, g, b);
}

template<class T, class U>
inline void drawDenseOpticalFlowHLS(U phi, U l2, T& r, T& g, T& b)
{
    T N = (std::numeric_limits<T>::max)();
    T h = cv::saturate_cast<T>(phi / boost::math::constants::two_pi<U>() * 180);
    T s = cv::saturate_cast<T>(l2 * N);

    HLStoRGB(h, s, N, r, g, b);
}

} // namespace detail

//! @endcond

template<class T>
inline void middleburyDenseFlow(cv::Mat3b& image, const cv::Mat_<T>& u,
    const cv::Mat_<T>& v,
    const boost::optional<T>& maxFlow)
{
    int width = image.cols, height = image.rows;
    int x, y;

    T max_u = -999, max_v = -999;
    T min_x = 999, min_y = 999;
    T max_magnitude = -1;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            T du = u(y, x);
            T dv = v(y, x);

            if (detail::isUnknownFlow(du, dv))
                continue;

            max_u = (std::max)(max_u, du);
            max_v = (std::max)(max_v, dv);
            min_x = (std::min)(min_x, du);
            min_y = (std::min)(min_y, dv);
            T magnitude = sqrt(du * du + dv * dv);
            max_magnitude = (std::max)(max_magnitude, magnitude);
        }
    }

    if (maxFlow)
        max_magnitude = *maxFlow;

    if (isClose(max_magnitude, T(0))) // if flow == 0 everywhere
        max_magnitude = T(1);

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            T du = u(y, x);
            T dv = v(y, x);

            if (detail::isUnknownFlow(du, dv))
                image(y, x) = cv::Vec3b(0, 0, 0);
            else
                image(y, x) = detail::flowColor(du / max_magnitude, dv / max_magnitude);
        }
    }
}

template<class T>
inline void middleburyDenseFlow(cv::Mat3b& image, const cv::Mat_<T>& u,
    const cv::Mat_<T>& v, T maxFlow)
{
    middleburyDenseFlow(image, u, v, boost::make_optional(maxFlow));
}

template<class T>
inline void middleburyDenseFlow(cv::Mat3b& image, const cv::Mat_<T>& u,
    const cv::Mat_<T>& v)
{
    middleburyDenseFlow(image, u, v, boost::optional<T>());
}

/**
 * @brief Draws a @e dense optical flow field using HSV color coding.
 *
 * @ingroup opticalflow
 */
template<class T, class U>
void drawDenseOpticalFlowHSV(cv::Mat_<cv::Vec<T, 3> >& image,
    const cv::Mat_<U>& u, const cv::Mat_<U>& v)
{
    detail::drawDenseOpticalFlow(image, u, v,
        detail::drawDenseOpticalFlowHSV<T, U>);
}

/**
 * @brief Draws a @e dense optical flow field using HLS color coding.
 *
 * @ingroup opticalflow
 */
template<class T, class U>
void drawDenseOpticalFlowHLS(cv::Mat_<cv::Vec<T, 3> >& image,
    const cv::Mat_<U>& u, const cv::Mat_<U>& v)
{
    detail::drawDenseOpticalFlow(image, u, v,
        detail::drawDenseOpticalFlowHLS<T, U>);
}

/**
 * @brief Draws a @e sparse optical flow field.
 *
 * @ingroup opticalflow
 */
template<class T, class U>
void drawSparseOpticalFlow(cv::Mat_<cv::Vec<T, 3> >& image, const cv::Mat_<U>& u,
    const cv::Mat_<U>& v, int step = 5)
{
    for (int x = 0; x < image.cols; x += step) {
        for (int y = 0; y < image.rows; y += step) {
            cv::Point start(x, y);
            cv::Point end(x + u(y, x), y + v(y, x));

            cv::line(image, start, end, CV_RGB(0, 0, 0));
        }
    }
}

} // namespace vision

#endif // VISION_FLOWCOLOR_HPP
