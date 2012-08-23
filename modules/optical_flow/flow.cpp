#include "flow.hpp"

namespace vision {
namespace detail {

cv::Mat1b colorWheel()
{
    // Relative lengths of color transitions: these are chosen based on
    // perceptual similarity (e.g. one can distinguish more shades between red
    // and yellow than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int ncols = RY + YG + GC + CB + BM + MR;

    cv::Mat1b colorwheel(ncols, 3);

    int i;
    int k = 0;

    for (i = 0; i < RY; ++i, ++k) {
        colorwheel(k, 2) = 255;
        colorwheel(k, 1) = 255 * i / RY;
        colorwheel(k, 0) = 0;
    }

    for (i = 0; i < YG; ++i, ++k) {
        colorwheel(k, 2) = 255 - 255 * i / YG;
        colorwheel(k, 1) = 255;
        colorwheel(k, 0) = 0;
    }

    for (i = 0; i < GC; ++i, ++k) {
        colorwheel(k, 2) = 0;
        colorwheel(k, 1) = 255;
        colorwheel(k, 0) = 255 * i / GC;
    }

    for (i = 0; i < CB; ++i, ++k) {
        colorwheel(k, 2) = 0;
        colorwheel(k, 1) = 255 - 255 * i / CB;
        colorwheel(k, 0) = 255;
    }

    for (i = 0; i < BM; ++i, ++k) {
        colorwheel(k, 2) = 255 * i / BM;
        colorwheel(k, 1) = 0;
        colorwheel(k, 0) = 255;
    }

    for (i = 0; i < MR; ++i, ++k) {
        colorwheel(k, 2) = 255;
        colorwheel(k, 1) = 0;
        colorwheel(k, 0) = 255 - 255 * i / MR;
    }

    return colorwheel;
}

} // namespace detail
} // namespace vision
