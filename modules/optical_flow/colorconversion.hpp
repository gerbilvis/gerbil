/**
 * @file
 * @brief Color conversion functionality.
 * @author Sergiu Dotenco
 * @ingroup colorconversion
 */

#ifndef VISION_COLORCONVERSION_HPP
#define VISION_COLORCONVERSION_HPP

#pragma once

#ifdef HAVE_TBB
#include <tbb/atomic.h>
#include <tbb/mutex.h>
#endif // HAVE_TBB

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <boost/cstdint.hpp>
#include <boost/static_assert.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

#include "utility.hpp"

namespace vision {

//! @cond private_details

namespace detail {

const boost::uint8_t saturateLut[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
    93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
    169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
    199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
    244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

template<class T>
inline boost::uint8_t fast_cast(T t)
{
    assert(-256 <= (t) || (t) <= 512);
    return saturateLut[(t) + 256];
}

template<class T>
inline void calc_min(T& a, T b)
{
    a -= fast_cast(a - b);
}

template<class T>
inline void calc_max(T& a, T b)
{
    a += fast_cast(b - a);
}

} // namespace detail

//! @endcond

/**
 * @brief Color value conversion traits.
 *
 * @ingroup colorconversion
 */
template<class T, class E = void>
struct ColorValueTraits
{
    static T (max)()
    {
        return (std::numeric_limits<T>::max)();
    }
};

//! @cond private_details

template<class T>
struct ColorValueTraits<T,
    typename boost::enable_if<
        boost::is_floating_point<T>
    >::type
>
{
    static T (max)()
    {
        return T(1);
    }
};

/**
 * @brief Converts an RGB color value to HSV.
 *
 * The conversion of RGB components @f$R, G, B \in [0, C_{\max}]@f$ is performed
 * according to the following rules. First, the values
 * @f[
 *  C_{\text{high}}=\max(R,G,B),
 *  \quad
 *  C_{\text{low}}=\min(R,G,B),
 *  \quad
 *  C_{\text{rng}}=C_{\text{high}}-C_{\text{low}}
 * @f]
 * are computed. Then, the saturation
 * @f[
 *  S_{\text{HSV}}=
 *  \begin{cases}
 *    \dfrac{C_{\text{rng}}}{C_{\text{high}}} & C_{\text{high}} > 0 \\
 *    0 & C_{\text{high}} \leq 0
 *  \end{cases}
 * @f]
 * and the value
 * @f[
 *  V_{\text{HSV}}=\frac{C_{\text{high}}}{C_{\text{max}}}
 * @f]
 * If @f$S_{\text{HSV}}=0@f$, then @f$H_{\text{HSV}}@f$ is undefined. Otherwise,
 * @f[
 *  R'=\frac{C_{\text{high}}-R}{C_{\text{rng}}},
 *  \quad
 *  G'=\frac{C_{\text{high}}-G}{C_{\text{rng}}},
 *  \quad
 *  B'=\frac{C_{\text{high}}-B}{C_{\text{rng}}}
 * @f]
 * followed by
 * @f[
 *  H'=
 *  \begin{cases}
 *    B'-G' & R=C_{\text{high}} \\
 *    R'-B'+2 & G=C_{\text{high}} \\
 *    G'-R'+4 & B=C_{\text{high}}
 *  \end{cases}
 * @f]
 * has to be computed. The resulting @f$H'@f$ value falls into the @f$[-1,5]@f$
 * range which is normalized to @f$[0,1]@f$ using
 * @f[
 *  H_{\text{HSV}} =
 *  \frac{1}{6}
 *  \begin{cases}
 *    H'+6 & H' < 0 \\
 *    H' & H' \geq 0
 *  \end{cases}
 * @f]
 *
 * @param r The red component of the pixel.
 * @param g The green component of the pixel.
 * @param b The blue component of the pixel.
 * @param[out] h The resulting hue value.
 * @param[out] s The resulting saturation value.
 * @param[out] v The resulting brightness value.
 *
 * @ingroup colorconversion
 */
template<class T, class RealType>
inline void RGBtoHSV(T r, T g, T b, RealType& h, RealType& s, RealType& v,
    T high = (ColorValueTraits<T>::max)())
{
    BOOST_STATIC_ASSERT_MSG(boost::is_floating_point<RealType>::value,
        "Color conversion parameter type is required to be floating point");

    // allow ADL
    using std::min;
    using std::max;

    T minimum = (min)(b, (min)(g, r));
    T maximum = (max)(b, (max)(g, r));
    T c = maximum - minimum;

    s = maximum > 0 ? RealType(c) / maximum : 0;
    v = RealType(maximum) / high;

    if (c > 0) {
        RealType r0 = (RealType(maximum) - r) / c;
        RealType g0 = (RealType(maximum) - g) / c;
        RealType b0 = (RealType(maximum) - b) / c;

        RealType h0;

        if (r == maximum)
            h0 = b0 - g0;
        else if (g == maximum)
            h0 = r0 - b0 + 2;
        else
            h0 = g0 - r0 + 4;

        h = (h0 < 0 ? h0 + 6 : h0) / RealType(6);
    }
    else
        h = 0;

    assert(h >= 0 && h <= 1);
    assert(s >= 0 && s <= 1);
    assert(v >= 0 && v <= 1);
}

/**
 * @brief Converts a HSV color value to RGB.
 *
 * A HSV tuple @f$(H_{\text{HSV}}, S_{\text{HSV}}, V_{\text{HSV}})@f$ with
 * @f$H_{\text{HSV}}, S_{\text{HSV}} \in [0,1]@f$ is converted to the RGB color
 * space using
 * @f[
 *   H'= 6\cdot H_{\text{HSV}} \bmod 6
 * @f]
 * with @f$0\leq H' < 6@f$ and then
 * @f[
 *   c_1 = \lfloor H' \rfloor,
 *   \quad
 *   c_2 = H' - c_1
 * @f]
 * and
 * @f[
 *   \begin{split}
 *     x &= (1-S_{\text{HSV}})\cdot V_{\text{HSV}},
 *     \\
 *     y &= (1-S_{\text{HSV}}\cdot c_2) \cdot V_{\text{HSV}},
 *     \\
 *     z &= (1-S_{\text{HSV}}\cdot(1-c_2))\cdot V_{\text{HSV}}
 *   \end{split}
 * @f]
 * Normalized RGB values @f$R',G',B'\in[0,1]@f$ are calculated in dependence
 * from @f$c_1@f$ and @f$v=V_{\text{HSV}},x,y@f$ and @f$z@f$ according to:
 * @f[
 *   (R',G',B')
 *   \gets
 *   \begin{cases}
 *     (v,z,x) & \text{if}~c_1=0
 *     \\
 *     (y,v,x) & \text{if}~c_1=1
 *     \\
 *     (x,v,z) & \text{if}~c_1=2
 *     \\
 *     (x,y,v) & \text{if}~c_1=3
 *     \\
 *     (z,x,v) & \text{if}~c_1=4
 *     \\
 *     (v,x,y) & \text{if}~c_1=5
 *   \end{cases}
 * @f]
 * Finally, the RGB components are scaled to @f$[0,N-1]@f$ using
 * @f[
 *   \begin{split}
 *     R &\gets \min(\operatorname{round}(N\cdot R'),N-1)
 *     \\
 *     G &\gets \min(\operatorname{round}(N\cdot G'),N-1)
 *     \\
 *     B &\gets \min(\operatorname{round}(N\cdot B'),N-1)
 *   \end{split}
 * @f]
 *
 * @param h The hue value.
 * @param s The saturation value.
 * @param v The brightness value.
 * @param[out] r The resulting red value.
 * @param[out] g The resulting green value.
 * @param[out] b The resulting blue value.
 *
 * @ingroup colorconversion
 */
template<class RealType, class T>
inline void HSVtoRGB(RealType h, RealType s, RealType v, T& r, T& g, T& b,
                     T N = (ColorValueTraits<T>::max)())
{
    BOOST_STATIC_ASSERT_MSG(boost::is_floating_point<RealType>::value,
        "Color conversion parameter type is required to be floating point");

    using std::fmod; // allow ADL
    using std::min;

    RealType rr = 0, gg = 0, bb = 0;
    RealType hh = fmod(RealType(6) * h, RealType(6));
    T c1 = static_cast<T>(hh);
    RealType c2 = hh - c1;
    RealType x = (1 - s) * v;
    RealType y = (1 - (s * c2)) * v;
    RealType z = (1 - (s * (1 - c2))) * v;

    assert(c1 >= 0 && c1 < 6);

    switch (c1) {
        case 0:
            rr = v; gg = z; bb = x;
            break;
        case 1:
            rr = y; gg = v; bb = x;
            break;
        case 2:
            rr = x; gg = v; bb = z;
            break;
        case 3:
            rr = x; gg = y; bb = v;
            break;
        case 4:
            rr = z; gg = x; bb = v;
            break;
        case 5:
            rr = v; gg = x; bb = y;
            break;
#ifdef _MSC_VER
        default:
            __assume(0); // cannot be reached
#endif // _MSC_VER
    }

    r = (min<T>)(round<T>(rr * N), N);
    g = (min<T>)(round<T>(gg * N), N);
    b = (min<T>)(round<T>(bb * N), N);
}

//! @endcond

/**
 * @brief Converts an RGB color value to HSV.
 *
 * @param[out] h The resulting hue value.
 * @param[out] s The resulting saturation value.
 * @param[out] v The resulting brightness value.
 *
 * @ingroup colorconversion
 */
inline void RGBtoHSV(boost::uint8_t r, boost::uint8_t g, boost::uint8_t b,
              boost::uint8_t& h, boost::uint8_t& s, boost::uint8_t& v)
{
    using namespace detail;
    using cv::saturate_cast;

    const int hsv_shift = 12;
    static int sdiv_table[256];
    static int hdiv_table180[256];

#ifdef HAVE_TBB
    static tbb::atomic<bool> initialized;
#else // HAVE_TBB
    static volatile bool initialized = false;
#endif // HAVE_TBB

    const int* const hdiv_table = hdiv_table180;
    const int hr = 180;

    if (!initialized) {
#if defined(HAVE_TBB)
        tbb::mutex::scoped_lock lock;
#elif defined(_ATL)
        WTL::CStaticDataInitCriticalSectionLock lock;
        lock.Lock();
#endif // defined(HAVE_TBB)

        if (!initialized) {
            initialized = true;

            for (std::size_t i = 1; i != 256; ++i) {
                sdiv_table[i] = saturate_cast<int>((255 << hsv_shift) / (1.0 *
                    i));
                hdiv_table180[i] = saturate_cast<int>((180 << hsv_shift) /
                    (6.0 * i));
            }
        }

#if !defined(HAVE_TBB) && defined(_ATL)
        lock.Unlock();
#endif // !defined(HAVE_TBB) && defined(_ATL)
    }

    int h2, s2, v2 = b;
    int vmin = b, diff;
    int vr, vg;
    calc_max(v2, static_cast<int>(g));
    calc_max(v2, static_cast<int>(r));
    calc_min(vmin, static_cast<int>(g));
    calc_min(vmin, static_cast<int>(r));
    diff = v2 - vmin;
    vr = v2 == r ? -1 : 0;
    vg = v2 == g ? -1 : 0;
    s2 = (diff * sdiv_table[v2] + (1 << (hsv_shift - 1))) >> hsv_shift;
    h2 = (vr & (g - b)) +
        (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h2 = (h2 * hdiv_table[diff] + (1 << (hsv_shift - 1))) >> hsv_shift;
    h2 += h2 < 0 ? hr : 0;
    h = saturate_cast<boost::uint8_t>(h2);
    s = saturate_cast<boost::uint8_t>(s2);
    v = saturate_cast<boost::uint8_t>(v2);
}

/**
 * @brief Converts a HSV color value to RGB.
 *
 * @param h The hue value.
 * @param s The saturation value.
 * @param v The brightness value.
 * @param[out] r1 The resulting red value.
 * @param[out] g1 The resulting green value.
 * @param[out] b1 The resulting blue value.
 *
 * @ingroup colorconversion
 */
inline void HSVtoRGB(boost::uint8_t h, boost::uint8_t s, boost::uint8_t v,
                     boost::uint8_t& r1, boost::uint8_t& g1, boost::uint8_t& b1)
{
    using cv::saturate_cast;

    const double scale = (std::numeric_limits<boost::uint8_t>::max)();
    double h2 = h;
    double s2 = s / scale;
    double v2 = v / scale;
    const double hscale = 6.0 / 180.0;
    double b, g, r;

    if (std::abs(s2) <= std::numeric_limits<double>::epsilon())
        b = g = r = v2;
    else {
        static const int sector_data[][3] = {
            { 1, 3, 0 },
            { 1, 0, 2 },
            { 3, 0, 1 },
            { 0, 2, 1 },
            { 0, 1, 3 },
            { 2, 1, 0 }
        };

        double tab[4];
        int sector;
        h2 *= hscale;

        if (h2 < 0) {
            do {
                h2 += 6;
            }
            while (h2 < 0);
        }
        else if (h2 >= 6) {
            do {
                h2 -= 6;
            }
            while (h2 >= 6);
        }

        sector = static_cast<int>(std::floor(h2));
        h2 -= sector;

        tab[0] = v2;
        tab[1] = v2 * (1.0 - s2);
        tab[2] = v2 * (1.0 - s2 * h2);
        tab[3] = v2 * (1.0 - s2 * (1.0 - h2));

        b = tab[sector_data[sector][0]];
        g = tab[sector_data[sector][1]];
        r = tab[sector_data[sector][2]];
    }

    r1 = saturate_cast<boost::uint8_t>(r * scale);
    g1 = saturate_cast<boost::uint8_t>(g * scale);
    b1 = saturate_cast<boost::uint8_t>(b * scale);
}

/**
 * @brief Converts an RGB color value to HLS.
 *
 * The conversion is of the RGB components @f$R, G, B \in [0, C_{\max}]@f$ is
 * performed according to the following rules. First, the values
 * @f[
 *  C_{\text{high}}=\max(R,G,B),
 *  \quad
 *  C_{\text{low}}=\min(R,G,B),
 *  \quad
 *  C_{\text{rng}}=C_{\text{high}}-C_{\text{low}}
 * @f]
 * are computed. If @f$S_{\text{HLS}}=0@f$, then @f$H_{\text{HLS}}@f$ is
 * undefined. Otherwise,
 * @f[
 *  R'=\frac{C_{\text{high}}-R}{C_{\text{rng}}},
 *  \quad
 *  G'=\frac{C_{\text{high}}-G}{C_{\text{rng}}},
 *  \quad
 *  B'=\frac{C_{\text{high}}-B}{C_{\text{rng}}}
 * @f]
 * followed by
 * @f[
 *  H'=
 *  \begin{cases}
 *    B'-G' & R=C_{\text{high}} \\
 *    R'-B'+2 & G=C_{\text{high}} \\
 *    G'-R'+4 & B=C_{\text{high}}
 *  \end{cases}
 * @f]
 * has to be computed. The resulting @f$H'@f$ value falls into the @f$[-1,5]@f$
 * range which is normalized to @f$[0,1]@f$ using
 * @f[
 *  H_{\text{HLS}}
 *  \gets
 *  \frac{1}{6}
 *  \begin{cases}
 *    H'+6 & H' < 0 \\
 *    H' & H' \geq 0
 *  \end{cases}
 * @f]
 * The remaining values are computed according to
 * @f[
 *   L_{\text{HLS}}
 *   \gets
 *   \frac{C_{\text{high}}+C_{\text{low}}}{2}
 * @f]
 * and
 * @f[
 *   S_{\text{HLS}}
 *   \gets
 *   \begin{dcases}
 *     0 & L_{\text{HLS}}=0
 *     \\
 *     0.5 \cdot
 *       \frac{C_{\text{rng}}}{L_{\text{HLS}}} & 0 < L_{\text{HLS}} \leq 0.5
 *     \\
 *     0.5 \cdot
 *       \frac{C_{\text{rng}}}{1-L_{\text{HLS}}} & 0.5 < L_{\text{HLS}} < 1
 *     \\
 *     0 & L_{\text{HLS}}=1
 *   \end{dcases}
 * @f]
 *
 * @param r The red component of the pixel.
 * @param g The green component of the pixel.
 * @param b The blue component of the pixel.
 * @param[out] H The resulting hue value.
 * @param[out] L The resulting lightness value.
 * @param[out] S The resulting saturation value.
 *
 * @ingroup colorconversion
 */
template<class T, class RealType>
inline void RGBtoHLS(T r, T g, T b, RealType& H, RealType& L, RealType& S,
                     T high = (ColorValueTraits<T>::max)())
{
    BOOST_STATIC_ASSERT_MSG(boost::is_floating_point<RealType>::value,
        "Color conversion parameter type is required to be floating point");

    using std::max;
    using std::min;

    RealType R = r / RealType(high);
    RealType G = g / RealType(high);
    RealType B = b / RealType(high);

    // R,G,B assumed to be in [0,1]
    RealType maximum = (max)(R, (max)(G, B)); // highest color value
    RealType minimum = (min)(R, (min)(G, B)); // lowest color value
    RealType range = maximum - minimum;
    L = (RealType(maximum) + minimum) / RealType(2);
    S = 0;

    if (0 < L && L < 1) {
        RealType d = L <= RealType(0.5) ? L : (1 - L);
        S = RealType(0.5) * range / d;
    }

    H = 0;

    if (maximum > 0 && range > 0) {
        RealType rr = maximum - R / range;
        RealType gg = maximum - G / range;
        RealType bb = maximum - B / range;
        RealType hh;

        if (R == maximum)
            hh = bb - gg;
        else if (G == maximum)
            hh = rr - bb + 2;
        else
            hh = gg - rr + 4;

        H = (hh < 0 ? hh + 6 : hh) / 6;
    }
}

/**
 * @brief Converts an RGB color value to HLS.
 *
 * An HLS tuple @f$(H_{\text{HLS}}, L_{\text{HLS}}, S_{\text{HLS}})@f$ is
 * assumed to be @f$H_{\text{HLS}}, L_{\text{HLS}}, S_{\text{HLS}} \in [0,1]@f$.
 * If @f$L_{\text{HLS}}=0@f$ or @f$L_{\text{HLS}}=1@f$, then the resulting RGB
 * color is
 * @f[
 *   (R',G',B') \gets
 *   \begin{cases}
 *     (0,0,0) & L_{\text{HLS}}=0
 *     \\
 *     (1,1,1) & L_{\text{HLS}}=1
 *   \end{cases}
 * @f]
 * Otherwise the appropriate color sector has to be calculated first using
 * @f[
 *   H'= 6\cdot H_{\text{HSV}} \bmod 6
 * @f]
 * with @f$0\leq H' < 6@f$ and then
 * @f[
 *   c_1 = \lfloor H' \rfloor,
 *   \quad
 *   c_2 = H' - c_1,
 *   \quad
 *   d =
 *   \begin{cases}
 *     S_{\text{HLS}}\cdot L_{\text{HLS}} & L_{\text{HLS}} \leq 0.5
 *     \\
 *     S_{\text{HLS}}\cdot (L_{\text{HLS}}-1) & L_{\text{HLS}} > 0.5
 *   \end{cases}
 * @f]
 * and
 * @f[
 *     w = L_{\text{HLS}}+d,
 *     \quad
 *     x = L_{\text{HLS}}-d,
 *     \quad
 *     y = w-(w-x)\cdot c_2,
 *     \quad
 *     z = x+(w-x)\cdot c_2
 * @f]
 * Normalized RGB values @f$R',G',B'\in[0,1]@f$ are calculated according to:
 * @f[
 *   (R',G',B')
 *   \gets
 *   \begin{cases}
 *     (w,z,x) & \text{if}~c_1=0
 *     \\
 *     (y,w,x) & \text{if}~c_1=1
 *     \\
 *     (x,w,z) & \text{if}~c_1=2
 *     \\
 *     (x,y,w) & \text{if}~c_1=3
 *     \\
 *     (z,x,w) & \text{if}~c_1=4
 *     \\
 *     (w,x,y) & \text{if}~c_1=5
 *   \end{cases}
 * @f]
 * Finally, the RGB components are scaled to @f$[0,N-1]@f$ using
 * @f[
 *   \begin{split}
 *     R &\gets \min(\operatorname{round}(N\cdot R'),N-1)
 *     \\
 *     G &\gets \min(\operatorname{round}(N\cdot G'),N-1)
 *     \\
 *     B &\gets \min(\operatorname{round}(N\cdot B'),N-1)
 *   \end{split}
 * @f]
 *
 * @param H The hue value.
 * @param L The lightness value.
 * @param S The saturation value.
 * @param[out] r The resulting red value.
 * @param[out] g The resulting green value.
 * @param[out] b The resulting blue value.
 *
 * @ingroup colorconversion
 */
template<class RealType, class T>
inline void HLStoRGB (RealType H, RealType L, RealType S, T& r, T& g, T& b,
                      T high = (ColorValueTraits<T>::max)())
{
    using cv::saturate_cast;

    BOOST_STATIC_ASSERT_MSG(boost::is_floating_point<RealType>::value,
        "Color conversion parameter type is required to be floating point");

    using std::fmod;

    // H,L,S assumed to be in [0,1]
    RealType R = 0, G = 0, B = 0;

    if (L <= 0)
        // black
        R = G = B = 0;
    else if (L >= 1)
        // white
        R = G = B = 1;
    else {
        RealType hh = fmod(RealType(6) * H, RealType(6));
        T c1 = static_cast<T>(hh);
        RealType c2 = hh - c1;
        RealType d = L <= RealType(0.5) ? S * L : S * (1 - L);
        RealType w = L + d;
        RealType x = L - d;
        RealType y = w - (w - x) * c2;
        RealType z = x + (w - x) * c2;

        switch (c1) {
            case 0:
                R = w; G = z; B = x;
                break;
            case 1:
                R = y; G = w; B = x;
                break;
            case 2:
                R = x; G = w; B = z;
                break;
            case 3:
                R = x; G = y; B = w;
                break;
            case 4:
                R = z; G = x; B = w;
                break;
            case 5:
                R = w; G = x; B = y;
                break;
#ifdef _MSC_VER
            default:
                __assume(0); // cannot be reached
#endif // _MSC_VER
        }
    }

    r = saturate_cast<T>(R * high);
    g = saturate_cast<T>(G * high);
    b = saturate_cast<T>(B * high);
}

#ifdef vision_EXPORTS
VISION_EXTERN_TEMPLATE void VISION_EXPORT RGBtoHSV
	(boost::uint8_t r, boost::uint8_t g, boost::uint8_t b,
		float& h, float& s, float& v, boost::uint8_t high);
VISION_EXTERN_TEMPLATE void VISION_EXPORT RGBtoHSV
	(boost::uint8_t r, boost::uint8_t g, boost::uint8_t b,
		double& h, double& s, double& v, boost::uint8_t);
VISION_EXTERN_TEMPLATE void VISION_EXPORT RGBtoHSV
	(boost::uint8_t r, boost::uint8_t g, boost::uint8_t b,
		long double& h, long double& s, long double& v, boost::uint8_t);
#endif // defined(vision_EXPORTS)

} // namespace vision

#endif // VISION_COLORCONVERSION_HPP
