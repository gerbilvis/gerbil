/**
 * @file
 * @brief Utility and helper functions.
 * @author Sergiu Dotenco
 * @ingroup utility
 */

#ifndef VISION_UTILITY_HPP
#define VISION_UTILITY_HPP

#pragma once

#include <opencv2/core/core.hpp>

#if USE_TBB
#include <tbb/tbb.h>
#endif // USE_TBB

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#include <boost/math/constants/constants.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

#if USE_TBB
#include <tbb/tbb.h>
#endif // USE_TBB

#include "vision_config.hpp"
#include "vision_export.hpp"

namespace vision {

/**
 * @brief Indicates whether @a lhs is numerically close to @a rhs.
 */
template<class T>
inline bool isClose(T lhs, T rhs)
{
    using std::abs;

    return abs(lhs - rhs) <= std::numeric_limits<T>::epsilon();
}

/**
 * @brief Returns the bounds of the specified @a image.
 *
 * @param image The image whose bounding rectangle should be returned.
 *
 * @return Image bounds with the top-left coordinate located at @f$(0,0)@f$.
 *
 * @ingroup utility
 */
cv::Rect VISION_EXPORT bounds(const cv::Mat& image);

/**
 * @brief Inflates a rectangle by specified values.
 */
template<class T>
inline void inflate(cv::Rect_<T>& rect, T left, T top, T right, T bottom)
{
    rect.x -= left;
    rect.y -= top;
    rect.width += left + right;
    rect.height += top + bottom;
}

/**
 * @brief Inflates a rectangle by specified values.
 */
template<class T>
inline void inflate(cv::Rect_<T>& rect, T dx, T dy)
{
    inflate(rect, dx, dy, dx, dy);
}

/**
 * @brief Inflates a rectangle by specified values.
 */
template<class T>
inline void inflate(cv::Rect_<T>& rect, T cxy)
{
    inflate(rect, cxy, cxy);
}

/**
 * @brief Scales a rectangle by specified percentage.
 */
template<class T, class F>
inline void scale(cv::Rect_<T>& rect, F sx, F sy)
{
    assert(sx >= 0);
    assert(sy >= 0);

    T cx = T((1 - sx) * rect.width);
    T cy = T((1 - sy) * rect.height);

    inflate(rect, -cx, -cy);

    // Normalize
    if (rect.width < 0)
        rect.width = 0;

    if (rect.height < 0)
        rect.height = 0;
}

/**
 * @brief Scales a rectangle by specified percentage.
 */
template<class T, class F>
inline void scale(cv::Rect_<T>& rect, F sxy)
{
    T ratio = !isClose(rect.height, T(0)) ?
        rect.width * 100 / rect.height : T(0);

    assert(sxy >= 0);

    T cx = T((1 - sxy) * rect.width);
    T cy = T((1 - sxy) * rect.height);

    inflate(rect, -cx, -cy);

    if (!isClose(ratio, T(0)))
        rect.height = rect.width * 100 / ratio;

    // Normalize
    if (rect.width < 0)
        rect.width = 0;

    if (rect.height < 0)
        rect.height = 0;
}

//! @cond private_details

namespace detail {

BOOST_MPL_HAS_XXX_TRAIT_DEF(iterator_category)
BOOST_MPL_HAS_XXX_TRAIT_DEF(value_type)

/**
 * @brief Checks whether the specified type @a T is an iterator.
 *
 * This is a fallback metafunction which expands to @c false.
 *
 * @ingroup metafunction
 */
template<class T, class E = void>
struct is_iterator
    : boost::mpl::false_
{
};

/**
 * @brief Checks whether the specified type @a T is an iterator.
 *
 * This metafunction expands to @c true in case @a T is a pointer type.
 *
 * @ingroup metafunction
 */
template<class T>
struct is_iterator
    <
        T, typename boost::enable_if<boost::is_pointer<T> >::type
    >
    : boost::mpl::true_
{
};

/**
 * @brief Checks whether the specified type @a T is an iterator.
 *
 * This metafunction expands to @c true in case @a T contains the @c
 * iterator_category inner type.
 *
 * @ingroup metafunction
 */
template<class T>
struct is_iterator
    <
        T, typename boost::enable_if<has_iterator_category<T> >::type
    >
    : boost::mpl::true_
{
};

/**
 * @brief Extracts the value type of an iterator.
 *
 * This is an @e undefined fallback metafunction in case @c T is not an
 * iterator.
 *
 * @ingroup metafunction
 */
template<class T, class E = void>
struct iterator_value;

/**
 * @brief Extracts the value type of an iterator.
 *
 * This metafunction extracts the value type in case @c T is a pointer.
 *
 * @ingroup metafunction
 */
template<class T>
struct iterator_value
    <
        T, typename boost::enable_if<boost::is_pointer<T> >::type
    >
{
    /**
     * @brief The iterator value type.
     */
    typedef typename boost::remove_all_extents<
            typename boost::remove_cv<
                typename boost::remove_pointer<T>::type
            >::type
        >::type type;
};

/**
 * @brief Extracts the value type of an iterator.
 *
 * This metafunction extracts the value type in case @c T contains the @c
 * iterator_category @c value_type inner types.
 *
 * @ingroup metafunction
 */
template<class T>
struct iterator_value
    <
        T, typename boost::enable_if<boost::mpl::and_
        <
            has_iterator_category<T>,
            has_value_type<T>
        > >::type
    >
{
    /**
     * @brief The iterator value type.
     */
    typedef typename T::value_type type;
};

template<class T>
inline T& valueAt(cv::Mat& roi, int y, int x, int channel)
{
    return roi.channels() == 3 ?
        roi.at<cv::Vec<T, 3> >(y, x)[channel] :
        roi.channels() == 2 ?
            roi.at<cv::Vec<T, 2> >(y, x)[channel] :
            roi.at<T>(y, x);
}

} // namespace detail

//! @endcond

/**
 * @brief Rounds floating point values to nearest integer.
 *
 * The rounding is performed according to
 * @f[
 *   x'=
 *   \begin{cases}
 *     \lfloor x+0.5 \rfloor & x > 0 \\
 *     \lceil x-0.5 \rceil & x \leq 0
 *   \end{cases}
 * @f]
 * where @f$x@f$ is the value being rounded.
 *
 * @param value The value to be rounded.
 *
 * @return @a value rounded to its nearest integer.
 *
 * @ingroup utility
 */
template<class D, class T>
inline D round(T value
#ifndef DOXYGEN
    , typename boost::disable_if<boost::is_integral<D> >::type* /*dummy*/ = NULL
#endif // DOXYGEN
    )
{
    // allow ADL
    using std::floor;
    using std::ceil;

    return D(value > 0 ? floor(value + T(0.5)) : ceil(value - T(0.5)));
}

#ifndef DOXYGEN
/**
 * @briefs Rounds floating point values to nearest integer.
 *
 * @param value The value to be rounded.
 *
 * @return @a value rounded to its nearest integer.
 *
 * @ingroup utility
 */
template<class D, class T>
inline D round(T value
    , typename boost::enable_if<boost::is_integral<D> >::type* /*dummy*/ = NULL)
{
    return D(value > 0 ? value + T(0.5) : value - T(0.5));
}

#endif // DOXYGEN

//! @cond private_details

namespace detail {

template<class T, std::size_t N>
struct Pow
{
    static T pow(T value)
    {
        return value * Pow<T, N - 1>::pow(value);
    }
};

template<class T>
struct Pow<T, 0>
{
    static T pow(T)
    {
        return T(1);
    }
};

} // namespace detail

//! @endcond

/**
 * @brief Calculates the power of the specified @a value.
 *
 * @tparam N The exponent.
 *
 * @ingroup utility
 */
template<std::size_t N, class T>
inline T pow(T value)
{
    return detail::Pow<T, N>::pow(value);
}

/**
 * @brief Calculates the square of the specified @a value.
 *
 * @ingroup utility
 */
template<class T>
inline T sqr(T value)
{
    return pow<2>(value);
}

/**
 * @brief Zeros the specified matrix.
 *
 * @param mat The matrix to clear.
 *
 * @ingroup utility
 */
template<class T>
inline void clear(cv::Mat_<T>& mat)
{
    std::fill(mat.begin(), mat.end(), T(0));
}

/**
 * @brief Returns the back projection of the histogram probability.
 *
 * @ingroup utility
 */
template<class Model>
inline cv::Mat probabilityDistribution(const cv::Mat& image,
    const cv::Rect& b, const Model& model)
{
    const Model& tmp = model;
    const cv::Rect r = b & bounds(image);

    cv::Mat1b m = cv::Mat1b::zeros(image.rows, image.cols);

    for (int i = 0; i < r.height; ++i) {
        for (int j = 0; j < r.width; ++j) {
            m(i + r.y, j + r.x) = cv::saturate_cast<uchar>
                (tmp[tmp.bin(image, j + r.x, i + r.y)] * r.area());
        }
    }

    return m;
}

#if USE_TBB || defined(DOXYGEN)

//! @cond private_details

namespace detail {

template<class T, class Func, class Join>
struct Accumulate
{
    explicit Accumulate(T init, Func func, Join join)
        : init(init)
        , value(init)
        , func(func)
        , join_(join)
    {
    }

    Accumulate(const Accumulate& other, tbb::split)
        : init(other.init)
        , value(other.init)
        , func(other.func)
        , join_(other.join_)
    {
    }

    template<class Range>
    void operator()(const Range& r)
    {
        value = func(value, std::accumulate(r.begin(), r.end(), init, func));
    }

    void join(const Accumulate& other)
    {
        value = join_(value, other.value);
    }

    T init;
    T value;
    Func func;
    Join join_;
};

} // namespace detail

//! @endcond

/**
 * @brief Accumulates a value for a specific range in parallel.
 *
 * Computes the sum of all the elements in a specified range including some
 * initial value by computing successive partial sums or computes the result of
 * successive partial results similarly obtained from using a specified binary
 * operation other than the sum. The computation is performed in parallel.
 *
 * @ingroup parallel
 */
template<class T, class Iterator, class Func, class Join>
inline T parallel_accumulate(Iterator first, Iterator last, T init, Func func,
    Join join)
{
    detail::Accumulate<T, Func, Join> result(init, func, join);
    tbb::parallel_reduce(tbb::blocked_range<Iterator>(first, last), result);

    return result.value;
}

/**
 * @brief Accumulates a value for a specific range in parallel.
 *
 * Computes the sum of all the elements in a specified range including some
 * initial value by computing successive partial sums or computes the result of
 * successive partial results similarly obtained from using a specified binary
 * operation other than the sum. The computation is performed in parallel.
 *
 * @ingroup parallel
 */
template<class T, class Iterator, class Func>
inline T parallel_accumulate(Iterator first, Iterator last, T init, Func func)
{
    return parallel_accumulate(first, last, init, func, std::plus<T>());
}

/**
 * @brief Accumulates a value for a specific range in parallel.
 *
 * Computes the sum of all the elements in a specified range including some
 * initial value by computing successive partial sums or computes the result of
 * successive partial results similarly obtained from using a specified binary
 * operation other than the sum. The computation is performed in parallel.
 *
 * @ingroup parallel
 */
template<class T, class Iterator>
inline T parallel_accumulate(Iterator first, Iterator last, T init)
{
    return parallel_accumulate(first, last, init, std::plus<T>());
}

#endif // USE_TBB

#ifndef DOXYGEN
#ifndef vision_EXPORTS
VISION_EXTERN_TEMPLATE void VISION_EXPORT inflate<float>(cv::Rect_<float>& rect,
    float left, float top, float right, float bottom);
VISION_EXTERN_TEMPLATE void VISION_EXPORT inflate<double>(cv::Rect_<double>& rect,
    double left, double top, double right, double bottom);
VISION_EXTERN_TEMPLATE void VISION_EXPORT inflate<long double>(cv::Rect_<long double>& rect,
    long double left, long double top, long double right, long double bottom);

VISION_EXTERN_TEMPLATE void VISION_EXPORT inflate<float>(cv::Rect_<float>& rect, float cxy);
VISION_EXTERN_TEMPLATE void VISION_EXPORT inflate<double>(cv::Rect_<double>& rect, double cxy);
VISION_EXTERN_TEMPLATE void VISION_EXPORT inflate<long double>(cv::Rect_<long double>& rect,
    long double cxy);
#endif // !defined(vision_EXPORTS)
#endif // !defined(DOXYGEN)

} // namespace vision

#endif // VISION_UTILITY_HPP
