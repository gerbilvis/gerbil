/**
 * @file
 * @brief Pyramidal Horn-Schunck dense optical flow implementation.
 * @author Sergiu Dotenco
 * @ingroup opticalflow
 */

#ifndef VISION_PYRAMIDALHORNSCHUNCKOPTICALFLOW_HPP
#define VISION_PYRAMIDALHORNSCHUNCKOPTICALFLOW_HPP

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if USE_TBB
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#endif // USE_TBB

#include <boost/next_prior.hpp>
#include <boost/throw_exception.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utility.hpp"
#include "vision_config.hpp"
#include "vision_export.hpp"

namespace vision {

/**
 * @brief Pyramidal Horn-Schunck dense optical flow.
 *
 * @param RealType The floating point type.
 *
 * @ingroup opticalflow
 */
template<class RealType>
class PyramidalHornSchunckOpticalFlow
{
public:
    //! The floating point type.
    typedef RealType real_type;

    //! Optical flow computation flags.
    enum ComputeFlags
    {
        //! None.
        none,
        //! Reuse the velocities passed to @ref compute.
        reuseVelocities = 1
    };

    /**
     * @brief Initializes the class.
     */
    PyramidalHornSchunckOpticalFlow()
        : alpha_(RealType(10))
        , iterations_(100)
        , levels_(5)
        , scaleFactor_(RealType(2))
    {
    }

    /**
     * @brief Computes the dense optical flow field between @a image1 and @a
     *        image2.
     *
     * @param image1 Previous image.
     * @param image2 Next image.
     * @param[in,out] u The horizontal flow.
     * @param[in,out] v The vertical flow.
     * @param flags A combination of @ref ComputeFlags values.
     *
     * @exception std::invalid_argument Thrown, if the number of rows or columns
     *            used by @c image1, @c image2, @c u and @c v do not match. The
     *            latter two matrices are checked only, if @ref reuseVelocities
     *            has been specified.
     */
    void compute(const cv::Mat1b& image1, cv::Mat1b& image2, cv::Mat_<RealType>& u,
        cv::Mat_<RealType>& v, unsigned flags = none)
    {
        using std::pow;
        using std::ceil;
        using std::log;

        int cols = image1.cols;
        int rows = image2.rows;

        if (image1.size != image2.size)
            BOOST_THROW_EXCEPTION(std::invalid_argument
                ("mismatch between the size of the previous and the next image"));

        if (flags & reuseVelocities) {
            if (u.empty())
                u = cv::Mat_<RealType>::zeros(rows, cols);
            else if (image1.size != u.size)
                BOOST_THROW_EXCEPTION(std::invalid_argument
                    ("u velocity matrix size mismatch"));

            if (v.empty())
                v = cv::Mat_<RealType>::zeros(rows, cols);
            if (image1.size != v.size)
                BOOST_THROW_EXCEPTION(std::invalid_argument
                    ("v velocity matrix size mismatch"));
        }

        if (!(flags & reuseVelocities)) {
            u = cv::Mat_<RealType>::zeros(rows, cols);
            v = cv::Mat_<RealType>::zeros(rows, cols);
        }

        // Build a coarse-to-fine pyramid
        typedef std::vector<std::pair<cv::Mat1b, cv::Mat1b> > PyramidVector;

        const int size = (std::min)(cols, rows);
        // Compute the maximum number of pyramid levels
        const std::size_t maxLevels = round<std::size_t>
            (log(size / RealType(10)) / log(scaleFactor_));

        assert(maxLevels > 0);
        const std::size_t levels = (std::min)(maxLevels, levels_);

        PyramidVector pyramid;
        pyramid.reserve(levels);

        pyramid.push_back(std::make_pair(image1, image2));

        for (std::size_t level = 1; level != levels; ++level) {
            cv::Mat1b tmp1;
            cv::Mat1b tmp2;

            // Downsample the images
            RealType scale = pow(scaleFactor_, RealType(level));
            int cols = round<int>(RealType(image1.cols) / scale);
            int rows = round<int>(RealType(image1.rows) / scale);

            cv::resize(image1, tmp1, cv::Size(cols, rows));
            cv::resize(image2, tmp2, cv::Size(cols, rows));

            pyramid.insert(pyramid.begin(), std::make_pair(tmp1, tmp2));
        }

        if (flags & reuseVelocities) {
            cv::Mat_<RealType> u_tmp = u.clone();
            cv::Mat_<RealType> v_tmp = v.clone();

            const cv::Size size(pyramid.front().first.cols,
                pyramid.front().first.rows);

            cv::resize(u_tmp, u, size);
            cv::resize(v_tmp, v, size);
        }

        for (PyramidVector::iterator it = pyramid.begin();
            it != pyramid.end(); ++it) {
            std::size_t level = pyramid.size() - static_cast<std::size_t>
                (std::distance(pyramid.begin(), it)) - 1;
            RealType currentFactor = pow(scaleFactor_, RealType(level));

            const cv::Size currentSize(it->first.cols, it->first.rows);

            cv::Mat_<RealType> u_new = cv::Mat_<RealType>::zeros(currentSize);
            cv::Mat_<RealType> v_new = cv::Mat_<RealType>::zeros(currentSize);

            processLevel(it->first, it->second, it->first.rows, it->first.cols,
                currentFactor, u, v, u_new, v_new);

            if (level > 0) {
                PyramidVector::iterator next = boost::next(it);
                const cv::Size nextSize(next->first.cols, next->first.rows);

                // Prolongate flow vectors to the next pyramid level
                cv::resize(u_new, u, nextSize);
                cv::resize(v_new, v, nextSize);

                // Scale the flow vectors to compensate previous downsampling
                u.convertTo(u, u.type(), scaleFactor_);
                v.convertTo(v, v.type(), scaleFactor_);
            }
        }
    }

    //! @name Properties
    //! @{

    /**
     * @brief Gets a value indicating the weight of the regularization term.
     */
    RealType alpha() const
    {
        return alpha_;
    }

    /**
     * @brief Sets a value indicating the weight of the regularization term. The
     *        bigger the term is, the smoother the velocity field becomes.
     */
    void setAlpha(RealType value)
    {
        alpha_ = value;
    }

    /**
     * @brief Gets the maximum number of iterations that will be executed while
     *        computing the optical flow.
     */
    unsigned iterations() const
    {
        return iterations_;
    }

    /**
     * @brief Sets the maximum number of iterations to be executed while
     *        computing the optical flow.
     *
     * @param value The maximum number of iterations.
     */
    void setIterations(unsigned value)
    {
        iterations_ = value;
    }

    /**
     * @brief Gets a value indicating the number of pyramid levels.
     */
    std::size_t levels() const
    {
        return levels_;
    }

    /**
     * @brief Sets the maximum number of pyramid levels.
     *
     * The actual number of pyramid levels @f$\ell@f$ is determined according to
     * @f[
     * \ell = \log_2 \frac{\min(M, N)}{k}
     * @f]
     * @f$M,N@f$ being the image height and width, respectively. @f$k=10@f$ is
     * the image width or height on the coarsest pyramid level.
     */
    void setLevels(std::size_t value)
    {
        levels_ = value;
    }

    /**
     * @brief Gets a value indicating the scale factor for each pyramid level.
     */
    RealType scaleFactor() const
    {
        return scaleFactor_;
    }

    /**
     * @brief Sets the scale factor for each pyramid level.
     */
    void setScaleFactor(RealType value)
    {
        scaleFactor_ = value;
    }

    //! @}

private:
    //! @cond private_details

    struct ComputeDerivatives
    {
        ComputeDerivatives(const cv::Mat1b& image1, const cv::Mat1b& image2,
            cv::Mat_<RealType>& Ex, cv::Mat_<RealType>& Ey,
            cv::Mat_<RealType>& Et)
            : image1(image1)
            , image2(image2)
            , Ex(Ex)
            , Ey(Ey)
            , Et(Et)
        {
        }

#if USE_TBB
        void operator()(const tbb::blocked_range2d<int>& range) const
        {
            for (int y = range.rows().begin(); y != range.rows().end(); ++y) {
                for (int x = range.cols().begin(); x != range.cols().end(); ++x) {
                    (*this)(y, x);
                }
            }
        }
#endif // USE_TBB

        void operator()(int i, int j) const
        {
            int cols = Ex.cols;
            int rows = Ex.rows;

            // edge clamping
            int i_plus_1 = i < rows - 1 ? i + 1 : i;
            int j_plus_1 = j < cols - 1 ? j + 1 : j;

            // [ -1 1 ]
            // [ -1 1 ]
            Ex(i, j) = RealType(0.25) * (RealType(image1(i, j_plus_1)) -
                image1(i, j) + image1(i_plus_1, j_plus_1) - image1(i_plus_1, j) +
                image2(i, j_plus_1) - image2(i, j) + image2(i_plus_1,
                j_plus_1) - image2(i_plus_1, j));

            // [  1  1 ]
            // [ -1 -1 ]
            Ey(i, j) = RealType(0.25) * (RealType(image1(i_plus_1, j)) -
                image1(i, j) + image1(i_plus_1, j_plus_1) - image1(i, j_plus_1) +
                image2(i_plus_1, j) - image2(i, j) + image2(i_plus_1,
                j_plus_1) - image2(i, j_plus_1));

            Et(i, j) = RealType(0.25) * (RealType(image2(i, j)) - image1(i, j) +
                image2(i_plus_1, j) - image1(i_plus_1, j) + image2(i,
                j_plus_1) - image1(i, j_plus_1) + image2(i_plus_1,
                j_plus_1) - image1(i_plus_1, j_plus_1));
        }

        const cv::Mat1b& image1;
        const cv::Mat1b& image2;
        cv::Mat_<RealType>& Ex;
        cv::Mat_<RealType>& Ey;
        cv::Mat_<RealType>& Et;
    };

    struct ComputeVelocities
    {
        ComputeVelocities(cv::Mat_<RealType>& Ex, cv::Mat_<RealType>& Ey,
            cv::Mat_<RealType>& Et, cv::Mat_<RealType>& u,
            cv::Mat_<RealType>& v, cv::Mat_<RealType>& u_new,
            cv::Mat_<RealType>& v_new, RealType alpha2, unsigned iterations,
            RealType factor)
            : Ex(Ex)
            , Ey(Ey)
            , Et(Et)
            , u(u)
            , v(v)
            , u_new(u_new)
            , v_new(v_new)
            , alpha2(alpha2)
            , iterations(iterations)
            , factor(factor)
        {
        }

#if USE_TBB
        void operator()(const tbb::blocked_range2d<int>& range) const
        {
            for (int y = range.rows().begin(); y != range.rows().end(); ++y) {
                for (int x = range.cols().begin(); x != range.cols().end(); ++x) {
                    (*this)(y, x);
                }
            }
        }
#endif // defined(USE_TBB)

        void operator()(int i, int j) const
        {
            int cols = u.cols;
            int rows = u.rows;

            int i_dst = i;
            int j_dst = j;

            // edge clamping
            int i_plus_1 = i < rows - 1 ? i + 1 : i;
            int j_plus_1 = j_dst < cols - 1 ? j_dst + 1 : j_dst;
            int i_minus_1 = i > 0 ? i - 1 : i;
            int j_minus_1 = j_dst > 0 ? j_dst - 1 : j_dst;

            RealType u_avg = RealType(1) / 6 * (u(i_minus_1, j_dst) +
                u(i_dst, j_plus_1) + u(i_plus_1, j_dst) +
                u(i_dst, j_minus_1)) + RealType(1) / 12 *
                (u(i_minus_1, j_minus_1) + u(i_minus_1, j_plus_1) +
                u(i_plus_1, j_plus_1) + u(i_plus_1, j_minus_1));

            RealType v_avg = RealType(1) / 6 * (v(i_minus_1, j_dst) +
                v(i_dst, j_plus_1) + v(i_plus_1, j_dst) +
                v(i_dst, j_minus_1)) + RealType(1) / 12 *
                (v(i_minus_1, j_minus_1) + v(i_minus_1, j_plus_1) +
                v(i_plus_1, j_plus_1) + v(i_plus_1, j_minus_1));

            RealType factor = Ex(i, j) * u_avg + Ey(i, j) * v_avg +
                Et(i, j);
            RealType d = alpha2 + Ex(i, j) * Ex(i, j) + Ey(i, j) * Ey(i, j);

            u_new(i_dst, j_dst) = u_avg - Ex(i, j) * factor / d;
            v_new(i_dst, j_dst) = v_avg - Ey(i, j) * factor / d;
        }

        cv::Mat_<RealType>& Ex;
        cv::Mat_<RealType>& Ey;
        cv::Mat_<RealType>& Et;
        cv::Mat_<RealType>& u;
        cv::Mat_<RealType>& v;
        cv::Mat_<RealType>& u_new;
        cv::Mat_<RealType>& v_new;
        RealType alpha2;
        unsigned iterations;
        RealType factor;
    };

    //! @endcond

    void processLevel(const cv::Mat1b& image1, const cv::Mat1b& image2,
        int rows, int cols, RealType factor, cv::Mat_<RealType>& u,
        cv::Mat_<RealType>& v,
        cv::Mat_<RealType>& u_new, cv::Mat_<RealType>& v_new)
    {
        cv::Mat_<RealType> Ex(rows, cols);
        cv::Mat_<RealType> Ey(rows, cols);
        cv::Mat_<RealType> Et(rows, cols);

        ComputeDerivatives derivativesBody(image1, image2, Ex, Ey, Et);

#if USE_TBB
        tbb::parallel_for(tbb::blocked_range2d<int>(0, rows, 0, cols),
            derivativesBody);
#else // !USE_TBB
        // compute derivatives
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                derivativesBody(i, j);
            }
        }
#endif // USE_TBB

        const RealType alpha2 = alpha_ * alpha_;

        ComputeVelocities velocitiesBody(Ex, Ey, Et, u, v, u_new, v_new, alpha2,
            iterations_, factor);
        bool stop = false;

#if USE_TBB
        for (unsigned l = 0; l < iterations_ && !stop; ++l) {
            tbb::parallel_for(tbb::blocked_range2d<int>(0, rows, 0, cols),
                velocitiesBody);

            swap(velocitiesBody.u_new, velocitiesBody.u);
            swap(velocitiesBody.v_new, velocitiesBody.v);
        }
#else // !USE_TBB
        for (unsigned l = 0; l < iterations_; ++l) {
            for (int j = 0; j < cols; ++j) {
                for (int i = 0; i < rows; ++i) {
                    velocitiesBody(i, j);
                }
            }

            swap(velocitiesBody.u_new, velocitiesBody.u);
            swap(velocitiesBody.v_new, velocitiesBody.v);
        }
#endif // USE_TBB
    }

    RealType alpha_;
    unsigned iterations_;
    std::size_t levels_;
    RealType scaleFactor_;
};

#ifndef DOXYGEN
#ifndef vision_EXPORTS
VISION_EXTERN_TEMPLATE class VISION_EXPORT PyramidalHornSchunckOpticalFlow<float>;
VISION_EXTERN_TEMPLATE class VISION_EXPORT PyramidalHornSchunckOpticalFlow<double>;
#endif // !defined(vision_EXPORTS)
#endif // !defined(DOXYGEN)

} // namespace vision

#endif // VISION_PYRAMIDALHORNSCHUNCKOPTICALFLOW_HPP
