/**
 * @file
 * @brief Horn-Schunck dense optical flow implementation.
 * @author Sergiu Dotenco
 * @ingroup opticalflow
 */

#ifndef VISION_HORNSCHUNCKOPTICALFLOW_HPP
#define VISION_HORNSCHUNCKOPTICALFLOW_HPP

#pragma once

#include <opencv2/core/core.hpp>

#if USE_TBB
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#endif // USE_TBB

#include <boost/throw_exception.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "vision_config.hpp"
#include "vision_export.hpp"

namespace vision {

/**
 * @brief Horn-Schunck dense optical flow.
 *
 * @param RealType The floating point type.
 *
 * @ingroup opticalflow
 */
template<class RealType>
class HornSchunckOpticalFlow
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
    HornSchunckOpticalFlow()
        : alpha_(RealType(10))
        , iterations_(100)
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
    void compute(const cv::Mat1b& image1, cv::Mat1b& image2,
        cv::Mat_<RealType>& u, cv::Mat_<RealType>& v, unsigned flags = none)
    {
        if (image1.size != image2.size)
            BOOST_THROW_EXCEPTION(std::invalid_argument
                ("mismatch between the size of the previous and the next image"));

        if (flags & reuseVelocities) {
            if (image1.size != u.size)
                BOOST_THROW_EXCEPTION(std::invalid_argument
                    ("u velocity matrix size mismatch"));

            if (image1.size != v.size)
                BOOST_THROW_EXCEPTION(std::invalid_argument
                    ("v velocity matrix size mismatch"));
        }

        int cols = image1.cols;
        int rows = image2.rows;

        if (!(flags & reuseVelocities)) {
            u = cv::Mat_<RealType>::zeros(rows, cols);
            v = cv::Mat_<RealType>::zeros(rows, cols);
        }

        cv::Mat_<RealType> Ex(rows, cols);
        cv::Mat_<RealType> Ey(rows, cols);
        cv::Mat_<RealType> Et(rows, cols);

        cv::Mat_<RealType> u_new(rows, cols);
        cv::Mat_<RealType> v_new(rows, cols);

        ComputeDerivatives derivativesBody(image1, image2, Ex, Ey, Et);

#if USE_TBB
        tbb::parallel_for(tbb::blocked_range2d<int>(0, rows, 0, cols),
            derivativesBody);
#else // !defined(USE_TBB)
        // compute derivatives
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                derivativesBody(i, j);
            }
        }
#endif // !defined(USE_TBB)

        const RealType alpha2 = alpha_ * alpha_;

        ComputeVelocities velocitiesBody(Ex, Ey, Et, u, v, u_new, v_new, alpha2,
            iterations_);
        bool stop = false;

#if USE_TBB
        for (unsigned l = 0; l < iterations_ && !stop; ++l) {
            tbb::parallel_for(tbb::blocked_range2d<int>(0, rows, 0, cols),
                velocitiesBody);

            swap(velocitiesBody.u_new, velocitiesBody.u);
            swap(velocitiesBody.v_new, velocitiesBody.v);
        }
#else // !defined(USE_TBB)
        for (unsigned l = 0; l < iterations_; ++l) {
            for (int j = 0; j < cols; ++j) {
                for (int i = 0; i < rows; ++i) {
                    velocitiesBody(i, j);
                }
            }

            swap(velocitiesBody.u_new, velocitiesBody.u);
            swap(velocitiesBody.v_new, velocitiesBody.v);
        }
#endif // !defined(USE_TBB)
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
                image1(i, j) + image1(i_plus_1, j_plus_1) -
                image1(i_plus_1, j) + image2(i, j_plus_1) - image2(i, j) +
                image2(i_plus_1, j_plus_1) - image2(i_plus_1, j));

            // [  1  1 ]
            // [ -1 -1 ]
            Ey(i, j) = RealType(0.25) * (RealType(image1(i_plus_1, j)) -
                image1(i, j) + image1(i_plus_1, j_plus_1) -
                image1(i, j_plus_1) + image2(i_plus_1, j) - image2(i, j) +
                image2(i_plus_1, j_plus_1) - image2(i, j_plus_1));

            Et(i, j) = RealType(0.25) * (RealType(image2(i, j)) -
                image1(i, j) + image2(i_plus_1, j) - image1(i_plus_1, j) +
                image2(i, j_plus_1) - image1(i, j_plus_1) + image2(i_plus_1,
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
            cv::Mat_<RealType>& v_new, RealType alpha2, unsigned iterations)
            : Ex(Ex)
            , Ey(Ey)
            , Et(Et)
            , u(u)
            , v(v)
            , u_new(u_new)
            , v_new(v_new)
            , alpha2(alpha2)
            , iterations(iterations)
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
            int cols = Ex.cols;
            int rows = Ex.rows;

            // edge clamping
            int i_plus_1 = i < rows - 1 ? i + 1 : i;
            int j_plus_1 = j < cols - 1 ? j + 1 : j;
            int i_minus_1 = i > 0 ? i - 1 : i;
            int j_minus_1 = j > 0 ? j - 1 : j;

            RealType u_avg = RealType(1) / 6 * (u(i_minus_1, j) +
                u(i, j_plus_1) + u(i_plus_1, j) + u(i, j_minus_1)) +
                RealType(1) / 12 * (u(i_minus_1, j_minus_1) +
                u(i_minus_1, j_plus_1) + u(i_plus_1, j_plus_1) +
                u(i_plus_1, j_minus_1));

            RealType v_avg = RealType(1) / 6 * (v(i_minus_1, j) +
                v(i, j_plus_1) + v(i_plus_1, j) + v(i, j_minus_1)) +
                RealType(1) / 12 * (v(i_minus_1, j_minus_1) +
                v(i_minus_1, j_plus_1) + v(i_plus_1, j_plus_1) +
                v(i_plus_1, j_minus_1));

            RealType factor = Ex(i, j) * u_avg + Ey(i, j) * v_avg +
                Et(i, j);
            RealType d = alpha2 + Ex(i, j) * Ex(i, j) + Ey(i, j) * Ey(i, j);

            u_new(i, j) = u_avg - Ex(i, j) * factor / d;
            v_new(i, j) = v_avg - Ey(i, j) * factor / d;
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
    };

    //! @endcond

    RealType alpha_;
    unsigned iterations_;
};

#ifndef DOXYGEN
#ifndef vision_EXPORTS
VISION_EXTERN_TEMPLATE class VISION_EXPORT HornSchunckOpticalFlow<float>;
VISION_EXTERN_TEMPLATE class VISION_EXPORT HornSchunckOpticalFlow<double>;
#endif // !defined(vision_EXPORTS)
#endif // !defined(DOXYGEN)

} // namespace vision

#endif // VISION_HORNSCHUNCKOPTICALFLOW_HPP
