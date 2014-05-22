/* This file is (only) included in isosom.h.
 * The following include statement is for the IDE */
#include "isosom.h"

/* Note: all methods are inline to avoid the code being exported by multiple
 * compile units. For most methods we wanted inline anyway, so it is o.k... */

template<>
inline int IsoSOM<2>::updateNeighborhood(size_t index,
										 const multi_img::Pixel &input,
										 double sigma, double learnRate)
{
	if (learnRate < 0.01) // not worthy to continue
		return 0;

	if (config.gaussKernel)
		return updateNeighborhoodGauss2D(index, input, sigma, learnRate, 0);
	else
		return updateNeighborhoodUniform(index, input, sigma, learnRate);
}

// convert from 1d to 2d index
template <>
inline void IsoSOM<2>::coord(size_t in, int &x, int &y) const
{
	x = in % dsize[0]; // column
	y = in / dsize[0]; // row
}

// convert from 1d to 3d index (z is always 0)
template <>
inline void IsoSOM<2>::coord(size_t in, int &x, int &y, int &z) const
{
	z = 0;
	coord(in, x, y);
}

// convert from 1d to 4d index (z, w are always 0)
template <>
inline void IsoSOM<2>::coord(size_t in, int &x, int &y, int &z, int &w) const
{
	w = z = 0;
	coord(in, x, y);
}

template <>
inline cv::Size IsoSOM<2>::size2D() const
{
	return cv::Size(dsize[0], dsize[0]);
}

template <>
inline cv::Point IsoSOM<2>::getCoord2D(size_t in) const
{
	cv::Point ret;
	coord(in, ret.x, ret.y);
	return ret;
}
