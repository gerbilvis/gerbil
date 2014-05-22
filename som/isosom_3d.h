/* This file is (only) included in isosom.h.
 * The following include statement is for the IDE */
#include "isosom.h"

/* Note: all methods are inline to avoid the code being exported by multiple
 * compile units. For most methods we wanted inline anyway, so it is o.k... */

template<>
inline int IsoSOM<3>::updateNeighborhood(size_t index, const multi_img::Pixel &input,
							   double sigma, double learnRate)
{
	if (learnRate < 0.01) // not worthy to continue
		return 0;

	/* NOTE: we square the sigma config parameter here! */

	if (!config.gaussKernel)
		return updateNeighborhoodUniform(index, input, sigma*sigma, learnRate);

	int totalUpdates = 0;
	for (int deltaZ = 0; true; ++deltaZ)
	{
		// for deltaZ == 0 this will update the middle slice, for greater
		// values it will update two slices in each call
		int updates =
			updateNeighborhoodGauss2D(index, input, sigma*sigma, learnRate, deltaZ);
		if (!updates)
			break;

		totalUpdates += updates;
	}
	return totalUpdates;
}

template<>
inline void IsoSOM<3>::coord(size_t in, int &x, int &y) const
{
	throw std::runtime_error("2d coordinate lookup not valid on 3d SOM.");
}

template<>
// convert from 1d to 3d index
inline void IsoSOM<3>::coord(size_t in, int &x, int &y, int &z) const
{
	x = in % dsize[0];
	y = (in / dsize[0]) % dsize[0];
	z = in / dsize[1];
}

// convert from 1d to 4d index (w is always 0)
template <>
inline void IsoSOM<3>::coord(size_t in, int &x, int &y, int &z, int &w) const
{
	w = 0;
	coord(in, x, y, z);
}

template <>
inline cv::Size IsoSOM<3>::size2D() const
{
	return cv::Size(dsize[1], dsize[0]);
}

template <>
inline cv::Point IsoSOM<3>::getCoord2D(size_t in) const
{
	cv::Point ret;
	int z;
	coord(in, ret.x, ret.y, z);
	// append all z dimensions to the right
	ret.x += dsize[0] * z;
	return ret;
}
