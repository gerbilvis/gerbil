/* This file is (only) included in isosom.h.
 * The following include statement is for the IDE */
#include "isosom.h"

/* Note: all methods are inline to avoid the code being exported by multiple
 * compile units. For most methods we wanted inline anyway, so it is o.k... */

template<>
inline int IsoSOM<4>::updateNeighborhood(size_t index,
										 const multi_img::Pixel &input,
										 double sigma, double learnRate)
{
	if (config.gaussKernel)
		throw std::runtime_error("Gauss kernel not implemented for 4D SOM!");

	/* NOTE: we take the cube of sigma config parameter here! */

	return updateNeighborhoodUniform(index, input, sigma*sigma*sigma,
									 learnRate);
}

template<>
inline void IsoSOM<4>::coord(size_t in, int &x, int &y) const
{
	throw std::runtime_error("2d coordinate lookup not valid on 4d SOM.");
}

template<>
inline void IsoSOM<4>::coord(size_t in, int &x, int &y, int &z) const
{
	throw std::runtime_error("3d coordinate lookup not valid on 4d SOM.");
}

// convert from 1d to 4d index
template <>
inline void IsoSOM<4>::coord(size_t in, int &x, int &y, int &z, int &w) const
{
	x = in % dsize[0];
	y = (in / dsize[0]) % dsize[0];
	z = (in / dsize[1]) % dsize[0];
	w = in / dsize[2];
}

template <>
inline cv::Size IsoSOM<4>::size2D() const
{
	return cv::Size(dsize[1], dsize[1]);
}

template <>
inline cv::Point IsoSOM<4>::getCoord2D(size_t in) const
{
	cv::Point ret;
	int z, w;
	coord(in, ret.x, ret.y, z, w);
	// append all z dimensions to the right
	ret.x += dsize[0] * z;
	// append all w dimensions to the bottom
	ret.y += dsize[0] * w;
	return ret;
}
