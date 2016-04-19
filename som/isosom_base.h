/* This file is (only) included in isosom.h.
 * The following include statement is for the IDE */
#include "isosom.h"

#include <algorithm>

template <size_t N>
IsoSOM<N>::IsoSOM(SOMConfig const& config, size_t nbands, bool randomize)
	: GenSOM(config)
{
	// cache sizes needed for index conversions
	dsize[0] = config.dsize;
	for (size_t i = 1; i < N; ++i)
		dsize[i] = dsize[i-1] * config.dsize;

	// initialize neurons
	init(dsize[N-1], nbands, /* randomize */ true);
}

template <size_t N>
std::vector<float> IsoSOM<N>::getCoord(size_t idx, bool normalize) const
{
	// we might write some superfluous zeros, but we don't care
	int tmp[4]; assert(N <= 4);
	coord(idx, tmp[0], tmp[1], tmp[2], tmp[3]);

	std::vector<float> ret(N); // correct data type and correct length
	for (size_t i = 0; i < N; ++i)
		ret[i] = tmp[i] * (normalize ? 1.f/(dsize[0] - 1.f) : 1.f);
	return ret;
}

template <size_t N>
int IsoSOM<N>::updateNeighborhoodGauss2D(size_t index, const multi_img::Pixel &input,
								 double sigma, double learnRate, int deltaZ)
{
	/* deltaZ tells us that instead of updating one flat 2D SOM, we update two
	 *  slices of a 3D SOM where we mirror at the Z axis. We are responsible for
	 *  the slices at Z positions +deltaZ and -deltaZ */

	// proxy variables to keep semantics in the code intact
	int width, height, depth;
	width = height = depth = dsize[0];
	int deltaZSq = deltaZ*deltaZ; // used in dot products

	// update counter
	int updates = 0;

	// Get position of index in the 3d grid (a 2d SOM will give correct 2d pos)
	cv::Point3i pos;
	coord(index, pos.x, pos.y, pos.z);

	if (!deltaZ) {
		// update at center. distance = 0, we can assume full weight
		neurons[index].update(input, learnRate);
		updates = 1;
	} else {
		// one update in each center of both slices
		for (int j = 0, dZ = deltaZ; j < 2; ++j, dZ = -dZ) {
			// <(0,0,deltaZ),(0,0,deltaZ)> == deltaZSq
			double w = gaussWeight(deltaZSq, sigma, learnRate);
			if (w < 0.01) // no more worthwile updates
				return 0;

			if (pos.z + dZ >= 0 && pos.z + dZ < depth)
			{ ++updates; n(pos.x, pos.y, pos.z + dZ).update(input, w); }
		}
	}

	int maxDist; // maximum distance of updates _along one axis_

	// update neurons on horizontal and vertical axis with equal distance
	/* (mirror over the axis that is non-zero, swap the variables and mirror
	 *  again => 2*2=4 updates)
	 */
	{
		int i;
		for (i = 1;; i++)
		{
			bool posX = pos.x + i < width;
			bool negX = pos.x - i >= 0;
			bool posY = pos.y + i < height;
			bool negY = pos.y - i >= 0;
			if ( !(posX | negX | posY | negY) ) break; // we're done already

			// <(i,0,deltaZ),(i,0,deltaZ)> == i*i + deltaZSq
			double w = gaussWeight(i*i + deltaZSq, sigma, learnRate);
			if (w < 0.01)
				break;

			// if dZ is zero, only one iteration with dZ == 0
			// else two iterations, with values dZ and -dZ
			for (int j = 0, dZ = deltaZ; j < (deltaZ ? 2 : 1); ++j, dZ = -dZ) {
				int pZ = pos.z + dZ;
				if (pZ >= 0 && pZ < depth) {
					// x axis
					if (posX)
					{ ++updates; n(pos.x + i, pos.y, pZ).update(input, w); }
					if (negX)
					{ ++updates; n(pos.x - i, pos.y, pZ).update(input, w); }
					// y axis
					if (negY)
					{ ++updates; n(pos.x, pos.y - i, pZ).update(input, w); }
					if (posY)
					{ ++updates; n(pos.x, pos.y + i, pZ).update(input, w); }
				}
			}
		}
		maxDist = i;
	}

	// update neurons on diagonal directions with equal distance
	// (mirror over all axis, swap has no effect => 2*2=4 updates per weight)
	{
		for (int i = 1; i < maxDist; i++)
		{
			bool posX = pos.x + i < width;
			bool negX = pos.x - i >= 0;
			bool posY = pos.y + i < height;
			bool negY = pos.y - i >= 0;
			if (!((posX | negX) & (posY | negY))) break; // we're done already

			// <(i,i,deltaZ),(i,i,deltaZ)> = i*i + i*i + deltaZSq = 2*i*i+deltaZSq
			double w = gaussWeight(2.*i*i + deltaZSq, sigma, learnRate);
			if (w < 0.01)
				break;

			// if dZ is zero, only one iteration with dZ == 0
			// else two iterations, with values dZ and -dZ
			for (int j = 0, dZ = deltaZ; j < (deltaZ ? 2 : 1); ++j, dZ = -dZ) {
				int pZ = pos.z + dZ;
				if (pZ >= 0 && pZ < depth) {
					if (posY) {
						if (posX) { // first quadrant
							++updates;
							n(pos.x + i, pos.y + i, pZ).update(input, w);
						}
						if (negX) { // second quadrant
							++updates;
							n(pos.x - i, pos.y + i, pZ).update(input, w);
						}
					}
					if (negY) {
						if (negX) { // third quadrant
							++updates;
							n(pos.x - i, pos.y - i, pZ).update(input, w);
						}
						if (posX) { // fourth quadrant
							++updates;
							n(pos.x + i, pos.y - i, pZ).update(input, w);
						}
					}
				}
			}
		}
	}

	// update remaining neurons
	// (mirror over all axis and swap => 2*2*2=8 updates per weight)
	// x is always greater then y. so if (y,y) was outside the L2-circle,
	// (x,y) will be outside even further
	for (int y = 1; y < maxDist; y++)
	{
		for (int x = y + 1; x < maxDist; x++)
		{
			bool posXX = pos.x + x < width;
			bool negXX = pos.x - x >= 0;
			bool posXY = pos.x + y < width;
			bool negXY = pos.x - y >= 0;
			bool posYX = pos.y + x < height;
			bool negYX = pos.y - x >= 0;
			bool posYY = pos.y + y < height;
			bool negYY = pos.y - y >= 0;

			if (!(
					// one of the first four is updated
					((posYY | negYY) & (posXX | negXX))
					// one of the second four is updated
				 || ((posYX | negYX) & (posXY | negXY))
				 ) // ergo nothing is updated
			   ) break;

			// <(x,y,deltaZ),(x,y,deltaZ)> == x*x + y*y + deltaZSq
			double w = gaussWeight(x*x + y*y + deltaZSq, sigma, learnRate);
			if (w < 0.01)
				break;

			// if dZ is zero, only one iteration with dZ == 0
			// else two iterations, with values dZ and -dZ
			for (int j = 0, dZ = deltaZ; j < (deltaZ ? 2 : 1); ++j, dZ = -dZ) {
				int pZ = pos.z + dZ;
				if (pZ >= 0 && pZ < depth) {

					if (posYY && posXX) { //  first quadrant
						++updates;
						n(pos.x + x, pos.y + y, pZ).update(input, w);
					}
					if (posYY && negXX) { // second quadrant
						++updates;
						n(pos.x - x, pos.y + y, pZ).update(input, w);
					}
					if (negYY && negXX) { //  third quadrant
						++updates;
						n(pos.x - x, pos.y - y, pZ).update(input, w);
					}
					if (negYY && posXX) { // fourth quadrant
						++updates;
						n(pos.x + x, pos.y - y, pZ).update(input, w);
					}
					// swapping x and y mirrors over diagonal of the quadrant
					if (posYX && posXY) { //  first quadrant
						++updates;
						n(pos.x + y, pos.y + x, pZ).update(input, w);
					}
					if (posYX && negXY) { // second quadrant
						++updates;
						n(pos.x - y, pos.y + x, pZ).update(input, w);
					}
					if (negYX && negXY) { //  third quadrant
						++updates;
						n(pos.x - y, pos.y - x, pZ).update(input, w);
					}
					if (negYX && posXY) { // fourth quadrant
						++updates;
						n(pos.x + y, pos.y - x, pZ).update(input, w);
					}
				}
			}
		}
	}
	return updates;
}

// for debugging
// #include <opencv2/highgui/highgui.hpp>
// #include <cstdio>

template <size_t N>
int IsoSOM<N>::updateNeighborhoodUniform(size_t index, const multi_img::Pixel &input, double sigma, double learnRate)
{
	// kernel size
	int ksize = (int)sigma;
	// upper bound
	int bound = dsize[0] - 1;

	// update counter
	int updates = 0;

	// we might write some superfluous zeros, but we don't care
	int pos[4]; assert(N <= 4);
				 //  x       y       z       w
	coord(index, pos[0], pos[1], pos[2], pos[3]);

	// for debugging
	// static int iter = 0;
	// cv::Mat1b dbg(dsize[0], dsize[0], (uchar)0);

	/* outer y, inner x (memory alignment optimization for 2D); inner z, w */
	int miny = std::max(pos[1] - ksize, 0);
	int maxy = std::min(pos[1] + ksize, bound);
	for (int y = miny; y <= maxy; ++y) {
		int delta = std::abs(pos[1] - y);
		int minx = std::max(pos[0] - (ksize - delta), 0);
		int maxx = std::min(pos[0] + (ksize - delta), bound);

		for (int x = minx; x <= maxx; ++x) {
			// dbg(y,x) = 255; // for debugging
			if (N == 2) {
				++updates;
				n(x, y).update(input, learnRate);
			}
			if (N > 2) {
				int minz = std::max(pos[2] - (ksize - delta), 0);
				int maxz = std::min(pos[2] + (ksize - delta), bound);

				for (int z = minz; z <= maxz; ++z) {
					if (N == 3) {
						++updates;
						n(x, y, z).update(input, learnRate);
					}
					if (N > 3) {
						int minw = std::max(pos[3] - (ksize - delta), 0);
						int maxw = std::min(pos[3] + (ksize - delta), bound);

						for (int w = minw; w <= maxw; ++w) {
							++updates;
							n(x, y, z, w).update(input, learnRate);
						}
					}
				}
			}
		}
	}
	// dbg(pos) = 127; // for debugging

	// for debugging
	// if (iter++ % 1000 == 0) {
	// 	char name[100];
	// 	sprintf(name, "dbg_iter_%05d.png", iter);
	// 	cv::imwrite(name, dbg);
	// }
	return updates;
}
