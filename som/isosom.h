#ifndef ISOSOM_H
#define ISOSOM_H

#include "gensom.h"

namespace som {

template <size_t N>
class IsoSOM : public GenSOM
{
public:
	IsoSOM(const SOMConfig &config, size_t nbands, bool randomize);

	int updateNeighborhood(size_t index,
						   const multi_img::Pixel &input,
						   double sigma, double learnRate);

	std::vector<float> getCoord(size_t idx, bool normalize = true) const;
	cv::Size size2D() const;
	cv::Point getCoord2D(size_t idx) const;

protected:
	// helper called by updateNeighborhood for 2D, part of 3D case
	int updateNeighborhoodGauss2D(size_t index,
						   const multi_img::Pixel &input,
						   double sigma, double learnRate, int deltaZ);

	// helper called by updateNeighborhood for all cases
	int updateNeighborhoodUniform(size_t index,
						   const multi_img::Pixel &input,
						   double sigma, double learnRate);

	// convert from 1d to Nd index
	inline void coord(size_t in, int &x, int &y) const;
	inline void coord(size_t in, int &x, int &y, int &z) const;
	inline void coord(size_t in, int &x, int &y, int &z, int &w) const;

	// convert from Nd to 1d index, N < 5
	inline size_t idx(int x, int y = 0, int z = 0, int w = 0) const {
		size_t i = x + (y * dsize[0]);
		if (N>2)
			i += z * dsize[1];
		if (N>3)
			i += w * dsize[2];
		return i;
	}

	// return neuron at Nd index, N < 5
	inline Neuron & n(int x, int y = 0, int z = 0, int w = 0) {
		return neurons[idx(x, y, z, w)];
	}

	// recursive size of each dim.; ie dsize[N-1] is the total amount of neurons
	size_t dsize[(N == 0 ? 1 : N)];
};

#include "isosom_base.h"
#include "isosom_2d.h"
#include "isosom_3d.h"
#include "isosom_4d.h"

}
#endif // ISOSOM_H
