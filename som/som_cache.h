#ifndef SOM_CACHE_H
#define SOM_CACHE_H

#include "gensom.h"
#include <progress_observer.h>

namespace som {

/** Compute closest n neurons in SOM for each multi_img pixel.
 *
 * The results are computed on construction.
*/
class SOMClosestN
{
public:

	/** Compute closest n for all pixels. */
	SOMClosestN(GenSOM const& som,
				multi_img const& img,
				int n,
				ProgressObserver *po = 0);

	/** Copy closest n result for pixel with coordinates p.
	 *
	 * Precondition: result.size() == n
	 * The result vector is sorted by ascending distance.
	 * Note this implementation creates a copy of the result vector. A more
	 * efficient read-only implementation is provided below.
	 */
	std::vector<DistIndexPair> closestNCopy(const cv::Point2i& p) const;

	// Iterator access into internal result storage
	struct resultAccess {
		std::vector<DistIndexPair>::const_iterator first;
		std::vector<DistIndexPair>::const_iterator last;
	};

	/** Access closest n result for pixel with coordinates p.
	 *
	 * Returns first and last iterators that point into internal result storage
	 * (std::vector).
	 * last - first = n.
	 */
	resultAccess closestN(const cv::Point2i& p) const;

	/* const variables that can be useful to query, don't hurt being public */
	GenSOM const& som;
	const int height; /// multi_img height
	const int width;  /// multi_img width
	const int n;      /// closest n

protected:
	// Compute offset into results from pixel coordinates.
	inline size_t roff(int y, int x) const {
		assert(0 <= x && x < width);
		assert(0 <= y && y < height);
		size_t off = ((y * width) + x) * n;
		assert(off < results.size());
		return off;
	}

	// neuron distances and SOM indices
	// size = width * height * n
	std::vector<DistIndexPair> results;
	ProgressObserver *po;
	friend class ClosestNTbb;
};

}
#endif // SOM_CACHE_H
