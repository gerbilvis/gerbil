#include "som_cache.h"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <algorithm>

namespace som {

SOMClosestN::SOMClosestN(GenSOM const& som, multi_img const& img, int n,
						 ProgressObserver *po)
	: som(som),
	  height(img.height),
	  width(img.width),
	  n(n > 0 ? n : throw std::runtime_error("SOMClosestN bad n")),
	  results(height * width * n),
	  po(po)
{
	tbb::parallel_for(tbb::blocked_range2d<int>(0, height, // row range
	                                            0, width), // column range
	                  [&](const tbb::blocked_range2d<int> &r) {
		float done = 0;
		float total = (height * width);
		for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
			for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
				const multi_img::Pixel& pixel = img(y, x);
				const size_t offset = roff(y, x);
				som.findClosestN(pixel,
				                 results.begin() + offset,
				                 results.begin() + offset + n);
				done++;
				if (po && ((int)done % 1000 == 0)) {
					if (!po->update(done / total, true))
						return;
					done = 0;
				}
			}
		}
		if (po)
			po->update(done / total, true);
	});
}

std::vector<DistIndexPair> SOMClosestN::closestNCopy(const cv::Point2i &p) const
{
	// copy from internal storage to external vector that we return
	resultAccess answer = closestN(p);
	return std::vector<DistIndexPair>(answer.first, answer.last);
}

SOMClosestN::resultAccess SOMClosestN::closestN(const cv::Point2i &p) const
{
	resultAccess ret;
	ret.first = results.begin() + roff(p.y, p.x);
	ret.last = ret.first + n;
	assert(ret.last <= results.end());
	return ret;
}

}
