#include "som_cache.h"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <algorithm>


class ClosestNTbb {
public:
	ClosestNTbb(SOMClosestN &o, multi_img const& img)
		: o(o), img(img)
	{}

	void operator()(const tbb::blocked_range2d<int> &r) const
	{
		// iterate over all pixels in range
		float done = 0;
		float total = (o.height * o.width);
		for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
			for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
				const multi_img::Pixel& pixel = img(y,x);
				const size_t roff = o.roff(x,y);
				o.som.findClosestN(pixel,
									o.results.begin() + roff,
									o.results.begin() + roff + o.n);
				done++;
				if (o.po && ((int)done % 1000 == 0)) {
					if (!o.po->update(done / total, true))
						return;
					done = 0;
				}
			}
		}
		if (o.po)
			o.po->update(done / total, true);
	}
private:
	SOMClosestN &o;
	multi_img const& img;
};


SOMClosestN::SOMClosestN(GenSOM const& som, multi_img const& img, int n,
						 vole::ProgressObserver *po)
	: som(som),
	  height(img.height),
	  width(img.width),
	  n(n > 0 ? n : throw std::runtime_error("SOMClosestN bad n")),
	  results(height * width * n),
	  po(po)
{
	tbb::parallel_for(tbb::blocked_range2d<int>(0, img.height, // row range
												0, img.width), // column range
					  ClosestNTbb(*this, img));
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
	ret.first = results.begin() + roff(p.x, p.y);
	ret.last = ret.first + n;
	assert(ret.last <= results.end());
	return ret;
}
