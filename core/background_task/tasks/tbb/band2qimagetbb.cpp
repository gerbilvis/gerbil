#include <background_task/background_task.h>
#include <shared_data.h>
#include <multi_img.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include "band2qimagetbb.h"

class Conversion {
public:
	Conversion(multi_img::Band &band, QImage &image,
		multi_img::Value minval, multi_img::Value maxval)
		: band(band), image(image), minval(minval), maxval(maxval) {}
	void operator()(const tbb::blocked_range2d<int> &r) const;
private:
	multi_img::Band &band;
	QImage &image;
	multi_img::Value minval;
	multi_img::Value maxval;
};

bool Band2QImageTbb::run()
{
	if (band >= (*multi)->size()) {
		return false;
	}

	multi_img::Band &source = (*multi)->bands[band];
	QImage *target = new QImage(source.cols, source.rows, QImage::Format_ARGB32);
	Conversion computeConversion(source, *target, (*multi)->minval, (*multi)->maxval);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, source.rows, 0, source.cols),
		computeConversion, tbb::auto_partitioner(), stopper);

	if (stopper.is_group_execution_cancelled()) {
		delete target;
		return false;
	} else {
		SharedDataSwapLock lock(image->mutex);
		image->replace(target);
		return true;
	}
}

void Conversion::operator()(const tbb::blocked_range2d<int> &r) const
{
	multi_img::Value scale = 255.0 / (maxval - minval);
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		const multi_img::Value *srcrow = band[y];
		QRgb *destrow = (QRgb*)image.scanLine(y);
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			unsigned int color = (srcrow[x] - minval) * scale;
			destrow[x] = qRgba(color, color, color, 255);
		}
	}
}
