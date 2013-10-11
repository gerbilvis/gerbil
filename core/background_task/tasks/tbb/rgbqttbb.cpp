#include <shared_data.h>

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <stopwatch.h>


#include "rgbqttbb.h"

class Rgb {
public:
	Rgb(cv::Mat_<cv::Vec3f> &bgr, QImage &rgb)
		: bgr(bgr), rgb(rgb) {}
	void operator()(const tbb::blocked_range2d<int> &r) const
	{
		for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
			cv::Vec3f *row = bgr[y];
			QRgb *destrow = (QRgb*)rgb.scanLine(y);
			for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
				cv::Vec3f &c = row[x];
				destrow[x] = qRgb(c[2]*255., c[1]*255., c[0]*255.);
			}
		}
	}

private:
	cv::Mat_<cv::Vec3f> &bgr;
	QImage &rgb;
};

bool RgbTbb::run()
{
	if (!BgrTbb::run())
		return false;
	if (stopper.is_group_execution_cancelled())
		return false;

	cv::Mat3f &bgrmat = *(*bgr);
	QImage *newRgb = new QImage(bgrmat.cols, bgrmat.rows, QImage::Format_ARGB32);

	Rgb computeRgb(bgrmat, *newRgb);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, bgrmat.rows, 0, bgrmat.cols),
		computeRgb, tbb::auto_partitioner(), stopper);

	if (stopper.is_group_execution_cancelled()) {
		delete newRgb;
		return false;
	} else {
		SharedDataSwapLock lock(rgb->mutex);
		rgb->replace(newRgb);
		return true;
	}
}
