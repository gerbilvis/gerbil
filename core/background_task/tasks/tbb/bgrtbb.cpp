#include "shared_data.h"

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <stopwatch.h>

#include "multi_img/multi_img_tbb.h"
#include "multi_img/cieobserver.h"

#include "bgrtbb.h"

bool BgrTbb::run()
{
		multi_img_base& source = multi->getBase();
		cv::Mat_<cv::Vec3f> xyz(source.height, source.width, 0.);
		float greensum = 0.f;
		for (unsigned int i = 0; i < source.size(); ++i) {
			int idx = ((int)(source.meta[i].center + 0.5f) - 360) / 5;
			if (idx < 0 || idx > 94)
				continue;

			multi_img::Band band;
			source.getBand(i, band);
			Xyz computeXyz(source, xyz, band, idx);
			tbb::parallel_for(tbb::blocked_range2d<int>(0, xyz.rows, 0, xyz.cols),
				computeXyz, tbb::auto_partitioner(), stopper);

			greensum += CIEObserver::y[idx];

			if (stopper.is_group_execution_cancelled())
				break;
		}

		if (greensum == 0.f)
			greensum = 1.f;

		cv::Mat_<cv::Vec3f> *newBgr = new cv::Mat_<cv::Vec3f>(source.height, source.width);
		Bgr computeBgr(source, xyz, *newBgr, greensum);
		tbb::parallel_for(tbb::blocked_range2d<int>(0, newBgr->rows, 0, newBgr->cols),
			computeBgr, tbb::auto_partitioner(), stopper);

		if (stopper.is_group_execution_cancelled()) {
			delete newBgr;
			return false;
		} else {
			SharedDataSwapLock lock(bgr->mutex);
			bgr->replace(newBgr);
			return true;
		}
	}
