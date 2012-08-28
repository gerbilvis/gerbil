/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "multi_img_tasks.h"

namespace CIEObserver {
	extern float x[];
	extern float y[];
	extern float z[];
}

namespace MultiImg {

void BgrSerial::run() 
{
	SharedDataRead rlock(multi->lock);
	cv::Mat_<cv::Vec3f> *newBgr = new cv::Mat_<cv::Vec3f>();
	*newBgr = (*multi)->bgr(); 
	SharedDataWrite wlock(bgr->lock);
	delete bgr->swap(newBgr);
}

void BgrTbb::run() 
{
	SharedDataRead rlock(multi->lock);

	cv::Mat_<cv::Vec3f> xyz((*multi)->height, (*multi)->width, 0.);
	float greensum = 0.f;
	for (size_t i = 0; i < (*multi)->size(); ++i) {
		int idx = ((int)((*multi)->meta[i].center + 0.5f) - 360) / 5;
		if (idx < 0 || idx > 94)
			continue;

		Xyz computeXyz(multi, xyz, i, idx);
		tbb::parallel_for(tbb::blocked_range2d<int>(0, xyz.rows, 0, xyz.cols), 
			computeXyz, tbb::auto_partitioner(), stopper);

		greensum += CIEObserver::y[idx];
	}

	if (greensum == 0.f)
		greensum = 1.f;

	cv::Mat_<cv::Vec3f> *newBgr = new cv::Mat_<cv::Vec3f>((*multi)->height, (*multi)->width);
	Bgr computeBgr(multi, xyz, *newBgr,  greensum);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, newBgr->rows, 0, newBgr->cols), 
		computeBgr, tbb::auto_partitioner(), stopper);

	if (stopper.is_group_execution_cancelled()) {
		stopper.reset();
		delete newBgr;
		return;
	} else {
		SharedDataWrite wlock(bgr->lock);
		delete bgr->swap(newBgr);
	}
}

void BgrTbb::Xyz::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
		for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
			cv::Vec3f &v = xyz(i, j);
			float intensity = (*multi)->bands[band](i, j) * 1.f/(*multi)->maxval;
			v[0] += CIEObserver::x[cie] * intensity;
			v[1] += CIEObserver::y[cie] * intensity;
			v[2] += CIEObserver::z[cie] * intensity;
		}
	}
}

void BgrTbb::Bgr::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
		for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
			cv::Vec3f &vs = xyz(i, j);
			cv::Vec3f &vd = bgr(i, j);
			for (unsigned int j = 0; j < 3; ++j)
				vs[j] /= greensum;
			(*multi)->xyz2bgr(vs, vd);
		}
	}
}

}
