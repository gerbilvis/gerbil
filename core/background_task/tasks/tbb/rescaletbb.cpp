#include <shared_data.h>
#include <background_task/background_task.h>
#include <shared_data.h>
#include <multi_img.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <multi_img/multi_img_tbb.h>

#include "rescaletbb.h"


bool RescaleTbb::run()
{
	multi_img *temp = new multi_img(**source,
		cv::Rect(0, 0, (*source)->width, (*source)->height));
	temp->roi = (*source)->roi;
	RebuildPixels rebuildPixels(*temp);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, temp->size()),
		rebuildPixels, tbb::auto_partitioner(), stopper);
	temp->dirty.setTo(0);
	temp->anydirt = false;

	multi_img *target = NULL;
	if (newsize != temp->size()) {

		target = new multi_img((*source)->height, (*source)->width, newsize);
		target->minval = temp->minval;
		target->maxval = temp->maxval;
		target->roi = temp->roi;
		Resize computeResize(*temp, *target, newsize);
		tbb::parallel_for(tbb::blocked_range2d<int>(0, temp->height, 0, temp->width),
			computeResize, tbb::auto_partitioner(), stopper);

		ApplyCache applyCache(*target);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()),
			applyCache, tbb::auto_partitioner(), stopper);
		target->dirty.setTo(0);
		target->anydirt = false;

		if (!stopper.is_group_execution_cancelled()) {
			cv::Mat_<float> tmpmeta1(cv::Size(temp->meta.size(), 1)), tmpmeta2;
			std::vector<multi_img::BandDesc>::const_iterator it;
			unsigned int i;
			for (it = temp->meta.begin(), i = 0; it != temp->meta.end(); it++, i++) {
				tmpmeta1(0, i) = it->center;
			}
			cv::resize(tmpmeta1, tmpmeta2, cv::Size(newsize, 1));
			for (size_t b = 0; b < newsize; b++) {
				target->meta[b] = multi_img::BandDesc(tmpmeta2(0, b));
			}
		}

		delete temp;
		temp = NULL;

	} else {
		target = temp;
	}

	if (!includecache)
		target->resetPixels();

	if (stopper.is_group_execution_cancelled()) {
		delete target;
		return false;
	} else {
		SharedDataSwapLock lock(current->mutex);
		current->replace(target);
		return true;
	}
}


