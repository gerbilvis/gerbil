#include <shared_data.h>

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <background_task/background_task.h>
#include "normrangetbb.h"

bool NormRangeTbb::run()
{
	switch (mode) {
	case multi_img::NORM_OBSERVED:
		if (!DataRangeTbb::run())
			return false;
		break;
	case multi_img::NORM_THEORETICAL:
		if (!stopper.is_group_execution_cancelled()) {
			SharedDataSwapLock lock(range->mutex);
			// hack!
			if (target == 0) { // image
				(*range)->min = (multi_img::Value)MULTI_IMG_MIN_DEFAULT;
				(*range)->max = (multi_img::Value)MULTI_IMG_MAX_DEFAULT;
			} else { // gradient
				(*range)->min = (multi_img::Value)-log(MULTI_IMG_MAX_DEFAULT);
				(*range)->max = (multi_img::Value)log(MULTI_IMG_MAX_DEFAULT);
			}
		}
		break;
	default:
		if (!stopper.is_group_execution_cancelled()) {
			SharedDataSwapLock lock(range->mutex);
			// keep previous setting
			(*range)->min = minval;
			(*range)->max = maxval;
		}
		break;
	}

	if (!stopper.is_group_execution_cancelled() && update) {
		SharedDataSwapLock lock(multi->mutex);
		(*multi)->minval = (*range)->min;
		(*multi)->maxval = (*range)->max;
		return true;
	} else {
		return false;
	}
}
