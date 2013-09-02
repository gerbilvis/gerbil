
#include "normrangecuda.h"

bool NormRangeCuda::run()
{
	switch (mode) {
	case multi_img::NORM_OBSERVED:
		if (!DataRangeCuda::run())
			return false;
		break;
	case multi_img::NORM_THEORETICAL:
		if (!stopper.is_group_execution_cancelled()) {
			SharedDataSwapLock lock(range->mutex);
			// hack!
			if (target == 0) {
				(*range)->min = (multi_img::Value)MULTI_IMG_MIN_DEFAULT;
				(*range)->max = (multi_img::Value)MULTI_IMG_MAX_DEFAULT;
			} else {
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
