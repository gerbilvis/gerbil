
#include <multi_img_tasks.h>
#include "normrangecuda.h"

bool NormRangeCuda::run()
{
	switch (mode) {
	case MultiImg::NORM_OBSERVED:
		if (!MultiImg::DataRangeCuda::run())
			return false;
		break;
	case MultiImg::NORM_THEORETICAL:
		if (!stopper.is_group_execution_cancelled()) {
			SharedDataSwapLock lock(range->mutex);
			// hack!
			if (target == 0) {
				(*range)->first = (multi_img::Value)MULTI_IMG_MIN_DEFAULT;
				(*range)->second = (multi_img::Value)MULTI_IMG_MAX_DEFAULT;
			} else {
				(*range)->first = (multi_img::Value)-log(MULTI_IMG_MAX_DEFAULT);
				(*range)->second = (multi_img::Value)log(MULTI_IMG_MAX_DEFAULT);
			}
		}
		break;
	default:
		if (!stopper.is_group_execution_cancelled()) {
			SharedDataSwapLock lock(range->mutex);
			// keep previous setting
			(*range)->first = minval;
			(*range)->second = maxval;
		}
		break;
	}

	if (!stopper.is_group_execution_cancelled() && update) {
		SharedDataSwapLock lock(multi->mutex);
		(*multi)->minval = (*range)->first;
		(*multi)->maxval = (*range)->second;
		return true;
	} else {
		return false;
	}
}
