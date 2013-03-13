#include "fullimageswitcher.h"

bool FullImageSwitcher::run()
{
	SharedDataSwapLock regLock(regular->lock);
	SharedDataSwapLock limLock(limited->lock);
	multi_img *regPtr = regular->swap(NULL);
	multi_img_base *limPtr = limited->swap(NULL);
	assert(regPtr == NULL || limPtr == NULL);
	switch (target) {
	case REGULAR:
		if (regPtr != NULL)
			regular->swap(regPtr);
		else
			regular->swap(dynamic_cast<multi_img *>(limPtr));
		break;
	case LIMITED:
		limited->swap(limPtr == NULL ? regPtr : limPtr);
		break;
	default:
		assert(false);
	}
	return true;
}
