#ifndef MULTI_IMG_VIEWER_CURPOS_H
#define MULTI_IMG_VIEWER_CURPOS_H

#include <multi_img.h>

/* translate image value to value in our coordinate system */
static inline multi_img::Value curpos(
	const multi_img::Value& val, int dim,
	const multi_img::Value& minval, const multi_img::Value& binsize,
	const std::vector<multi_img::Value> &illuminant)
{
	multi_img::Value curpos = (val - minval) / binsize;
	if (!illuminant.empty())
		curpos /= illuminant[dim];
	return curpos;
}

#endif // MULTI_IMG_VIEWER_CURPOS_H
