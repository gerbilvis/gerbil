#ifndef NORMRANGECUDA_H
#define NORMRANGECUDA_H

#include "background_task/tasks/cuda/datarangecuda.h"

class NormRangeCuda : public DataRangeCuda {
public:
	NormRangeCuda(SharedMultiImgPtr multi,
		SharedMultiImgRangePtr range, multi_img::NormMode mode, int target,
		multi_img::Value minval, multi_img::Value maxval, bool update,
		cv::Rect targetRoi = cv::Rect())
		: DataRangeCuda(multi, range, targetRoi),
		mode(mode), target(target), minval(minval), maxval(maxval), update(update) {}
	virtual ~NormRangeCuda() {}
	virtual bool run();
protected:
	multi_img::NormMode mode;
	int target;
	multi_img::Value minval;
	multi_img::Value maxval;
	bool update;
};

#endif // NORMRANGECUDA_H
