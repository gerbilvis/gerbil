#ifndef NORMRANGECUDA_H
#define NORMRANGECUDA_H

#ifdef GERBIL_CUDA

#include "background_task/tasks/cuda/datarangecuda.h"

class NormRangeCuda : public DataRangeCuda {
public:
	NormRangeCuda(SharedMultiImgPtr multi,
		SharedMultiImgRangePtr range, multi_img::NormMode mode, int target,
		multi_img::Value minval, multi_img::Value maxval, bool update)
		: DataRangeCuda(multi, range),
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

#endif

#endif // NORMRANGECUDA_H
