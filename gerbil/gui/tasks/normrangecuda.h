#ifndef NORMRANGECUDA_H
#define NORMRANGECUDA_H

class NormRangeCuda : public MultiImg::DataRangeCuda {
public:
	NormRangeCuda(SharedMultiImgPtr multi,
		SharedMultiImgRangePtr range, MultiImg::NormMode mode, int target,
		multi_img::Value minval, multi_img::Value maxval, bool update,
		cv::Rect targetRoi = cv::Rect())
		: MultiImg::DataRangeCuda(multi, range, targetRoi),
		mode(mode), target(target), minval(minval), maxval(maxval), update(update) {}
	virtual ~NormRangeCuda() {}
	virtual bool run();
protected:
	MultiImg::NormMode mode;
	int target;
	multi_img::Value minval;
	multi_img::Value maxval;
	bool update;
};

#endif // NORMRANGECUDA_H
