#ifndef NORMRANGETBB_H
#define NORMRANGETBB_H

#include "datarangetbb.h"

class NormRangeTbb : public DataRangeTbb {
public:
	NormRangeTbb(SharedMultiImgPtr multi,
		SharedMultiImgRangePtr  range, multi_img::NormMode mode, int target,
		multi_img::Value minval, multi_img::Value maxval, bool update,
		cv::Rect targetRoi = cv::Rect())
		: DataRangeTbb(multi, range, targetRoi),
		mode(mode), target(target), minval(minval), maxval(maxval), update(update) {}
	virtual ~NormRangeTbb() {}
	virtual bool run();
protected:
	multi_img::NormMode mode;
	int target;
	multi_img::Value minval;
	multi_img::Value maxval;
	bool update;
};

#endif // NORMRANGETBB_H
