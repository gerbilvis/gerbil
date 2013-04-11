#ifndef NORMRANGETBB_H
#define NORMRANGETBB_H

class NormRangeTbb : public MultiImg::DataRangeTbb {
public:
	NormRangeTbb(SharedMultiImgPtr multi,
		data_range_ptr range, MultiImg::NormMode mode, int target,
		multi_img::Value minval, multi_img::Value maxval, bool update,
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0))
		: MultiImg::DataRangeTbb(multi, range, targetRoi),
		mode(mode), target(target), minval(minval), maxval(maxval), update(update) {}
	virtual ~NormRangeTbb() {}
	virtual bool run();
protected:
	MultiImg::NormMode mode;
	int target;
	multi_img::Value minval;
	multi_img::Value maxval;
	bool update;
};

#endif // NORMRANGETBB_H
