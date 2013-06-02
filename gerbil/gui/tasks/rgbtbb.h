#ifndef RGBTBB_H
#define RGBTBB_H

class RgbTbb : public MultiImg::BgrTbb {
public:
	RgbTbb(SharedMultiImgPtr multi, mat3f_ptr bgr, qimage_ptr rgb,
		cv::Rect targetRoi = cv::Rect())
		: MultiImg::BgrTbb(multi, bgr, targetRoi), rgb(rgb) {}
	virtual ~RgbTbb() {}
	virtual bool run();
protected:
	qimage_ptr rgb;
};

#endif // RGBTBB_H
