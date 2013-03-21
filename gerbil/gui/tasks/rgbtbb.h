#ifndef RGBTBB_H
#define RGBTBB_H

class RgbTbb : public MultiImg::BgrTbb {
public:
	RgbTbb(multi_img_base_ptr multi, mat3f_ptr bgr, qimage_ptr rgb,
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0))
		: MultiImg::BgrTbb(multi, bgr, targetRoi), rgb(rgb) {}
	virtual ~RgbTbb() {}
	virtual bool run();
protected:
	qimage_ptr rgb;
};

#endif // RGBTBB_H
