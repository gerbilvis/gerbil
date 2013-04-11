#ifndef RGBSERIAL_H
#define RGBSERIAL_H

class RgbSerial : public MultiImg::BgrSerial {
public:
	RgbSerial(SharedMultiImgPtr multi, mat3f_ptr bgr, qimage_ptr rgb,
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0))
		: MultiImg::BgrSerial(multi, bgr, targetRoi), rgb(rgb) {}
	virtual ~RgbSerial() {}
	virtual bool run();
protected:
	qimage_ptr rgb;
};

#endif // RGBSERIAL_H
