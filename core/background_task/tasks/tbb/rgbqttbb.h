#ifndef RGBTBB_H
#define RGBTBB_H

#include "bgrtbb.h"

class RgbTbb : public BgrTbb {
public:
	RgbTbb(SharedMultiImgPtr multi, mat3f_ptr bgr, qimage_ptr rgb)
		: BgrTbb(multi, bgr), rgb(rgb) {}
	virtual ~RgbTbb() {}
	virtual bool run();
protected:
	qimage_ptr rgb;
};

#endif // RGBTBB_H
