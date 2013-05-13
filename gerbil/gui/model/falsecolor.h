#ifndef FALSECOLOR_H
#define FALSECOLOR_H

#include <multi_img.h>

class FalseColor
{
public:
	FalseColor(const multi_img& img) : img(img) { }

	// Resets current true / false color representations
	void resetCaches();

	// always calls resetCaches()
	void setMultiImg(const multi_img& img);

	QImage getRgb();

	QImage getPca();

	QImage getSom();

private:
	multi_img img;
//	QImage rgbImg, pcaImg, somImg; TODO
};

#endif // FALSECOLOR_H
