#ifndef FALSECOLOR_H
#define FALSECOLOR_H

#include <multi_img.h>
#include <QImage>
#include <rgb.h>

class FalseColor
{
public:
	FalseColor(const multi_img& img);

	// Resets current true / false color representations
	void resetCaches();

	// always calls resetCaches()
	void setMultiImg(const multi_img& img);

	QImage getRgb();

	QImage getPca();

	QImage getSom();

private:
	const multi_img* img;
	QImage rgbImg, pcaImg, somImg;
	gerbil::RGB rgb;
};

#endif // FALSECOLOR_H
