
#include "falsecolor.h"

#include <multi_img.h>
#include <QImage>
#include <qtopencv.h>
#include <rgb.h>
#include <opencv2/highgui/highgui.hpp>

// TODO:
// Auf Img (Konstruktor) wird ein Pointer gespeichert
// -> dieses muss bestehen bleiben, bis
// a) setMultiImg mit einem anderen Img ausgefuehrt wird
// b) das FalseColor object deleted wird

// TODO: Abhaengigkeiten rgb
// Wie "with edge detect" handhaben? momentan leeres img + cerr
// leeres img wird langfristig wohl auch dann returned, wenn das img im hintergrund berechnet wird

// QImages have implicit data sharing, so the returned objects act as a pointer, the data is not copied
// (The QImages should be requested for each usage again, because they are not updated inplace,
// if resetCaches is called)

FalseColor::FalseColor(const multi_img &img) : img(&img)
{
	// TODO: init rgb.config, if non-default setup is neccessary
}

void FalseColor::resetCaches()
{
	// empty job queue

	// reset all images
	rgbImg = QImage();
	pcaImg = QImage();
	somImg = QImage();
}

void FalseColor::setMultiImg(const multi_img& img)
{
	this->img = &img;

	resetCaches();
}

QImage FalseColor::getRgb()
{
	if (rgbImg.isNull())
	{
		cv::Mat3b mat = (cv::Mat3b)(img->bgr() * 255.0f);
		rgbImg = vole::Mat2QImage(mat);
	}

	return rgbImg;
}

QImage FalseColor::getPca()
{
	if (pcaImg.isNull())
	{
		cv::Mat3b mat = (cv::Mat3b)(rgb.executePCA(*img) * 255.0f);
		pcaImg = vole::Mat2QImage(mat);
	}

	return pcaImg;
}

QImage FalseColor::getSom()
{
#ifdef WITH_EDGE_DETECT
	if (somImg.isNull())
	{
		cv::Mat3b mat = (cv::Mat3b)(rgb.executeSOM(*img) * 255.0f);
		somImg = vole::Mat2QImage(mat);
	}
#else
	// without edge detect -> somImg is always empty
	std::cerr << "ERROR: SOM functionality missing!" << std::endl;
#endif

	return somImg;
}
