
#include "falsecolor.h"

#include <multi_img.h>
#include <qtopencv.h>
#include <rgb.h>

#include <QImage>
#include <opencv2/core/core.hpp>

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
	for (int i = 0; i < COLSIZE; ++i) {
		payload *p = new payload;
#ifndef WITH_EDGE_DETECT
		if (i == SOM)
			continue;
#endif
		map.insert((coloring)i, p);
	}

	// TODO: init rgb.config, if non-default setup is neccessary
	map.value(CMF)->rgb.config.algo = gerbil::COLOR_XYZ;
	map.value(PCA)->rgb.config.algo = gerbil::COLOR_PCA;
#ifdef WITH_EDGE_DETECT
	map.value(SOM)->rgb.config.algo = gerbil::COLOR_SOM;
#endif
}

FalseColor::~FalseColor()
{
	PayloadList l = map.values();
	foreach(payload *p, l) {
		delete p;
	}
}


void FalseColor::resetCaches()
{
	// empty job queue

	// reset all images
	PayloadList l = map.values();
	foreach(payload *p, l) {
		p->img = QImage();
		/* TODO: maybe send the empty image as signal to
		 * disable viewing of obsolete information
		 */
	}
}

void FalseColor::setMultiImg(const multi_img& img)
{
	this->img = &img;

	resetCaches();
}

QImage FalseColor::get(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL);
	if (p->img.isNull()) {
		cv::Mat3b mat = (cv::Mat3b)(p->rgb.execute(*img) * 255.0f);
		p->img = vole::Mat2QImage(mat);
	}
	return p->img;
}
