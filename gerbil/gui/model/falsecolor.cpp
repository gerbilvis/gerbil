
#include "falsecolor.h"

#include <multi_img.h>
#include <qtopencv.h>

void FalseColor::resetCaches()
{
	// TODO
}

void FalseColor::setMultiImg(const multi_img& img)
{
	this->img = img;

	resetCaches();
}

QImage FalseColor::getRgb()
{
	return QImage(); // TODO
}

QImage FalseColor::getPca()
{
	return QImage(); // TODO
}

QImage FalseColor::getSom()
{
	return QImage(); // TODO
}
