#include "color.h"
#include <cstdlib>

namespace rbase {

Color::Color(bool isValid)
	: r(-1), g(-1), b(-1), fail(-1), isBgr(true), isValid(isValid)
{
}

Color::Color(double r, double g, double b)
	: r(r), g(g), b(b), fail(-1), isBgr(true), isValid(true)
{
}

Color::~Color()
{
}

unsigned int Color::rgbToId(unsigned char r, unsigned char g, unsigned char b) {
	unsigned long int result = ((unsigned long int) b << 16)
		+ ((unsigned long int) g << 8)
		+ ((unsigned long int)r);
	return result;
}

void Color::initRandomly() {
	r = (rand() % 256);
	g = (rand() % 256);
	b = (rand() % 256);
	isValid = true;
}

void Color::initFromId(unsigned int id) {
	r = (id >> 16) & 255;
	g = (id >>  8) & 255;
	b = id         & 255;
	isValid = true;
}

void Color::toBgr() {
	if (isBgr) return;
	isBgr = true;
}

void Color::toRgb() {
	if (!isBgr) return;
	isBgr = false;
}

double Color::sum() const
{
	return r + g + b;
}

void Color::normalize()
{
	double sum = this->sum();
	r /= sum;
	g /= sum;
	b /= sum;
}

double Color::getChannelMedian()
{
	if (r > g) {
		if (b > r) return r;
		if (b > g) return b;
		return g;
	} else {
		if (b > g) return g;
		if (b > r) return b;
		return r;
	}
}

}
