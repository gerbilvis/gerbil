#ifndef RBASE_COLOR_H
#define RBASE_COLOR_H

namespace rbase {
class Color {
public:
	Color(bool isValid = false);
	Color(double r, double g, double b);
	virtual ~Color();

	double r;
	double g;
	double b;

	double fail;

	bool isBgr; // bgr is default order; so isBgr should almost always be true.
	bool isValid; // flag to denote whether the contained values can be used or not

	void toBgr(); // change the order of the color channels to bgr
	void toRgb(); // change the order of the color channels to bgr


	double sum() const; // compute sum of channels
	void normalize();   // scale colors: sum of channels == 1
	double getChannelMedian(); // return the middle intensity of the three channels

	static unsigned int rgbToId(unsigned char r, unsigned char g, unsigned char b);
	void initFromId(unsigned int id);

	// Assumes that srand has been called!
	void initRandomly();

	double &operator[](unsigned int index)
	{
		if (index == 0) { if (isBgr) { return b; } else { return r; } }
		if (index == 1) return g;
		if (index == 2) { if (isBgr) { return r; } else { return b; } }
		return fail;
	}

	const double &operator[](unsigned int index) const
	{
		if (index == 0) { if (isBgr) { return b; } else { return r; } }
		if (index == 1) return g;
		if (index == 2) { if (isBgr) { return r; } else { return b; } }
		return fail;
	}

	Color & operator*=(double factor)
	{
		r *= factor;
		g *= factor;
		b *= factor;
		return *this;
	}

	Color & operator/=(double factor)
	{
		r /= factor;
		g /= factor;
		b /= factor;
		return *this;
	}


	};
}

#endif // RBASE_COLOR_H

