#ifndef ILLUMINANTESTIMATORS_ILLUM_H
#define ILLUMINANTESTIMATORS_ILLUM_H

#include "color.h"

namespace illumestimators {
class Illum : public rbase::Color {
public:

	Illum(bool isValid = false);
	Illum(double r, double g, double b);
	virtual ~Illum();

	void toBgr();
	void toRgb();
		
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

	Illum & operator*=(double factor)
	{
		r *= factor;
		g *= factor;
		b *= factor;
		return *this;
	}

	Illum & operator/=(double factor)
	{
		r /= factor;
		g /= factor;
		b /= factor;
		return *this;
	}

	// Error functions
	double angularErrorDeg(Illum &otherEst);
	double angularErrorRad(Illum &otherEst);
	double rgError(Illum &otherEst);
	double rError(Illum &otherEst);
	double gError(Illum &otherEst);
	double rmsError(Illum &otherEst);
};

}

#endif // ILLUMINANTESTIMATORS_ILLUM_ESTIMATE_H
