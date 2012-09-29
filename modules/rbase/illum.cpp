#include "illum.h"
#include <limits>
#include <cmath>


namespace illumestimators {

Illum::Illum(bool isValid)
	: Color(isValid)
{
}

Illum::Illum(double r, double g, double b)
	: Color(r, g, b)
{
}

Illum::~Illum()
{
}

double Illum::angularErrorDeg(Illum &other)
{
	double angErrRad = angularErrorRad(other);
	if (angErrRad < 0) return -1;

	return (angErrRad * 180) / M_PI;
}

double Illum::angularErrorRad(Illum &other)
{
	if ((r < 0) || (g < 0) || (b < 0)) return -1;
	if ((other.r < 0) || (other.g < 0) || (other.b < 0)) return -1;

	double estNorm = std::sqrt(r*r + g*g + b*b);
	double otherNorm = std::sqrt(other.r*other.r + other.g*other.g + other.b*other.b);

	if ((estNorm < std::numeric_limits<double>::epsilon()) || (otherNorm < std::numeric_limits<double>::epsilon())) return -1;

	Illum normEst(r/estNorm, g/estNorm, b/estNorm);
	Illum normOther(other.r/otherNorm, other.g/otherNorm, other.b/otherNorm);

	double innerProduct = normEst.r * normOther.r + normEst.g * normOther.g + normEst.b * normOther.b;

	if (innerProduct > 1) innerProduct = 1;

	return std::acos(innerProduct);
}

double Illum::rmsError(Illum &other)
{
	if ((r < 0) || (g < 0) || (b < 0)) return -1;

	double sumEst = r + g + b;
	double sumOther = r + g + b;

	Illum estNorm(r/sumEst, g/sumEst, b/sumEst);
	Illum otherNorm(other.r/sumOther, other.g/sumOther, other.b/sumOther);

	double squared =
		  (estNorm.r-otherNorm.r) * (estNorm.r-otherNorm.r)
		+ (estNorm.g-otherNorm.g) * (estNorm.g-otherNorm.g)
		+ (estNorm.b-otherNorm.b) * (estNorm.b-otherNorm.b);

	return sqrt(squared / 3);
}

double Illum::rgError(Illum &other)
{
	if ((r < 0) || (g < 0) || (b < 0)) return -1;

	double sumEstimate = r + g + b;
	double sumOther = other.r + other.g + other.b;

	double squared = 
		  (r/sumEstimate) - (other.r/sumOther) * (r/sumEstimate) - (other.r/sumOther)
		+ (g/sumEstimate) - (other.g/sumOther) * (g/sumEstimate) - (other.g/sumOther);

	return sqrt(squared);
}

double Illum::rError(Illum &other)
{
	if ((r < 0) || (g < 0) || (b < 0)) return -1;

	double sumEstimate = r + g + b;
	double sumOther = other.r + other.g + other.b;

	return (r / sumEstimate) - (other.r / sumOther);
}

double Illum::gError(Illum &other)
{
	if ((r < 0) || (g < 0) || (b < 0)) return -1;

	double sumEstimate = r + g + b;
	double sumOther = other.r + other.g + other.b;

	return (g / sumEstimate) - (other.g / sumOther);
}

}

