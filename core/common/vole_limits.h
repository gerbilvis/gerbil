#ifndef VOLE_LIMITS_H
#define VOLE_LIMITS_H

#include <climits>
#include <cfloat>
#include "vole_exception.h"

class VoleLimits {
public:
	template <typename T>
	static T max() {
		throw new VoleException("Data Type not implemented in VoleLimits!");
		return 0;
	}

	template <typename T>
	static T min() {
		throw new VoleException("Data Type not implemented in VoleLimits!");
		return 0;
	}


// FIXME: Rausfinden, wann long double definiert ist (? ISOC99 ?)
};

#endif // VOLE_LIMITS_H
