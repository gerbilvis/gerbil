#include "representation.h"
#include <ostream>

std::ostream &operator<<(std::ostream& os, const representation::t& r)
{
	if (r < 0 || r >= representation::REPSIZE) {
		os << "INVALID";
		return os;
	}
	const char * const str[] = {
		"IMG", "NORM", "GRAD", "IMGPCA", "GRADPCA"
	};
	os << str[r];
	return os;
}
