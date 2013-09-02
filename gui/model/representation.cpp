#include "representation.h"

std::ostream &operator<<(std::ostream& os, const representation::t& r)
{
	if (r < 0 || r >= REPSIZE) {
		os << "INVALID";
		return os;
	}
	const char * const str[] = { "IMG", "GRAD", "IMGPCA", "GRADPCA" };
	os << str[r];
	return os;
}
