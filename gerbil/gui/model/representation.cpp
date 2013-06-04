#include "representation.h"

std::ostream &operator<<(std::ostream& os, const representation::t& r)
{
	if (r < 0 || r >= representation::REPSIZE) {
		os << "INVALID";
		return;
	}
	const char * const str[] = { "IMG", "GRAD", "IMGPCA", "GRADPCA" };
	os << str[r];
}
