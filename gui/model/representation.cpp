#include "representation.h"

std::ostream &operator<<(std::ostream& os, const representation::t& r)
{
	if (r < 0 || r >= representation::REPSIZE) {
		os << "INVALID";
		return os;
	}
	const char * const str[] = {
		"IMG"
	#ifdef WITH_IMGNORM
		, "NORM"
	#endif
		, "GRAD"
	#ifdef WITH_IMGPCA
		, "IMGPCA"
	#endif
	#ifdef WITH_GRADPCA
		, "GRADPCA"
	#endif
	};
	os << str[r];
	return os;
}
