#include "subscriptions.h"

size_t GenericHash<BandId>::operator()(const BandId &v) const {
	// No XOR here, repr and bandx are both small and likely to collide.
	return 65437 * // prime
	        std::hash<int>()(v.repr + 7) +
	        std::hash<int>()((v.band << 16) + 1031);
}
