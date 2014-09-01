#include "subscriptions.h"

template <>
std::size_t make_hash(ImageBandId const& t)
{
	// No XOR here, repr and bandx are both small and likely to collide.
	return 65437 * // prime
			make_hash<int>(t.repr + 7) +
			make_hash<int>((t.bandx << 16) + 1031 );
}
