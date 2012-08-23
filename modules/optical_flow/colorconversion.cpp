#include "colorconversion.hpp"

namespace vision {

template void VISION_EXPORT RGBtoHSV(boost::uint8_t r, boost::uint8_t g,
	boost::uint8_t b, float& h, float& s, float& v, boost::uint8_t high);
template void VISION_EXPORT RGBtoHSV(boost::uint8_t r, boost::uint8_t g,
	boost::uint8_t b, double& h, double& s, double& v, boost::uint8_t high);
template void VISION_EXPORT RGBtoHSV(boost::uint8_t r, boost::uint8_t g,
	boost::uint8_t b, long double& h, long double& s, long double& v, boost::uint8_t high);

} // namespace vision
