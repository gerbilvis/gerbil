#include "srgb2rgb_config.h"

using namespace boost::program_options;

namespace rbase {

Srgb2RgbConfig::Srgb2RgbConfig(const std::string& prefix) :
	Config(prefix), img("img")
{
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

std::string Srgb2RgbConfig::getString() const
{
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
	}
	s << img.getString()
	  << "output_file=" << output_file
		<< " # file name of linear RGB image" << std::endl
	  << "maxTargetIntensity=" << maxTargetIntensity
	  	<< " # scale output image to this intensity (default: 255)" << std::endl
	;

	return s.str();
}

#ifdef WITH_BOOST
void Srgb2RgbConfig::initBoostOptions()
{
	if (!prefix_enabled) {
	}

	options.add(img.options);
	options.add_options()
		(key("output_file,O"), value(&output_file)->default_value(""),
			"file name of linear RGB image")
	    (key("maxTargetIntensity,M"), value(&maxTargetIntensity)->default_value(255),
	  		"scale output image to this intensity (default: 255)")
	;
}
#endif // WITH_BOOST

} // namespace lille

