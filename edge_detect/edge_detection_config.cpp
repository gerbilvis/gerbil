#include "edge_detection_config.h"



#ifdef WITH_BOOST
using namespace boost::program_options;
#endif

namespace vole {

namespace desc {
}

EdgeDetectionConfig::EdgeDetectionConfig(const std::string &p)
	: Config(p),
	  outputDir("out"),
	  imgInputCfg(prefix + "input"),
	  somCfg(prefix + "som")
{
#ifdef WITH_BOOST
	initBoostOptions();
#endif // WITH_BOOST
}


#ifdef WITH_BOOST
void EdgeDetectionConfig::initBoostOptions()
{
	options.add(imgInputCfg.options);
	options.add(somCfg.options);

	if (prefix_enabled)	// skip input/output options
		return;

	options.add_options()
		(key("output,O"), value(&outputDir)->default_value("/tmp/"),
		 "Output directory")
		;
}

std::string EdgeDetectionConfig::getString() const {
	std::stringstream s;
	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << "output=" << outputDir << " # Working directory" << std::endl;
	}
	s << imgInputCfg.getString();
	s << somCfg.getString();
	return s.str();
}

} // namespace vole
#endif // WITH_BOOST
