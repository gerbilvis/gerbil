#include "edge_detection_config.h"

#ifdef WITH_BOOST
using namespace boost::program_options;
#endif

namespace edge_detect {

EdgeDetectionConfig::EdgeDetectionConfig(const std::string &p)
	: Config(p),
	  absolute(false),
	  outputDir("/tmp/"),
	  input(prefix + "input"),
	  som(prefix + "som")
{
#ifdef WITH_BOOST
	initBoostOptions();
#endif // WITH_BOOST
}

// descriptions of configuration options
namespace desc {
DESC_OPT(absolute, "Write out absolute edge map (no sign)")
DESC_OPT(outputDir, "Directory to store dx and dy edge maps")
}

#ifdef WITH_BOOST
void EdgeDetectionConfig::initBoostOptions()
{
	if (!prefix_enabled) { // input/output options only with prefix
		options.add(input.options);
		options.add_options()
			BOOST_OPT_S(outputDir, O)
			;
	}
	options.add_options()
		BOOST_OPT(absolute)
		;
	options.add(som.options);
}

std::string EdgeDetectionConfig::getString() const {
	std::stringstream s;
	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << input.getString();
		COMMENT_OPT(s, outputDir);
	}
	s << som.getString();
	return s.str();
}

} // module namespace
#endif // WITH_BOOST
