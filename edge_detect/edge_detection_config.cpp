#include "edge_detection_config.h"

#ifdef WITH_BOOST
using namespace boost::program_options;
#endif

namespace vole {

EdgeDetectionConfig::EdgeDetectionConfig(const std::string &p)
	: Config(p),
	  absolute(false),
	  outputDir("/tmp/"),
	  imgInputCfg(prefix + "input"),
	  somCfg(prefix + "som")
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
	options.add_options()
		BOOST_OPT(absolute)
		;
	options.add(somCfg.options);

	if (prefix_enabled)	// skip input/output options
		return;

	options.add(imgInputCfg.options);
	options.add_options()
		BOOST_OPT_S(outputDir, O)
		;
}

std::string EdgeDetectionConfig::getString() const {
	std::stringstream s;
	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << imgInputCfg.getString();
		COMMENT_OPT(s, outputDir);
	}
	s << somCfg.getString();
	return s.str();
}

} // namespace vole
#endif // WITH_BOOST
