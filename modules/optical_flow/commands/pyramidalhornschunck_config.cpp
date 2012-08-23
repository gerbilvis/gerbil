#include "pyramidalhornschunck_config.h"

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

PyramidalHornschunckConfig::PyramidalHornschunckConfig(const std::string& prefix) :
	Config(prefix)
{
	#ifdef WITH_BOOST_PROGRAM_OPTIONS
		initBoostOptions();
	#endif // WITH_BOOST_PROGRAM_OPTIONS
}

std::string PyramidalHornschunckConfig::getString() const
{
	std::stringstream s;

	/*	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {

	}
	*/
	return s.str();
}

#ifdef WITH_BOOST_PROGRAM_OPTIONS
void PyramidalHornschunckConfig::initBoostOptions()
{
	if (!prefix_enabled) {
		options.add_options()(key("input-p,P"), value(&previous)->default_value(""), "Previous image");
		options.add_options()(key("input-c,C"), value(&current)->default_value(""), "Current image");
		options.add_options()(key("alpha"), value(&alpha)->default_value(10), "Weighting of the smoothness term");
		options.add_options()(key("iterations"), value(&iterations)->default_value(100), "Max number of iterations");
		options.add_options()(key("levels"), value(&levels)->default_value(5), "Lvls");
		options.add_options()(key("scalefactor"), value(&scale)->default_value(2.0f), "ScaleFactor");
	}
}
#endif // WITH_BOOST_PROGRAM_OPTIONS

} // vole
