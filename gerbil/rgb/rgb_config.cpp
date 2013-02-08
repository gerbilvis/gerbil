/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "rgb_config.h"

using namespace boost::program_options;

namespace gerbil {

ENUM_MAGIC(rgbalg)

RGBConfig::RGBConfig(const std::string& p)
 : vole::Config(p), input("input")
#ifdef WITH_EDGE_DETECT
 , som(prefix + "som")
#endif
{
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

#ifdef WITH_BOOST
void RGBConfig::initBoostOptions() {
	options.add_options()
		(key("algo"), value(&algo)->default_value(COLOR_XYZ),
		                   "Algorithm to employ: XYZ true color,"
		                   "PCA or SOM false-color")
		(key("somDepth"), value(&som_depth)->default_value(10),
		                   "In SOM case: "
		                   "number of best matching neurons to incorporate")
		;

#ifdef WITH_EDGE_DETECT
	options.add(som.options);
#endif

	if (prefix_enabled)	// skip input/output options
		return;

	options.add(input.options);

	options.add_options()
		(key("output,O"), value(&output_file)->default_value("output_mask.png"),
		 "Output file name")
		;
}
#endif // WITH_BOOST

std::string RGBConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << input.getString()
		  << "output=" << output_file << "\t# Working directory" << std::endl
			;
	}
	s	<< "algo=" << algo << "\t# Algorithm" << std::endl
		<< "somDepth=" << som_depth << "\t# SOM depth" << std::endl
		;
#ifdef WITH_EDGE_DETECT
	s << som.getString();
#endif
	return s.str();
}

}
