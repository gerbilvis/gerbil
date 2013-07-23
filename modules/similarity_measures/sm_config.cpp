/*	
	Copyright(c) 2011 Daniel Danner,
	Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "sm_config.h"

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef WITH_BOOST
using namespace boost::program_options;
#endif

namespace vole {

ENUM_MAGIC(similarity_fun)

SMConfig::SMConfig(const std::string& prefix)
	: Config(prefix) {

	// default parameters
	measure = EUCLIDEAN;

#ifdef WITH_BOOST
	initBoostOptions();
#endif
}

std::string SMConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled)
		s << "[" << prefix << "]" << std::endl;

	s << "measure=" << measure << "\t# Measurement function" << std::endl
		 ;

	return s.str();
}

#ifdef WITH_BOOST
void SMConfig::initBoostOptions() {
	std::string measuredesc = "Similarity measurement function. Available are: ";
	measuredesc += similarity_funStr[0];
	for (unsigned int i = 1; i < sizeof(similarity_funStr)/sizeof(char*); ++i) {
		measuredesc += ", "; measuredesc += similarity_funStr[i];
	}
	options.add_options()
	    (key("measure"), value(&measure)->default_value(measure),
	     measuredesc.c_str())
	;

}
#endif // WITH_BOOST

}
