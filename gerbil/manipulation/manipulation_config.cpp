/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "manipulation_config.h"

#include <iostream>
#include <fstream>
#include <string>


#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

ManipulationConfig::ManipulationConfig(const std::string& prefix)
	: Config(prefix), m_InputConfig1("input1"), m_InputConfig2("input2") {

	m_strOutputFilename = "/tmp/result";

	initBoostOptions();
}

ManipulationConfig::~ManipulationConfig() {}


std::string ManipulationConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << m_InputConfig1.getString()
		  << "output=" << m_strOutputFilename << "\t# Working directory" << std::endl ;
	}
	
	return s.str();
}

#ifdef WITH_BOOST
void ManipulationConfig::initBoostOptions() {
	
	if (prefix_enabled)	// skip input/output options
		return;

	options.add(m_InputConfig1.options);
	options.add(m_InputConfig2.options);

	options.add_options()
			(key("output,O"), value(&m_strOutputFilename)->default_value(m_strOutputFilename),
			 "Output filename")
			(key("task,T"), value(&task)->default_value(std::string()),
			 "Task to perform: median, compare, divide")
			;
}
#endif // WITH_BOOST


}
