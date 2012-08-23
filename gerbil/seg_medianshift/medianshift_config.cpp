/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "medianshift_config.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace boost::program_options;

namespace vole {

MedianShiftConfig::MedianShiftConfig(const std::string& prefix)
	: Config(prefix), inputconfig("input") {
	/// set default values
	output_directory = "/tmp";
	K = 20;
	L = 10;
	k = 450;
	skipprop = false;
	signifThresh = 0.02;

	initBoostOptions();
}

MedianShiftConfig::~MedianShiftConfig() {}


std::string MedianShiftConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << inputconfig.getString()
		  << "output=" << output_directory << "\t# Working directory" << std::endl
			;
	}
	s
	  << "K=" << K << std::endl
	  << "L=" << L << std::endl
	  << "k=" << k << std::endl
	  << "signifThresh=" << signifThresh << std::endl
	  << "skipprop=" << (skipprop ? "true" : "false") << std::endl
		;
	return s.str();
}

#ifdef WITH_BOOST
void MedianShiftConfig::initBoostOptions() {
	options.add_options()
		("lsh.K", value(&K)->default_value(K),
		 "K for LSH")
		("lsh.L", value(&L)->default_value(L),
		 "L for LSH")
		("k", value(&k)->default_value(k),
		 "adaptive window size")
		("signifThresh", value(&signifThresh)->default_value(signifThresh),
		 "detect significant mode modes only, using this threshold")
		("skipprop", bool_switch(&skipprop)->default_value(skipprop),
		 "skip mode propagation step")
	;

	if (prefix_enabled)	// skip input/output options
		return;

	options.add(inputconfig.options);

	options.add_options()
		(key("output,O"), value(&output_directory)->default_value(output_directory),
		 "Working directory")
	;
}
#endif // WITH_BOOST

}
