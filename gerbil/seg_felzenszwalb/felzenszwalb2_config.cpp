/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "felzenszwalb2_config.h"

using namespace boost::program_options;

namespace gerbil {

FelzenszwalbConfig::FelzenszwalbConfig(const std::string& p)
 : vole::Config(p), similarity(prefix + "similarity")
{
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

#ifdef WITH_BOOST
void FelzenszwalbConfig::initBoostOptions() {
	options.add_options()
		(key("tc,c"), value(&c)->default_value(500),
						   "Threshold value c within sim. measure range")
		(key("min-size"), value(&min_size)->default_value(50),
						   "Minimum size of a superpixel")
		(key("eqhist"), bool_switch(&eqhist)->default_value(false),
							"Perform histogram equalization on edge weights")
		;

	options.add(similarity.options);

	if (prefix_enabled)	// skip input/output options
		return;

	options.add_options()
		(key("input,I"), value(&input_file)->default_value("input.png"),
		 "Image to process")
		(key("output,O"), value(&output_file)->default_value("output.png"),
		 "Output file name")
		;
}
#endif // WITH_BOOST

std::string FelzenszwalbConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << "input=" << input_file << "\t# Image to process" << std::endl
		  << "output=" << output_file << "\t# Working directory" << std::endl
			;
	}
	s << "c=" << c << std::endl
	  << "min-size=" << min_size
	  << "eqhist=" << (eqhist ? "true" : "false") << std::endl
		;
	s << similarity.getString();
	return s.str();
}

}
