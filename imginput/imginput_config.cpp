/*	
	Copyright(c) 2011 Daniel Danner,
	Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "imginput_config.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace boost::program_options;

namespace vole {

ImgInputConfig::ImgInputConfig(const std::string& prefix)
	: Config(prefix) {
	// set default values
	gradient = false;
	bands = 0;
	bandlow=0;
	bandhigh=0;

	initBoostOptions();
}

ImgInputConfig::~ImgInputConfig() {}


std::string ImgInputConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled)
		s << "[" << prefix << "]" << std::endl;

	s << "file=" << file << "\t# Image to process" << std::endl
	  << "roi=" << roi << std::endl
	  << "gradient=" << (gradient ? "true" : "false") << std::endl
	  << "bands=" << bands << std::endl
	  << "bandlow=" << bandlow << std::endl
	  << "bandhigh=" << bandhigh << std::endl
		 ;

	return s.str();
}

#ifdef WITH_BOOST
void ImgInputConfig::initBoostOptions() {
	options.add_options()
		(key("file"), value(&file)->default_value(file),
		 "Image to process")
		(key("roi"), value(&roi)->default_value(roi),
		 "apply ROI (x:y:w:h)")
		(key("gradient"), bool_switch(&gradient)->default_value(gradient),
		 "compute spectral gradient")
		(key("bands"), value(&bands)->default_value(bands),
		 "reduce number of bands by linear interpolation (0 means disabled)")
		(key("bandlow"), value(&bandlow)->default_value(bandlow),
		 "apply lower bound of band ROI")
   		(key("bandhigh"), value(&bandhigh)->default_value(bandhigh),
		 "apply upper bound of band ROI")
	;

}
#endif // WITH_BOOST

}
