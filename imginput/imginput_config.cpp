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

namespace imginput {

ImgInputConfig::ImgInputConfig(const std::string& prefix)
	: Config(prefix),
      normalize(false), gradient(false),
      removeIllum(0), addIllum(0), bands(0), bandlow(0), bandhigh(0),
      output("/tmp/image")
{
#ifdef WITH_BOOST
	initBoostOptions();
#endif // WITH_BOOST
}

ImgInputConfig::~ImgInputConfig() {}

// descriptions of configuration options
namespace desc {
DESC_OPT(file, "Input image filename")
DESC_OPT(roi, "Region of Interest, specify in the form x:y:w:h to apply")
DESC_OPT(normalize, "Normalize vector magnitudes")
DESC_OPT(gradient, "Transform to spectral gradient")
DESC_OPT(bands, "Reduce number of bands by linear interpolation (if >0)")
DESC_OPT(bandlow, "Select bands and use band index as lower bound (if >0)")
DESC_OPT(bandhigh, "Select bands and use band index as upper bound (if >0)")
DESC_OPT(removeIllum, "Remove black body illuminant specified in Kelvin (if >0)")
DESC_OPT(addIllum, "Add black body illuminant specified in Kelvin (if >0)")
DESC_OPT(output, "Basename of output descriptor file and directory")
}

std::string ImgInputConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled)
		s << "[" << prefix << "]" << std::endl;

	COMMENT_OPT(s, file);
	COMMENT_OPT(s, roi);
	COMMENT_OPT(s, normalize);
	COMMENT_OPT(s, gradient);
	COMMENT_OPT(s, bands);
	COMMENT_OPT(s, bandlow);
	COMMENT_OPT(s, bandhigh);
	COMMENT_OPT(s, removeIllum);
	COMMENT_OPT(s, addIllum);

	return s.str();
}

#ifdef WITH_BOOST_PROGRAM_OPTIONS
void ImgInputConfig::initBoostOptions() {
	options.add_options()
			BOOST_OPT(file)
			BOOST_OPT(roi)
			BOOST_BOOL(normalize)
			BOOST_BOOL(gradient)
			BOOST_OPT(bands)
			BOOST_OPT(bandlow)
			BOOST_OPT(bandhigh)
			BOOST_OPT(removeIllum)
			BOOST_OPT(addIllum)
	;
	if (!prefix_enabled) {
		options.add_options()BOOST_OPT_S(output,O);
	}
}
#endif // WITH_BOOST_PROGRAM_OPTIONS

}
