/*	
	Copyright(c) 2011 Daniel Danner,
	Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "probshift_config.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace boost::program_options;

namespace vole {

ProbShiftConfig::ProbShiftConfig(const std::string& prefix)
	: Config(prefix), inputconfig("input") {
	/// set default values
	output_directory = "/tmp";
	useLSH = false;
	lshK = 10;
	lshL = 20;
	useSpectral = false;
	minClusts = 2;
	maxClusts = 10;
	useConverged = false;

    useMeanShift = false;
    msClusts = -1;
    msBwFactor = 4;


	initBoostOptions();
}

ProbShiftConfig::~ProbShiftConfig() {}


std::string ProbShiftConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << inputconfig.getString()
		  << "output=" << output_directory << "\t# Output directory" << std::endl
			;
	}
	s
	  << "lsh=" << (useLSH ? "true" : "false") << std::endl
	  << "K=" << lshK << std::endl
	  << "L=" << lshL << std::endl
	  << "spectral=" << (useSpectral ? "true" : "false") << std::endl
	  << "useConv=" << (useConverged ? "true" : "false") << std::endl
	  << "minClusts=" << minClusts << std::endl
	  << "maxClusts=" << maxClusts << std::endl
	  << "useMeanShift=" << (useMeanShift ? "true" : "false") << std::endl
	  << "msClusts=" << msClusts << std::endl
	  << "msBwFactor=" << msBwFactor << std::endl
	  << "saveModes=" << saveModes << std::endl
	  << "loadModes=" << loadModes << std::endl
	  ;
	return s.str();
}

#ifdef WITH_BOOST
void ProbShiftConfig::initBoostOptions() {
	options.add_options()
		("lsh", bool_switch(&useLSH)->default_value(useLSH),
		 "use locality-sensitive hashing")
		("lshK", value(&lshK)->default_value(lshK),
		 "K for LSH")
		("lshL", value(&lshL)->default_value(lshL),
		 "L for LSH")
		("spectral", bool_switch(&useSpectral)->default_value(useSpectral),
		 "use unsupervised spectral partitioning (default is row-maximum)")
		("useConv", bool_switch(&useConverged)->default_value(useConverged),
		 "use converged TPM (spectral only)")
		("minClusts", value(&minClusts)->default_value(minClusts),
		 "minimum number of clusters (spectral only)")
		("maxClusts", value(&maxClusts)->default_value(maxClusts),
		 "maximum number of clusters (spectral only)")
		("useMeanShift", bool_switch(&useMeanShift)->default_value(useMeanShift),
		 "use Meanshift to merge modes in post-processing")
		("msClusts", value(&msClusts)->default_value(msClusts),
		 "scale bandwidths to get a fixed number of clusters when post-processing with Meanshift")
		("msBwFactor", value(&msBwFactor)->default_value(msBwFactor),
		 "adjustment factor for bandwidth values")
		("saveModes", value(&saveModes)->default_value(saveModes),
		 "save mode assignments to file (pre-meanshift)")
		("loadModes", value(&loadModes)->default_value(loadModes),
		 "load mode assignments from file (jumps directly to post-processing)")
	;

	if (prefix_enabled)	// skip input/output options
		return;

	options.add(inputconfig.options);

	options.add_options()
			(key("output,O"), value(&output_directory)->default_value(output_directory),
			 "Output directory")
			;
}
#endif // WITH_BOOST

}
