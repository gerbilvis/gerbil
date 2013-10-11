/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "meanshift_config.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace boost::program_options;

namespace vole {

ENUM_MAGIC(ms_sampling)

MeanShiftConfig::MeanShiftConfig(const std::string& prefix)
	: Config(prefix), input("input")
#ifdef WITH_SEG_FELZENSZWALB
	, superpixel("superpixel"),
	sp_withGrad(false), sp_weight(0)
#endif
{
	use_LSH = false;
	K = 20;
	L = 10;
	seed = 0;
	k = 1.f;
	starting = ALL;
	jump = 2;
	percent = 50;
	bandwidth = 0;
	Kmin = 1;
	Kjump = 1;
	epsilon = 0.05f;
	pruneMinN = 50;

	output_directory = "/tmp";
	batch = false;
	findKL = false;

	initBoostOptions();
}

MeanShiftConfig::~MeanShiftConfig() {}


std::string MeanShiftConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << input.getString()
		  << "output=" << output_directory << "\t# Working directory" << std::endl
		  << "prefix=" << output_prefix << "\t# Output file prefix" << std::endl
		  << "doFindKL=" << (findKL ? "true" : "false") << std::endl
		  << "Kmin=" << Kmin << std::endl
		  << "Kjump=" << Kjump << std::endl
		  << "epsilon=" << epsilon << std::endl
			;
	}
#ifdef WITH_SEG_FELZENSZWALB
	s << superpixel.getString();
	s << "sp_withGrad=" << (sp_withGrad ? "true" : "false") << std::endl;
	s << "sp_weight=" << sp_weight << std::endl;
#endif
	s << "useLSH=" << (use_LSH ? "true" : "false") << std::endl
	  << "K=" << K << std::endl
	  << "L=" << L << std::endl
	  << "seed=" << seed << std::endl
	  << "pilotk=" << k << std::endl
	  << "initmethod=" << starting << std::endl
	  << "initjump=" << jump << std::endl
	  << "initpercent=" << percent << std::endl
	  << "bandwidth=" << bandwidth << std::endl
		;
	return s.str();
}

#ifdef WITH_BOOST
void MeanShiftConfig::initBoostOptions() {
	options.add_options()
			("useLSH", bool_switch(&use_LSH)->default_value(use_LSH),
			 "use locality-sensitive hashing")
			("lshK", value(&K)->default_value(K),
			 "K for LSH")
			("lshL", value(&L)->default_value(L),
			 "L for LSH")
			("seed", value(&seed)->default_value(seed),
			 "random seed (0 means time-based)")
			("pilotk", value(&k)->default_value(k),
			 "number of neighbors used in the construction of the pilot density")
			("initmethod", value(&starting)->default_value(starting),
			 "start mean shift from ALL points, every Xth point (JUMP), "
#ifdef WITH_SEG_FELZENSZWALB
			 "a random selection of points (PERCENT), or SUPERPIXEL segmentation")
#else
			 "or a random selection of points (PERCENT)")
#endif
#ifdef WITH_SEG_FELZENSZWALB
			("sp_withGrad", bool_switch(&sp_withGrad)->default_value
																  (sp_withGrad),
			 "compute gradient as input to mean shift step (after superpixels)")
			("sp_weight", value(&sp_weight)->default_value(sp_weight),
			 "how to weight superpixel sizes: 0 do not weight, 1 fixed bandwidths,"
			 " 2 alter weight")
#endif
			("initjump", value(&jump)->default_value(jump),
			 "use points with indices 1+(jump*[1..infty])")
			("initpercent", value(&percent)->default_value(percent),
			 "randomly select given percentage of points")
			("bandwidth", value(&bandwidth)->default_value(bandwidth),
			 "use fixed bandwidth*dimensionality for mean shift window (else: adaptive)")

	;
#ifdef WITH_SEG_FELZENSZWALB
	options.add(superpixel.options);
#endif

	if (prefix_enabled)	// skip input/output options
		return;

	options.add(input.options);

	options.add_options()
			(key("output,O"), value(&output_directory)->default_value(output_directory),
			 "Working directory")
			(key("prefix"), value(&output_prefix)->default_value(output_prefix),
			 "Prefix to all output filenames")
			("batchmode", bool_switch(&batch)->default_value(batch),
			 "write out the total coverage (FALSE) or only label (index) image (TRUE)")
			("doFindKL", bool_switch(&findKL)->default_value(findKL),
			 "empirically determine optimal K, L values (1 < L < lsh.L)")
			("Kmin", value(&Kmin)->default_value(Kmin),
			 "minimum value of K to be tested (findKL only)")
			("Kjump", value(&Kjump)->default_value(Kjump),
			 "test every Kmin:Kjump:K values for K (findKL only)")
			("epsilon", value(&epsilon)->default_value(epsilon),
			 "error threshold (findKL only)")
	;
}
#endif // WITH_BOOST

}
