/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "graphseg_config.h"

using namespace boost::program_options;

namespace vole {

ENUM_MAGIC(graphsegalg)

GraphSegConfig::GraphSegConfig(const std::string& p)
 : Config(p), similarity(prefix + "similarity")
#ifdef WITH_EDGE_DETECT
 , som(prefix + "som")
#endif
{
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

#ifdef WITH_BOOST
void GraphSegConfig::initBoostOptions() {
	options.add_options()
		(key("algo"), value(&algo)->default_value(WATERSHED2),
		                   "Algorithm to employ: KRUSKAL, PRIM or\n"
		                   "WATERSHED2: power watersheds with q=2")
		(key("geodesic"), bool_switch(&geodesic)->default_value(false),
		                   "Set to true to use geodesic reconstruction of the weights")
		;

	options.add(similarity.options);

#ifdef WITH_EDGE_DETECT
	options.add_options()
		(key("som_similarity"), bool_switch(&som_similarity)->default_value(false),
							 "Set to true to use SOM as input to edge weights");
	options.add(som.options);
#endif

	if (prefix_enabled)	// skip input/output options
		return;

	options.add_options()
		(key("input,I"), value(&input_file)->default_value("input.png"),
		 "Image to process")
		(key("seeds,S"), value(&seed_file)->default_value("seeds.png"),
		 "Seed map (grayscale image)")
		(key("output,O"), value(&output_file)->default_value("output_mask.png"),
		 "Output file name")
		(key("multi_seed"), bool_switch(&multi_seed)->default_value(false),
		                   "Set to true for multi-label seed file, "
		                   "false for foreground/background seeds")
		;
}
#endif // WITH_BOOST

std::string GraphSegConfig::getString() const {
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << "input=" << input_file << "\t# Image to process" << std::endl
		  << "seeds=" << seed_file << "\t# Seed map" << std::endl
		  << "output=" << output_file << "\t# Working directory" << std::endl
			;
	}
	s << "seeds_multi=" << (multi_seed ? "true" : "false") << std::endl
	  << "algo=" << algo << "\t# Algorithm to employ: KRUSKAL, PRIM, WATERSHED2" << std::endl
	  << "geodesic=" << (geodesic ? "true" : "false") << std::endl  
		;
	s << similarity.getString();
#ifdef WITH_EDGE_DETECT
	s << "som_similarity=" << (som_similarity? "true" : "false") << std::endl;
	s << som.getString();
#endif
	return s.str();
}

}
