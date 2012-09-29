#include "gridseg_config.h"

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

GridSegConfig::GridSegConfig(const std::string& prefix) :
	Config(prefix)
{
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}


#ifdef WITH_BOOST
void GridSegConfig::initBoostOptions()
{
	if (!prefix_enabled) {
		options.add_options()
			(key("output,O"), value(&output_file)->default_value(""), "segmentation output image")
		;
	}

	options.add_options()
		(key("x_dim,X"), value(&x_dim)->default_value(0), "x-dimension of the segmentation (ignored if prior_segmentation is set)")
		(key("y_dim,Y"), value(&y_dim)->default_value(0), "y-dimension of the segmentation (ignored if prior_segmentation is set)")
//		(key("deterministic_coloring"), bool_switch(&deterministic_coloring)->default_value(false),
//			"output image encodes segment numbers (pro: reconstruction of segments, con: visually not appealing)")

		(key("prior_segmentation,P"), value(&prior_segmentation)->default_value(""), "Another segmentation (deterministically colored) that shall be intersected with the grid")
	  	(key("block_size,S"), value(&block_size)->default_value(-1), "side length of a block in pixels. -1: use number_blocks instead.")
	  	(key("number_blocks,B"), value(&number_blocks)->default_value(-1), "Number of blocks in y- and x-direction. -1: use block_size instead.")
		(key("fuse_max_area,F"), bool_switch(&fuse_max_area)->default_value(false), "mosaick image content, such that the maximum subarea per grid cell is spread over the whole cell")
	;
}
#endif // WITH_BOOST


std::string GridSegConfig::getString() const
{
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s  << "output=" << output_file << " # segmentation output image" << std::endl
		;
	}

	s << "x_dim=" << x_dim << " # x-dimension of the segmentation (ignored if prior_segmentation is set)" << std::endl
	  << "y_dim=" << y_dim << " # y-dimension of the segmentation (ignored if prior_segmentation is set)" << std::endl
//	  << "deterministic_coloring=" << deterministic_coloring
//		<< " # output image encodes segment numbers (pro: reconstruction of segments, con: visually not appealing)" << std::endl
	  << "block_size=" << block_size << " # side length of a block in pixels. -1: use number_blocks instead." << std::endl
	  << "number_blocks" << number_blocks << " # Number of blocks in y- and x-direction. -1: use block_size instead." << std::endl
	  << "prior_segmentation=" << prior_segmentation << " # Another segmentation (deterministically colored) that shall be intersected with the grid" << std::endl
	  << "fuse_max_area=" << (fuse_max_area ? 1 : 0) << " # mosaick image content, such that the maximum subarea per grid cell is spread over the whole cell" << std::endl
	;


	return s.str();
}
} // vole
