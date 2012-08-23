/**
 *  \file   vole_cache_config.cpp
 *  \brief  Config for cache management
 *  \author Christian Riess
 *  \date   2011
 */

#include "vole_cache_config.h"

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {


/***************************************************************************/


CacheConfig::CacheConfig(const std::string& prefix) :
	Config(prefix)
{
	#ifdef WITH_BOOST_PROGRAM_OPTIONS
		initBoostOptions();
	#endif // WITH_BOOST_PROGRAM_OPTIONS
}


/***************************************************************************/

#ifdef WITH_BOOST_PROGRAM_OPTIONS
void CacheConfig::initBoostOptions()
{
	if (!prefix_enabled) {
//		options.add_options()(key("input,I"), value(&inputFile)->default_value("input.png"), "Image to process");
//		options.add_options()(key("output,O"), value(&output_file)->default_value(""), "segmentation output image");
	}

	options.add_options()(key("cache_root"), value(&cache_root_dir)->default_value(""), "root directory of the cache");
	options.add_options()(key("cache_descriptor"), value(&cache_descriptor)->default_value(""), "descriptor for the particular cache to operate on");
}
#endif // WITH_BOOST_PROGRAM_OPTIONS

/***************************************************************************/


std::string CacheConfig::getString() const
{
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
//		s << "input=" << inputFile << " # Image to process" << std::endl;
//		s << "output=" << output_file << " # segmentation output image" << std::endl;
	}

	s << "cache_root=" << cache_root_dir << " # root directory of the cache" << std::endl;
	s << "cache_descriptor=" << cache_descriptor << " # descriptor for the particular cache to operate on" << std::endl;

	return s.str();
}


/***************************************************************************/



} // vole
