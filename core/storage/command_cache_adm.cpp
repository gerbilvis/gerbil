/**
 *  \file   CommandCacheAdministration.cpp
 *  \brief  Example command for the superpixel segmentation by Veksler et al..
 *  \author Michael Bleier
 *  \date   2010
 */


#include "command_cache_adm.h"
#include "opencv2/highgui/highgui.hpp"


namespace vole {


/***************************************************************************/


CommandCacheAdministration::CommandCacheAdministration():
	Command("cache_adm",
		config,
		"Christian Riess",
		"christian.riess@informatik.uni-erlangen.de"
	)
{
}


/***************************************************************************/


int CommandCacheAdministration::execute()
{
	std::cout << "root directory: " << config.cache_root_dir << std::endl;
	std::cout << "cache descriptor: " << config.cache_descriptor << std::endl;

	return 0;
}


/***************************************************************************/


void CommandCacheAdministration::printShortHelp() const
{
	std::cout << "Cache administration tool" << std::endl;
}


/***************************************************************************/


void CommandCacheAdministration::printHelp() const
{
	std::cout << "Cache administration tool (FIXME Doku++)" << std::endl;
}


/***************************************************************************/


} // vole
