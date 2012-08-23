#ifndef VOLE_CACHE_CONFIG_H
#define VOLE_CACHE_CONFIG_H


#include "vole_config.h"


namespace vole {


/***************************************************************************/

class CacheConfig : public Config {
public:
	CacheConfig(const std::string& prefix = std::string());

public:
	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST_PROGRAM_OPTIONS
		virtual void initBoostOptions();
	#endif // WITH_BOOST_PROGRAM_OPTIONS

public:
	std::string cache_root_dir;
	std::string cache_descriptor;
};


/***************************************************************************/


} // vole

#endif // VOLE_CACHE_CONFIG_H
