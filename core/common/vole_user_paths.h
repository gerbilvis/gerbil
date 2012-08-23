#ifndef VOLE_USER_PATHS_H
#define VOLE_USER_PATHS_H

#include "vole_config.h"

namespace vole {

class UserPaths : public Config {

public:
	UserPaths();

	std::string gvole_config_root;
	std::string gvole_data_root;

	virtual std::string getString() const;

	virtual ~UserPaths() {}

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

};

}

#endif // VOLE_USER_PATHS_H
