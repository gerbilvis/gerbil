/**
 *  \file   command_cache_adm.h
 *  \brief  Convenience command to clear the cache; could currently also be performed manuelly.
 *  \author Christian Riess
 *  \date   2011
 */


#ifndef VOLE_COMMAND_CACHE_ADMINISTRATION
#define VOLE_COMMAND_CACHE_ADMINISTRATION


#include "command.h"
#include "vole_cache_config.h"


namespace vole {


/***************************************************************************/


class CommandCacheAdministration : public Command {
public:
	CommandCacheAdministration();

public:
	int execute();
	void printShortHelp() const;
	void printHelp() const;

protected:
	CacheConfig config;
};


/***************************************************************************/


} // vole


#endif // VOLE_COMMAND_CACHE_ADMINISTRATION
