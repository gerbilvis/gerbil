#ifndef OPTICALFLOW_COMMANDS_COMMANDHORNSCHUNCK
#define OPTICALFLOW_COMMANDS_COMMANDHORNSCHUNCK

#include "command.h"
#include "hornschunck_config.h"

namespace vole {

class CommandHornschunck : public Command {
public:
	CommandHornschunck();

public:
	int execute();
	void printShortHelp() const;
	void printHelp() const;

protected:
	HornschunckConfig config;
};

} // vole

#endif // OPTICALFLOW_COMMANDS_COMMANDHORNSCHUNCK
