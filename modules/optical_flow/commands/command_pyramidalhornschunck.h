#ifndef OPTICALFLOW_COMMANDS_COMMANDPYRAMIDALHORNSCHUNCK
#define OPTICALFLOW_COMMANDS_COMMANDPYRAMIDALHORNSCHUNCK

#include "command.h"
#include "pyramidalhornschunck_config.h"

namespace vole {

class CommandPyramidalHornschunck : public Command {
public:
	CommandPyramidalHornschunck();

public:
	int execute();
	void printShortHelp() const;
	void printHelp() const;

protected:
	PyramidalHornschunckConfig config;
};

} // vole

#endif // OPTICALFLOW_COMMANDS_COMMANDHORNSCHUNCK
