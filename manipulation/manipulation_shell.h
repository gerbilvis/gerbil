#ifndef MANIPULATION_SHELL_H
#define MANIPULATION_SHELL_H

#include "manipulation_config.h"
#include <command.h>
#include "imginput.h"

namespace vole {

class ManipulationShell : public Command {
public:
	ManipulationShell();
	~ManipulationShell();
	int execute();
	
	void printShortHelp() const;
	void printHelp() const;

	ManipulationConfig config;
};

}

#endif
