#ifndef VOLE_CREATE_NEW_COMMAND_H
#define VOLE_CREATE_NEW_COMMAND_H

#include "command.h"
#include "create_new_command_config.h"

namespace vole {


class CreateNewCommand : public Command {
public:
	CreateNewCommand();

	virtual ~CreateNewCommand() {};

	int doCreation();
	int manageUserInteraction();

	virtual int execute();
	virtual void printShortHelp() const;
	virtual void printHelp() const;

protected:
	CreateNewCommandConfig config;

};

}

#endif // VOLE_CREATE_NEW_COMMAND_H
