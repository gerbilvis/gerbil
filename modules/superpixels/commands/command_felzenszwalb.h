#ifndef SUPERPIXELS_COMMANDS_COMMANDFELZENSZWALB
#define SUPERPIXELS_COMMANDS_COMMANDFELZENSZWALB

#include "command.h"
#include "felzenszwalb_config.h"

namespace vole {

class CommandFelzenszwalb : public Command {
public:
	CommandFelzenszwalb();

public:
	int execute();
	void printShortHelp() const;
	void printHelp() const;

protected:
	FelzenszwalbConfig config;
};

} // vole

#endif // SUPERPIXELS_COMMANDS_COMMANDFELZENSZWALB
