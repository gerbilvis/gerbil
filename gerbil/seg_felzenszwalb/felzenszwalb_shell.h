#ifndef FELZENSZWALB_SHELL_H
#define FELZENSZWALB_SHELL_H

#include "felzenszwalb2_config.h"
#include <command.h>

namespace gerbil {

class FelzenszwalbShell : public vole::Command {
public:
	FelzenszwalbShell();
	int execute();

	void printShortHelp() const;
	void printHelp() const;

protected:
	FelzenszwalbConfig config;
};

}

#endif
