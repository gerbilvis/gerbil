#ifndef FELZENSZWALB_SHELL_H
#define FELZENSZWALB_SHELL_H

#include "felzenszwalb_config.h"
#include <command.h>

namespace seg_felzenszwalb {

class FelzenszwalbShell : public shell::Command {
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
