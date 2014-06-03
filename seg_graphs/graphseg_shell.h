#ifndef GRAPHSEG_SHELL_H
#define GRAPHSEG_SHELL_H

#include "graphseg_config.h"
#include <command.h>

namespace seg_graphs {

class GraphSegShell : public shell::Command {
public:
	GraphSegShell();
	int execute();

	void printShortHelp() const;
	void printHelp() const;

protected:
	GraphSegConfig config;
};

}

#endif
