#ifndef VOLE_EXAMPLE_GRAHAM_SCAN_H
#define VOLE_EXAMPLE_GRAHAM_SCAN_H

#include <iostream>
#include <string>
#include "command.h"
#include "examples_config.h"

namespace vole {
	namespace examples {

// our class starts here
class GrahamScan : public vole::Command {
public:
	GrahamScan();

	int execute();

	void printShortHelp() const;
	void printHelp() const;

private:

	ExamplesConfig config;
};

	}
}

#endif // VOLE_EXAMPLE_GRAHAM_SCAN_H
