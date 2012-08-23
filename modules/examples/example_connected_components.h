#ifndef VOLE_EXAMPLE_CONNECTED_COMPONENTS_H
#define VOLE_EXAMPLE_CONNECTED_COMPONENTS_H

#include <iostream>
#include <string>
#include "command.h"
#include "examples_config.h"

namespace vole {
	/** ready-to-use classes for quick tests on vole modules
	 */
	namespace examples {

// our class starts here
class ConnectedComponents : public vole::Command {
public:
	ConnectedComponents();

	int execute();

	void printShortHelp() const;
	void printHelp() const;

private:

	ExamplesConfig config;
};

	}
}

#endif // VOLE_EXAMPLE_CONNECTED_COMPONENTS_H
