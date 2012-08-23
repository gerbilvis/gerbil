#ifndef VOLE_EXAMPLE_MINIMUM_ENCLOSING_RECTANGLE_H
#define VOLE_EXAMPLE_MINIMUM_ENCLOSING_RECTANGLE_H

#include <iostream>
#include <string>
#include "command.h"
#include "examples_config.h"

namespace vole {
	namespace examples {

class MinimumEnclosingRectangle : public vole::Command {
public:
	MinimumEnclosingRectangle ();

	int execute();

	void printShortHelp() const;
	void printHelp() const;

private:

	ExamplesConfig config;
};

} // namespace examples

} // namespace vole

#endif // VOLE_EXAMPLE_MINIMUM_ENCLOSING_RECTANGLE_H
