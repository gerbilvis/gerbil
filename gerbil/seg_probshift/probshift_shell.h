#ifndef PROBSHIFT_SHELL_H
#define PROBSHIFT_SHELL_H

#include "probshift_config.h"
#include <command.h>

namespace vole {

class ProbShiftShell : public Command {
public:
	ProbShiftShell();
	~ProbShiftShell();
	int execute();
	std::map<std::string, boost::any> execute(std::map<std::string, boost::any> &input, ProgressObserver *progress);

	void printShortHelp() const;
	void printHelp() const;

	ProbShiftConfig config;
};

}

#endif // PROBSHIFT_SHELL_H
