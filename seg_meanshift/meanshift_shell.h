#ifndef MEANSHIFT_SHELL_H
#define MEANSHIFT_SHELL_H

#include "meanshift_config.h"
#include <command.h>

namespace vole {

class MeanShiftShell : public Command {
public:
	MeanShiftShell();
	~MeanShiftShell();
	int execute();
	std::map<std::string, boost::any> execute(std::map<std::string, boost::any> &input, ProgressObserver *progress);

	void printShortHelp() const;
	void printHelp() const;

	MeanShiftConfig config;
};

}

#endif
