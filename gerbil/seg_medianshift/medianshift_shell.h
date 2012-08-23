#ifndef MEDIANSHIFT_SHELL_H
#define MEDIANSHIFT_SHELL_H

#include "medianshift_config.h"
#include <command.h>

namespace vole {

class MedianShiftShell : public Command {
public:
	MedianShiftShell();
	~MedianShiftShell();
	int execute();
	std::map<std::string, boost::any> execute(std::map<std::string, boost::any> &input, ProgressObserver *progress);

	void printShortHelp() const;
	void printHelp() const;

	MedianShiftConfig config;
};

}

#endif // MEDIANSHIFT_SHELL_H
