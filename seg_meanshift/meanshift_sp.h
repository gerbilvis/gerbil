#ifndef MEANSHIFT_SP_H
#define MEANSHIFT_SP_H

#include "meanshift_config.h"
#include <command.h>

namespace vole {

class MeanShiftSP : public Command {
public:
	MeanShiftSP();
	~MeanShiftSP();
	int execute();

	void printShortHelp() const;
	void printHelp() const;

	MeanShiftConfig config;
};

}

#endif
