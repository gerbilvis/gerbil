#ifndef MEANSHIFT_SOM_H
#define MEANSHIFT_SOM_H

#include "meanshift_config.h"
#include <command.h>

namespace vole {

class MeanShiftSOM : public Command {
public:
	MeanShiftSOM();
	~MeanShiftSOM();
	int execute();

	void printShortHelp() const;
	void printHelp() const;

	MeanShiftConfig config;
};

}

#endif
