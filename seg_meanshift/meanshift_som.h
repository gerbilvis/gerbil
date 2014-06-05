#ifndef MEANSHIFT_SOM_H
#define MEANSHIFT_SOM_H

#include "meanshift_config.h"
#include <command.h>

namespace seg_meanshift {

class MeanShiftSOM : public shell::Command {
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
